#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 14:52:04 2023

@author: Nikila Swaminathan

Fine-tuning a language model for depression detection and generating text based on the trained model. 
Let's break down the main components and functionalities:

Imports: The script imports various libraries and modules required for different tasks such as data 
handling, model training, and inference. Notable imports include trl, peft, transformers, datasets, and huggingface_hub.

Authentication: The script prompts the user to authenticate with the Hugging Face API by providing an API token.

Dataset Loading and Preprocessing:
The script loads a dataset named "nikilas/DepressionIncludingImageText" using the load_dataset function from the datasets library.
It renames the column 'text' to 'old_text'.
It defines a custom function combine_text_and_image_text to combine text and image descriptions.
It applies the custom function to create a new column named 'text'.

Dataset Splitting: The script splits the dataset into training, validation, and test sets using the train_test_split method.

Model Fine-Tuning: The script sets various parameters for fine-tuning the model, including QLoRA parameters, 
bitsandbytes parameters, and training arguments.
It loads a base language model (Mistral-7B-Instruct-v0.1) with QLoRA configuration.
It fine-tunes the base model using the SFTTrainer from the trl library.
Inference with Fine-Tuned Model: The script tests the performance of the fine-tuned model on some 
depression detection tasks by generating text based on the trained model.

Merge and Share: After fine-tuning, the script optionally merges the model with LoRA weights and saves the 
merged model for sharing on the Hugging Face Model Hub.

Testing the Merged Model: The script tests the merged model by generating text based on a random sample 
from the training dataset and comparing it with the ground truth label.

Overall, the script automates the process of loading a dataset, preprocessing it, fine-tuning a 
language model for a specific task, and evaluating the performance of the trained model.
"""

''' 
#Pre-requistes
!pip install -q torch
!pip install -q git+https://github.com/huggingface/transformers #huggingface transformers for downloading models weights
!pip install -q datasets #huggingface datasets to download and manipulate datasets
!pip install -q peft #Parameter efficient finetuning - for qLora Finetuning
!pip install -q bitsandbytes #For Model weights quantisation
!pip install -q trl #Transformer Reinforcement Learning - For Finetuning using Supervised Fine-tuning
!pip install -q wandb -U #Used to monitor the model score during training
'''

import pandas as pd
from huggingface_hub import notebook_login
import torch
import datasets
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

####################
###### Main ########
####################

log_file_name = "my_log_file_4.txt"

# Manually authenticate with the Hugging Face API
#print("Please authenticate with the Hugging Face API")
#notebook_login(input("Enter your API token: "))

# Load dataset
raw_datasets = load_dataset("nikilas/DepressionIncludingImageText", use_auth_token=True)
#print(raw_datasets.keys())
#can use the multimodal_dataset.csv from the data folder in the github

# Rename column 'text' to 'old_text'
raw_datasets = raw_datasets.rename_column('text', 'old_text')

# Define a custom function to combine 'old_text' and 'image2text' columns
def combine_text_and_image_text(example):
    eval_prompt = """
Social Media Post:

{0}

Image Description:

{1}

Analysis Request:
Question: Based on the content of the above social media post and the associated image description, are there discernible signs or indicators of depression?
Required Response Format: Please provide a 'yes' or 'no' answer, followed by a brief explanation of the reasoning behind this conclusion.
""".format(example['old_text'],example['image2text'])
    return {'text': eval_prompt}

# Use the .map() method to create a new column 'text' to hold teh combined text
raw_datasets = raw_datasets.map(combine_text_and_image_text)

# Print the keys to see the new column
#print(raw_datasets.keys())

# Remove columns 'old_text' and 'image2text'
columns_to_remove = ['old_text', 'image2text']
raw_datasets = raw_datasets.remove_columns(columns_to_remove)

print("Training before split ",len(raw_datasets['train']))

#Split the dataset into training, validation abd test datasets
train_testvalid = raw_datasets['train'].train_test_split(train_size=0.7, seed=42)

# Split the 30% (test + valid) in half test, half valid
test_valid = train_testvalid['test'].train_test_split(train_size=0.5, seed=42)

# Combine splits into a single DatasetDict
raw_datasets = datasets.DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'val': test_valid['train']})

print("Splits available in the DatasetDict:", raw_datasets.keys())

'''# Iterate over each split and print its details
for split in raw_datasets.keys():
  print ({split: len(raw_datasets[split]) })
  # Print the structure of the Dataset
  print(raw_datasets[split])

# Print the first example to see the new column's value
print(raw_datasets['train'][0]['text'])
print(raw_datasets['train'][0]['label'])
'''

df_train = pd.DataFrame(raw_datasets['train'])
#print(df_train.describe)

# Calculate the total number of samples in the training dataset
total_samples = len(df_train)

# Count the number of samples in the minority class (class with label 1)
minority_class_count = df_train['label'].value_counts()[0]

# Count the number of samples in the majority class (class with label 0)
majority_class_count = df_train['label'].value_counts()[1]

# Calculate the weight for the minority class
# This is done by dividing the total number of samples by twice the number of samples in the minority class.
# The idea is to increase the weight of the minority class so that the model pays more attention to it.
pos_weights = total_samples / (2 * minority_class_count)

# Similarly, calculate the weight for the majority class
# Here, the weight is usually smaller as the class already has a higher representation in the dataset.
neg_weights = total_samples / (2 * majority_class_count)

# Display the calculated weights
#Postive is not depressed
#Negative is depressesed state
print ("POS_WEIGHT, NEG_WEIGHT =",pos_weights,neg_weights)
###Setting Model Parameters
#Setting various parameters for the fine-tuning process, including 
#QLoRA (Quantization LoRA) parameters, bitsandbytes parameters, and training arguments

new_model = "mistralai-depression-detect" #set the name of the new model

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
#lora_r = 64
lora_r = 16

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
#lora_dropout = 0.1
lora_dropout = 0.05

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
#bnb_4bit_compute_dtype = torch.float16
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
#use_nested_quant = False
use_nested_quant = True

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "Mistral-Seq-classification-QLoRa"

# Number of training epochs
num_train_epochs = 1

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 4

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4

# Number of update steps to accumulate the gradients for
#gradient_accumulation_steps = 1
gradient_accumulation_steps = 4

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
#learning_rate = 2e-4
learning_rate = 4e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
#optim = "paged_adamw_32bit"
optim = "paged_adamw_8bit"

# Learning rate schedule (constant a bit better than cosine)
#lr_scheduler_type = "constant"
lr_scheduler_type = "linear"

# Number of training steps (overrides num_train_epochs)
#max_steps = -1
max_steps = 1000

# Number of warmup steps
#warmup_steps = 0
warmup_steps = 100

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
#save_steps = 25
save_steps = 20

# Log every X updates steps
logging_steps = 25

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
#max_seq_length = None
max_seq_length = 512

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}

################################################################################
#Loading the Mistral 7B Instruct base model
################################################################################
model_name = "mistralai/Mistral-7B-v0.1"
#model_name = "mistralai/Mistral-7B-Instruct-v0.1"
#model_name = "mistralai/mixtral-8x7b-instruct-v0.1"
#cannot load as it needs 28GB GPU RAM with 19 shrads

# Load the base model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)


base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0}
)
base_model = prepare_model_for_kbit_training(base_model)
'''
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)'''

# Load MitsralAi tokenizer
#tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id =  tokenizer.unk_token_id
#tokenizer.padding_side = "right"
tokenizer.padding_side = 'left'

base_model.config.use_cache = False
#base_model.config.pretraining_tp = 1
base_model.config.pad_token_id =  tokenizer.pad_token_id

# Release memory
torch.cuda.empty_cache()

################################################################################
#Base model Inference
################################################################################
# import random
model_input = tokenizer(raw_datasets['train'][0]['text'], 
                        return_tensors="pt").to("cuda")

base_model.eval()
with torch.no_grad():
  print(tokenizer.decode(
      base_model.generate(
          **model_input, 
          max_new_tokens=256, 
          pad_token_id=2)[0], 
      skip_special_tokens=True))

# Release memory
torch.cuda.empty_cache()

################################################################################
#Fine-Tuning with qLora and Supervised Finetuning
#Fine-tuning our model using qLora. We used SFTTrainer from the trl library 
#for supervised fine-tuning.
################################################################################
# Set LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=4000, # the total number of training steps to perform
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="wandb"
)
#Before you execute this step, sign into https://wandb.ai/authorize and get an API key for the following step.
import wandb
# Manually authenticate with the Weights & Biases
print("Please authenticate with the Weights & Biases:")
wandb.login(input("Enter your API token: "))

wandb.init(project=output_dir)

# Initialize the SFTTrainer for fine-tuning
trainer = SFTTrainer(
    model=base_model,
    train_dataset=raw_datasets['train'],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,  # You can specify the maximum sequence length here
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

################################################################################
#Starting the training process
################################################################################
# Start the training process
trainer.train()

# Save the fine-tuned model
trainer.model.save_pretrained(new_model)

################################################################################
#Inference with Fine-Tuned Model
#Now that we have our fine-tuned model, let's test the performance of fine-tuned 
#model with some depression detection tasks.
################################################################################

model_input = tokenizer(raw_datasets['train'][0]['text'], return_tensors="pt").to("cuda")
new_model.eval()
with torch.no_grad():
    generated_text = tokenizer.decode(
        new_model.generate(
            **model_input, 
            max_new_tokens=256, 
            pad_token_id=2)[0], 
        skip_special_tokens=True)
print(generated_text)

# Release memory
torch.cuda.empty_cache()
################################################################################
#Merge and Share
#After fine-tuning, if you want to merge the model with LoRA weights or share it with the 
#Hugging Face Model Hub, you can do so. This step is optional and depends on your specific use case.
################################################################################

# Merge the model with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)
merged_model= PeftModel.from_pretrained(base_model, new_model)
merged_model= new_model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained("merged_model",safe_serialization=True)
tokenizer.save_pretrained("merged_model")

# Merge the model with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)
merged_model= PeftModel.from_pretrained(base_model, new_model)
merged_model= merged_model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained("merged_model",safe_serialization=True)
tokenizer.save_pretrained("merged_model")

#Test the merged model
from random import randrange

sample = raw_datasets['train'] [randrange(len(raw_datasets['train']))]
prompt = sample['text']

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = merged_model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9,temperature=0.5)

print(f"Prompt:\n{prompt}\n")
print(f"\nGenerated text:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")
print(f"\nGround truth:\n{sample['label']}")
