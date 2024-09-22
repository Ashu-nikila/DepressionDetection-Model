#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 23:09:12 2024

@author: Nikila Swaminathan

This code is aimed at loading a pretrained language model from the Hugging Face model hub and using it for inference on a given 
dataset related to mental health. The main objective is to assess the model's performance in identifying signs of depression and 
stress based on textual and image data.

1. Load Dependencies:
   - The code starts by importing necessary libraries and modules, such as sklearn.metrics, huggingface_hub, datasets, 
   and pandas.

2. Load Dataset:
   - The code loads the dataset related to mental health from the Hugging Face datasets library using the provided dataset ID. 
   It filters the dataset based on classes such as 'Stress' and 'Depression' and splits it into training, validation, and test sets.

3. Preprocessing and Data Exploration:
   - It then converts the datasets into pandas DataFrames for further analysis. Basic statistics and class distributions are 
   displayed to understand the dataset better.

4. Model Evaluation:
   - The code proceeds to evaluate the pretrained language model's performance on the validation dataset. It iterates over 
   different models and generates predictions for each sample in the dataset.
   - For each model, it assesses accuracy, precision, recall, and F1-score. It also evaluates the model's performance after 
   removing instances where the model failed to provide a response or identify depression/stress.
   - Confusion matrices and classification reports are generated to analyze the model's performance visually.

5. Visualization:
   - The code includes visualization components such as class distribution plots and confusion matrix heatmaps to aid in result 
   interpretation.

6. Additional Functions:
   - Several functions are defined to handle text processing, model evaluation, and result interpretation.
   - Functions like `depr_fn_new`, `stress_fn_new`, `check_depression`, and `check_stress` are created to perform specific tasks 
   such as checking for signs of depression and stress in textual data.

7. Logging and Output:
   - The code logs intermediate results and generated texts to a specified log file for reference.
   - Finally, it prints and logs the evaluation results for each model, along with confusion matrices and classification reports.

Note: 
- The code assumes access to a pretrained language model from the Hugging Face model hub and a specific dataset related to mental health.
- It utilizes external resources such as the Hugging Face model hub, replicate client, and potentially paid APIs for accessing models.
- Certain functions like `evaluate_model`, `annotate_heatmap` are defined in the codebase.
- Results and interpretations are based on the provided dataset and model evaluations.
"""

#MentalLLM models loading
#https://huggingface.co/NEU-HAI/mental-alpaca
#and inferencing it for our data set. we are not removing thes top words or do any other preprocessing; though we trim it to 450 words.
#Outcome : Seems not expected, the model is not responding for more than 50% and even if it recsponds accracy is lower.

#!pip -q install datasets

# Importing necessary libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from huggingface_hub import notebook_login
from datasets import load_dataset
import datasets
import pandas as pd
import replicate
import os
import re
import matplotlib.pyplot as plt
#from getpass import getpass
from sklearn.metrics import confusion_matrix, classification_report

# Define a function to evaluate model predictions
def evaluate_model(predictions, labels):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return accuracy, precision, recall, f1

# Define a function to check if the generated text indicates depression
def check_depression(local_generated_text, local_text_message, log_file):
    
    # Processing the generated text
    # Remove the input text from the generated text
    if local_generated_text.startswith(local_text_message):
        processed_text = local_generated_text[len(local_text_message):].strip()
    else:
        processed_text = local_generated_text
    # Remove leading whitespace and special characters; also if there is a word called "answer"
    trimmed_text = re.sub(r'^\s*answer:\s*|^\s*\W+', '', processed_text, flags=re.IGNORECASE)

    if not bool(trimmed_text):
        is_depressed = -2
        log_file.write(f"\nIs the poster depressed? {is_depressed}")
        return is_depressed
    # Remove specific words/phrases
    trimmed_text = re.sub(r'Answer: |Response: ', '', trimmed_text)
    # Trim the word (removing leading and trailing whitespace and non-word characters)
    trimmed_text = re.sub(r'^[\s\W]+|[\s\W]+$', '', trimmed_text)
    # Extract the first word
    first_word = trimmed_text.split()[0].lower()
    first_word_trimmed = re.sub(r'^[\s\W]+|[\s\W]+$', '', first_word)
    # Check if the first word is 'yes' or 'no'
    is_yes = first_word_trimmed == "yes"
    is_no = first_word_trimmed == "no"

    if is_yes or is_no:
        is_depressed = 1 if is_yes else 0
    else:
        log_file.write(f"\nKeyword not found in {trimmed_text}")
        is_depressed = -1

    log_file.write(f"\nIs the poster depressed? {is_depressed}")
    return is_depressed

# Define a function to check if the generated text indicates stress
def check_stress(local_generated_text, local_text_message, log_file):
    
    # Remove the input text from the generated text
    if local_generated_text.startswith(local_text_message):
        processed_text = local_generated_text[len(local_text_message):].strip()
    else:
        processed_text = local_generated_text
    # Remove leading whitespace and special characters; also if there is a word called "answer"
    trimmed_text = re.sub(r'^\s*answer:\s*|^\s*\W+', '', processed_text, flags=re.IGNORECASE)

    if not bool(trimmed_text):
        is_stressed = -2
        log_file.write(f"\nIs the poster stressed? {is_stressed}")
        return is_stressed
    # Remove specific words/phrases
    trimmed_text = re.sub(r'Answer: |Response: ', '', trimmed_text)
    # Trim the word (removing leading and trailing whitespace and non-word characters)
    trimmed_text = re.sub(r'^[\s\W]+|[\s\W]+$', '', trimmed_text)
    # Extract the first word
    first_word = trimmed_text.split()[0].lower()
    first_word_trimmed = re.sub(r'^[\s\W]+|[\s\W]+$', '', first_word)
    # Check if the first word is 'yes' or 'no'
    is_yes = first_word_trimmed == "yes"
    is_no = first_word_trimmed == "no"

    if is_yes or is_no:
        is_stressed = 1 if is_yes else 0
    else:
        log_file.write(f"\nKeyword not found in {trimmed_text}")
        is_stressed = -1

    log_file.write(f"\nIs the poster stressed? {is_stressed}")
    return is_stressed

# Define a function to run the depression model
def depr_fn_new(modelname, post_text, image2text, log_file):
    prompt_text = """
Social Media Post:

{0}

Image Description:

{1}

Analysis Request:
Question: Based on the content of the above social media post and the associated image description, are there discernible signs or indicators of depression?
Required Response Format: Please provide a 'yes' or 'no' answer, followed by a brief explanation of the reasoning behind this conclusion.
""".format(post_text, image2text)

    output = replicate.run(
        modelname,
        input={
            "prompt": prompt_text,
            "max_length": 200,
            "agree_to_research_only": True
            }
    )

    generated_text = ""
    for item in output:
        generated_text += item

    log_file.write(f"generated_text: {generated_text}\n")

    is_depressed = check_depression(generated_text, prompt_text, log_file)
    return generated_text, is_depressed

# Define a function to run the stress model
def stress_fn_new(modelname, post_text, image2text, log_file):
    prompt_text = """
Social Media Post:

{0}

Image Description:

{1}

Analysis Request:
Question: Based on the content of the above social media post and the associated image description, are there discernible signs or indicators of stress?
Required Response Format: Please provide a 'yes' or 'no' answer, followed by a brief explanation of the reasoning behind this conclusion.
""".format(post_text, image2text)

    output = replicate.run(
        modelname,
        input={
            "prompt": prompt_text,
            "max_length": 200,
            "agree_to_research_only": True
            }
    )

    generated_text = ""
    for item in output:
        generated_text += item

    log_file.write(f"generated_text: {generated_text}\n")

    is_stressed = check_stress(generated_text, prompt_text, log_file)
    return generated_text, is_stressed

# Save confusion matrix plots as figures
def save_confusion_matrix_plot(cm, title, file_name):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                xticklabels=["Not Depressed", "Depressed"],
                yticklabels=["Not Depressed", "Depressed"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    
# Save confusion matrix plots as figures
def save_stress_confusion_matrix_plot(cm, title, file_name):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                xticklabels=["Not Stressed", "Stressed"],
                yticklabels=["Not Stressed", "Stressed"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    
####Main starts####

# Authenticate with Hugging Face Hub
# Manually authenticate with the Hugging Face API
print("Please authenticate with the Hugging Face API:")
notebook_login(input("Enter your API token: "))
#token: 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

# Load the dataset
raw_datasets = load_dataset("nikilas/DepressionIncludingImageText", use_auth_token=True)
#print(raw_datasets.keys())

# Filter the dataset for stress and depression classes
stress_dataset = raw_datasets.filter(lambda example: example['class'] == 'Stress')
#print("Training before split ",len(stress_dataset['train']))

# Split the stress dataset into training, validation, and test sets
train_testvalid = stress_dataset['train'].train_test_split(train_size=0.7, seed=42)
# Split the 30% (test + valid) into half test, half valid
test_valid = train_testvalid['test'].train_test_split(train_size=0.5, seed=42)
stress_dataset = datasets.DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'val': test_valid['train']})
#print("Splits available in the DatasetDict:", stress_dataset.keys())
'''
# Iterate over each split and print its details
for split in stress_dataset.keys():
  print ({split: len(stress_dataset[split]) })
  # Print the structure of the Dataset
  print(stress_dataset[split])
'''

# Filter the dataset based on a column's value
#IDD2018, Kaggle, SDCNL
depression_dataset = raw_datasets.filter(lambda example: example['dataset'] == 'SDCNL' or example['dataset'] == 'IDD2018')
#print("Training before split ",len(depression_dataset['train']))
# Split the depression dataset into training, validation, and test sets
train_testvalid = depression_dataset['train'].train_test_split(train_size=0.7, seed=42)
# Split the 30% (test + valid) into half test, half valid
test_valid = train_testvalid['test'].train_test_split(train_size=0.5, seed=42)
depression_dataset = datasets.DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'val': test_valid['train']})
'''
# Iterate over each split and print its details
for split in depression_dataset.keys():
  print ({split: len(depression_dataset[split]) })
  # Print the structure of the Dataset
  print(depression_dataset[split])
'''

# Visualize the Val data of depression dataset
# Convert the datasets to pandas DataFrames
#train_df = pd.DataFrame(depression_dataset['train'])
#test_df = pd.DataFrame(depression_dataset['test'])
val_df = pd.DataFrame(depression_dataset['val'])

df = val_df

print("Depression dataset - val")
# Display the number of samples and features
num_samples, num_features = df.shape
print(f'Number of samples: {num_samples}')
print(f'Number of features: {num_features}')

# Display basic statistics of the dataset
print('\nSummary Statistics:')
print(df.describe())

# Display class distribution for binary classification
class_distribution = df['label'].value_counts()
print('\nClass Distribution:')
print(class_distribution)

# Visualize the class distribution
plt.figure(figsize=(8, 4))
df['label'].value_counts().plot(kind='bar', color='skyblue')
plt.xlabel('Classes')
plt.ylabel('Count')
plt.title('Class Distribution')
# Save the plot as an image file
plot_file_path = "class_distribution_plot-Depression-LLMs.png"
plt.savefig(plot_file_path)
plt.show()

# Visualize the Val data of stress dataset
# Convert the datasets to pandas DataFrames
#train_df = pd.DataFrame(depression_dataset['train'])
#test_df = pd.DataFrame(depression_dataset['test'])
val_df = pd.DataFrame(stress_dataset['val'])

df = val_df
print("Stress dataset - val")
# Display the number of samples and features
num_samples, num_features = df.shape
print(f'Number of samples: {num_samples}')
print(f'Number of features: {num_features}')

# Display basic statistics of the dataset
print('\nSummary Statistics:')
print(df.describe())

# Display class distribution for binary classification
class_distribution = df['label'].value_counts()
print('\nClass Distribution:')
print(class_distribution)

# Visualize the class distribution
plt.figure(figsize=(8, 4))
df['label'].value_counts().plot(kind='bar', color='skyblue')
plt.xlabel('Classes')
plt.ylabel('Count')
plt.title('Class Distribution')
# Save the plot as an image file
plot_file_path = "class_distribution_plot-Stress-LLMs.png"
plt.savefig(plot_file_path)
plt.show()

######Run the models for the whole dataset######

# get a token: https://replicate.com/account
# install replicate client using "!pip install replicate"
#REPLICATE_API_TOKEN = getpass()
REPLICATE_API_TOKEN ="r8_Nv2gM1ybgolpjgWqEi8uwGZsILNqxVU0ALtb3"
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
#token for replicate
#r8_IIHG7hbkTN0No13c5QcwxqKPIQYidDu1kZitY(paid)
#r8_Nv2gM1ybgolpjgWqEi8uwGZsILNqxVU0ALtb3(paid)

# Define the name of the log file
log_file_name = "/content/my_log_file_new.txt"


#### Depression ####

# Select a subset of the evaluation dataset
evaluation_data = depression_dataset['val'].select(list(range(128)))

# Display label information
#print(evaluation_data['label'])
df = evaluation_data

print("Depression dataset - val")
# Display the number of samples and features
num_samples, num_features = df.shape
print(f'Number of samples: {num_samples}')
print(f'Number of features: {num_features}')

# Display basic statistics of the dataset
print('\nSummary Statistics:')
print(df.describe())

# Display class distribution for binary classification
class_distribution = df['label'].value_counts()
print('\nClass Distribution:')
print(class_distribution)

# Visualize the class distribution
plt.figure(figsize=(8, 4))
df['label'].value_counts().plot(kind='bar', color='skyblue')
plt.xlabel('Classes')
plt.ylabel('Count')
plt.title('Class Distribution')
# Save the plot as an image file
plot_file_path = "class_distribution_plot-Depression-LLMs_1.png"
plt.savefig(plot_file_path)
plt.show()

df = pd.DataFrame(columns=['input_text', 'image2text', 'label', 'gen_text','intermediate_prediction'])

results = {}

# Models to evaluate
models = ["mistralai/mixtral-8x7b-instruct-v0.1:7b3212fbaf88310cfef07a061ce94224e82efc8403c26fc67e8f6c065de51f21",
          "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
          "replicate/flan-t5-xl:3ae0799123a1fe11f8c89fd99632f843fc5f7a761630160521c4253149754523",
          "replicate/oasst-sft-1-pythia-12b:28d1875590308642710f46489b97632c0df55adb5078d43064e1dc0bf68117c3",
          "replicate/dolly-v2-12b:ef0e1aefc61f8e096ebe4db6b2bacc297daf2ef6899f0f7e001ec445893500e5",
          "replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b",
          "replicate/gpt-j-6b:b3546aeec6c9891f0dd9929c2d3bedbf013c12e02e7dd0346af09c37e008c827"]

model_names = ["mixtral","llama","flan","oasst","dolly","vicuna","gpt"]

# Open the log file in append mode
with open(log_file_name, "a") as log_file:
  for index_model, modelname in enumerate(models):
    predictions = []
    predictions_only_identified = []
    temp_predictions_1_2 = []
    indices_to_remove = []
    log_file.write(f"\nModel name: {modelname}")
    print (modelname)
    df = pd.DataFrame()

    for index, (input_text, image2text,label) in enumerate(zip(evaluation_data['text'], evaluation_data['image2text'], evaluation_data['label'])):
      # Process model_output to get the prediction of depression
      model_generated_text,prediction = depr_fn_new(modelname,input_text,image2text,log_file)

      # Append a new row to the DataFrame
      df = df.append({'input_text': input_text,
                    'image2text': image2text,
                    'label': label,
                    'gen_text': model_generated_text,
                    'intermediate_prediction':prediction},
                    ignore_index=True)

      if prediction == -1 or prediction == -2:
        # Write to the log file instead of printing
        log_file.write(f"\nProcessing text number: {index + 1}; Keyword not found in generated text or no response for the Input post: {input_text}\n")
        indices_to_remove.append(index)
        temp_predictions_1_2.append(prediction)
        if label == 0:
          predictions.append(1)
        else:
          predictions.append(0)
      else:
        # Write to the log file instead of printing
        log_file.write(f"\nProcessing text number: {index + 1}; Input text: {input_text}; Is the poster depressed? {prediction}; Actual: {evaluation_data[index]['label']}\n")
        predictions.append(prediction)
        predictions_only_identified.append(prediction)

      # Print the loop counter (index)
      print(f"Processing text number: {index + 1}, prediction: {prediction}, actual {evaluation_data[index]['label']}")

    # Filter out the unwanted indices
    temp_eval_data = evaluation_data.filter(
        lambda example,
        idx: idx not in indices_to_remove, with_indices=True)

    accuracy, precision, recall, f1 = evaluate_model(predictions, evaluation_data['label'])
    results[model_names[index_model]] = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    accuracy, precision, recall, f1 = evaluate_model(predictions_only_identified, temp_eval_data['label'])
    modelname_temp = model_names[index_model] + "_manipulated"
    results[modelname_temp] = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    print(predictions)
    print(temp_eval_data['label'])
    print(temp_predictions_1_2.count(-1),temp_predictions_1_2.count(-2),temp_predictions_1_2)
    print(evaluation_data['label'])
    log_file.write(f"\nPerformance of {model_names[index_model]}; accuracy: {accuracy}; precision: {precision}; recall: {recall}; f1: {f1}\n")
    log_file.write(f"\nPredictions of {model_names[index_model]}; {predictions}\n")
    log_file.write(f"\nEval data of {model_names[index_model]}; {evaluation_data['label']}\n")
    log_file.write(f"\nTemp eval data of {model_names[index_model]}; {temp_eval_data['label']}\n")
    log_file.write(f"\nCount of -1, Count of -2, temp_predictions; {temp_predictions_1_2.count(-1)},{temp_predictions_1_2.count(-2)},{temp_predictions_1_2}\n")

    cm = confusion_matrix(evaluation_data['label'],predictions)
    report = classification_report(evaluation_data['label'],predictions)
    print(cm)
    print(report)
    log_file.write(f"\nCM of {model_names[index_model]}; {cm}\n")
    log_file.write(f"\nCM report of {model_names[index_model]}; {report}\n")

    cm = confusion_matrix(temp_eval_data['label'],predictions_only_identified)
    report = classification_report(temp_eval_data['label'],predictions_only_identified)
    print(cm)
    print(report)
    log_file.write(f"\nCM of manipulated {model_names[index_model]}; {cm}\n")
    log_file.write(f"\nCM report of manipulated {model_names[index_model]}; {report}\n")
    # Save confusion matrix plots for depression
    plot_filename = 'confusion_matrix_depression'+ model_names[index_model]+'.png'
    save_confusion_matrix_plot(cm, 'Confusion Matrix - Depression', plot_filename) 

    csvfilename = 'dep_intermediate_gentext_'+model_names[index_model]+'.csv'
    df.to_csv(csvfilename, index=False)
    print (csvfilename)

  print(results)
  log_file.write(f"\nResults of {model_names[index_model]}; {results}\n")

#### Stress ####

# Select a subset of the evaluation dataset
evaluation_data = stress_dataset['val'].select(list(range(128)))

# Display label information
#print(evaluation_data['label'])
df = evaluation_data

print("Stress dataset - val")
# Display the number of samples and features
num_samples, num_features = df.shape
print(f'Number of samples: {num_samples}')
print(f'Number of features: {num_features}')

# Display basic statistics of the dataset
print('\nSummary Statistics:')
print(df.describe())

# Display class distribution for binary classification
class_distribution = df['label'].value_counts()
print('\nClass Distribution:')
print(class_distribution)

# Visualize the class distribution (optional, requires matplotlib)
plt.figure(figsize=(8, 4))
df['label'].value_counts().plot(kind='bar', color='skyblue')
plt.xlabel('Classes')
plt.ylabel('Count')
plt.title('Class Distribution')
# Save the plot as an image file
plot_file_path = "class_distribution_plot-Stress-LLMs_1.png"
plt.savefig(plot_file_path)
plt.show()

df = pd.DataFrame(columns=['input_text', 'image2text', 'label', 'gen_text','intermediate_prediction'])

results = {}

# Models to evaluate
models = ["mistralai/mixtral-8x7b-instruct-v0.1:7b3212fbaf88310cfef07a061ce94224e82efc8403c26fc67e8f6c065de51f21",
          "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
          "replicate/flan-t5-xl:3ae0799123a1fe11f8c89fd99632f843fc5f7a761630160521c4253149754523",
          "replicate/oasst-sft-1-pythia-12b:28d1875590308642710f46489b97632c0df55adb5078d43064e1dc0bf68117c3",
          "replicate/dolly-v2-12b:ef0e1aefc61f8e096ebe4db6b2bacc297daf2ef6899f0f7e001ec445893500e5",
          "replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b",
          "replicate/gpt-j-6b:b3546aeec6c9891f0dd9929c2d3bedbf013c12e02e7dd0346af09c37e008c827"]

model_names = ["mixtral","llama","flan","oasst","dolly","vicuna","gpt"]

# Open the log file in append mode
with open(log_file_name, "a") as log_file:
  for index_model, modelname in enumerate(models):
    predictions = []
    predictions_only_identified = []
    temp_predictions_1_2 = []
    indices_to_remove = []
    log_file.write(f"\nModel name: {modelname}")
    print (modelname)
    df = pd.DataFrame()

    for index, (input_text, image2text,label) in enumerate(zip(evaluation_data['text'], evaluation_data['image2text'], evaluation_data['label'])):
      # Process model_output to get the prediction of depression
      model_generated_text,prediction = stress_fn_new(modelname,input_text,image2text,log_file)

      # Append a new row to the DataFrame
      df = df.append({'input_text': input_text,
                    'image2text': image2text,
                    'label': label,
                    'gen_text': model_generated_text,
                    'intermediate_prediction':prediction},
                    ignore_index=True)

      if prediction == -1 or prediction == -2:
        # Write to the log file instead of printing
        log_file.write(f"\nProcessing text number: {index + 1}; Keyword not found in generated text or no response for the Input post: {input_text}\n")
        indices_to_remove.append(index)
        temp_predictions_1_2.append(prediction)
        if label == 0:
          predictions.append(1)
        else:
          predictions.append(0)
      else:
        # Write to the log file instead of printing
        log_file.write(f"\nProcessing text number: {index + 1}; Input text: {input_text}; Is the poster depressed? {prediction}; Actual: {evaluation_data[index]['label']}\n")
        predictions.append(prediction)
        predictions_only_identified.append(prediction)

      # Print the loop counter (index)
      print(f"Processing text number: {index + 1}, prediction: {prediction}, actual {evaluation_data[index]['label']}")

    # Filter out the unwanted indices
    temp_eval_data = evaluation_data.filter(
        lambda example,
        idx: idx not in indices_to_remove, with_indices=True)

    accuracy, precision, recall, f1 = evaluate_model(predictions, evaluation_data['label'])
    results[model_names[index_model]] = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    accuracy, precision, recall, f1 = evaluate_model(predictions_only_identified, temp_eval_data['label'])
    modelname_temp = model_names[index_model] + "_manipulated"
    results[modelname_temp] = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    print(predictions)
    print(temp_eval_data['label'])
    print(temp_predictions_1_2.count(-1),temp_predictions_1_2.count(-2),temp_predictions_1_2)
    print(evaluation_data['label'])
    log_file.write(f"\nPerformance of {model_names[index_model]}; accuracy: {accuracy}; precision: {precision}; recall: {recall}; f1: {f1}\n")
    log_file.write(f"\nPredictions of {model_names[index_model]}; {predictions}\n")
    log_file.write(f"\nEval data of {model_names[index_model]}; {evaluation_data['label']}\n")
    log_file.write(f"\nTemp eval data of {model_names[index_model]}; {temp_eval_data['label']}\n")
    log_file.write(f"\nCount of -1, Count of -2, temp_predictions; {temp_predictions_1_2.count(-1)},{temp_predictions_1_2.count(-2)},{temp_predictions_1_2}\n")

    cm = confusion_matrix(evaluation_data['label'],predictions)
    report = classification_report(evaluation_data['label'],predictions)
    print(cm)
    print(report)
    log_file.write(f"\nCM of {model_names[index_model]}; {cm}\n")
    log_file.write(f"\nCM report of {model_names[index_model]}; {report}\n")

    cm = confusion_matrix(temp_eval_data['label'],predictions_only_identified)
    report = classification_report(temp_eval_data['label'],predictions_only_identified)
    print(cm)
    print(report)
    log_file.write(f"\nCM of manipulated {model_names[index_model]}; {cm}\n")
    log_file.write(f"\nCM report of manipulated {model_names[index_model]}; {report}\n")
    # Save confusion matrix plots for stress
    plot_filename = 'confusion_matrix_stress'+ model_names[index_model]+'.png'
    save_stress_confusion_matrix_plot(cm, 'Confusion Matrix - Stress', plot_filename) 

    csvfilename = 'stress_intermediate_gentext_'+model_names[index_model]+'.csv'
    df.to_csv(csvfilename, index=False)
    print (csvfilename)

  print(results)
  log_file.write(f"\nResults of {model_names[index_model]}; {results}\n")



#####Manual Confusion matrix creation using the data collected######

import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Replace this with your confusion matrix
cf_matrix = np.array([[43, 16], [20, 49]])

group_names = ["True Neg","False Pos","False Neg","True Pos"]
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
# Create a heatmap with custom axis labels
plt.figure(figsize=(6, 4))
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap="viridis",
            xticklabels=["Not Depressed", "Depressed"],
            yticklabels=["Not Depressed", "Depressed"])
plt.title('Confusion matrix of mixtral-8x7B - Depression')
# Save the plot as an image file (e.g., PNG, JPEG, PDF, etc.)
plt.savefig('confusion_matrix_dep_heatmap.png', dpi=300, bbox_inches='tight')

plt.show()

# Replace this with your confusion matrix
cf_matrix = np.array([[49, 4], [3, 72]])
group_names = ["True Neg","False Pos","False Neg","True Pos"]
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)

# Create a heatmap with custom axis labels
plt.figure(figsize=(6, 4))
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap="viridis",
            xticklabels=["Not Stressed", "Stressed"],
            yticklabels=["Not Stressed", "Stressed"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix of mixtral-8x7B - Stress')
# Save the plot as an image file (e.g., PNG, JPEG, PDF, etc.)
plt.savefig('confusion_matrix_stress_heatmap.png', dpi=300, bbox_inches='tight')

plt.show()