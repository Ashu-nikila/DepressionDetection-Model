#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 22:21:13 2024

This script analyzes facial expressions, posture, and demeanor in selfies using an AI model.
It loads an image, processes it through the AI model, and generates text feedback based on the image content.

@author: Nikila Swaminathan

1. Library Imports: The script imports necessary libraries, including torch for PyTorch, BitsAndBytesConfig from the 
Transformers library for quantization configuration, and other utilities for image processing and file handling.

2. Pipeline Configuration: It configures the pipeline for image-to-text conversion using the Hugging Face Transformers library, 
specifying the model to be used and its associated quantization configuration.

3. Mounting Google Drive: The script mounts Google Drive to access image datasets stored in a specific directory.

4. Data Processing: It iterates through subdirectories in the dataset directory, collecting information about image files and 
their associated class names.

5. Prompt Definition: It defines a prompt to be used when generating text from images, instructing the AI model on what to 
analyze and provide feedback on.

6. Image Analysis Loop: Within a loop, the script reads each image file, passes it through the AI model along with the defined 
prompt, and captures the generated text response.

7. Logging and File Movement: It logs the results of the image analysis to a text file and moves processed image files to a 
designated directory.

8. Results Printing: Finally, it prints the results of the analysis, displaying the index, class name, filename, and corresponding 
text response in a tabular format.
"""

#LLaVA: Large Language and Vision Assistant
#https://huggingface.co/llava-hf/llava-1.5-7b-hf
#https://llava-vl.github.io/

##using pipeline##
#!pip install -q transformers==4.36.0
#!pip install -q bitsandbytes==0.41.3 accelerate==0.25.0
  
# Library Imports
import os
import random
import shutil
import requests
from PIL import Image
import torch
from transformers import BitsAndBytesConfig, pipeline

# Configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
model_id = "llava-hf/llava-1.5-7b-hf"

#It is important to prompt the model wth a specific format, which is:
#USER: <image>\n<prompt>\nASSISTANT:
prompt = "USER: <image>\nAnalyze the facial expressions, posture, and overall demeanor in the selfie. Pay attention to indicators such as sadness, lack of energy, or forced smiles. Evaluate the person's appearance, like disheveled hair or lack of personal grooming, which can be signs of neglect. Assess the background for indications of isolation or disarray.\nASSISTANT:"

# Pipeline Initialization
pipe = pipeline("image-to-text", 
                model=model_id, 
                model_kwargs={"quantization_config": quantization_config})

# Google Drive Mounting
from google.colab import drive
drive.mount('/content/drive')

# Dataset Directory
base_dir = '/content/drive/MyDrive/Colab Notebooks/Data/images/Selfie'

# List to store class and filename info
image_info = []

# Collecting Image Information
# Iterate through each subfolder (class) in the base directory
for class_name in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_name)
    # Make sure it's a directory
    if os.path.isdir(class_path):
        # Iterate through each file in the subfolder
        for filename in os.listdir(class_path):
            # Add class and filename to the list
            file_full_paths = os.path.join(class_path, filename)
            image_info.append((class_name, file_full_paths))

# Now, 'image_info' contains tuples of (class_name, filename)       
# Shuffling Image Info
random.shuffle(image_info)

# Maximum New Tokens for Prompt
max_new_tokens = 200

# Log File Path
log_file_name = "/content/my_log_file_text-to-image.txt"
done_dir = '/content/drive/MyDrive/Colab Notebooks/Data/images/Done'
# File Path for Results
results_file_name = "/content/results_text_to_image.txt"

# Process Each Image
results = []
with open(log_file_name, "a") as log_file:
    for index, (class_name, filename) in enumerate(image_info):
        if os.path.exists(filename):
            with Image.open(filename) as img:
                outputs = pipe(img, prompt=prompt, generate_kwargs={"max_new_tokens": max_new_tokens})
                generated_text = "".join(outputs[0]['generated_text'])
                assistant_response = generated_text.split("ASSISTANT:")[1].strip()
                results.append((index+1, class_name, filename, assistant_response))
                log_file.write(f"{index+1};{class_name};{filename};{assistant_response}\n")
                destination_path = os.path.join(done_dir, class_name)
                #once the image file is processed that are in selfie folder move to Done folder
                shutil.move(filename, destination_path)
                #if index % 20 == 0:
                #    break  # Exit the loop when the index reaches 100

# Print Results in a file
with open(results_file_name, "a") as results_file:
    # Print header
    results_file.write(f"{'Index':<10}{'Class':<10}{'Filename':<10}{'Response':<10}\n")
    results_file.write("-" * 30 + "\n")
    # Print header
    #print(f"{'Index':<10}{'Class':<10}{'Filename':<10}{'Response':<10}")
    #print("-" * 30)
    # Print each tuple in table format
    for item in results:
        #print(f"{item[0]:<10}{item[1]:<10}{item[2]:<10}{item[3]:<10}")
        results_file.write(f"{item[0]:<10}{item[1]:<10}{item[2]:<10}{item[3]:<10}\n")
