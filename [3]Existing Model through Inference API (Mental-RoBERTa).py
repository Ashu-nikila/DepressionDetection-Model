#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 00:59:52 2024

@author: Nikila Swaminathan

The script is designed to perform depression and stress detection using machine learning models. 
Here's a brief overview of its code flow logic and functionality:

Data Loading and Preprocessing:
The script starts by loading datasets related to depression and stress from the Hugging Face library.
It then preprocesses the text data by combining text and image text columns, lowercasing the text, 
cleaning special characters and punctuation, removing stopwords, and truncating the text to a 
maximum number of words.

Model Evaluation:
The script evaluates the performance of machine learning models in detecting depression and stress.
Evaluation metrics such as accuracy, precision, recall, and F1-score are calculated.
Confusion matrices and classification reports are generated to assess the model's performance.

API Integration:
The script integrates with Hugging Face's model inference API to make predictions on unseen text data.
It uses a custom function to preprocess the input text before making predictions using the deployed models.
Predictions are made for each input text, and the results are recorded.

Visualization:
The script visualizes the class distribution of the datasets using bar plots to understand the 
distribution of depressed and non-depressed instances.
It also generates confusion matrix plots to visually represent the model's performance in predicting 
depression and stress.

Result Recording:
The results, including model performance metrics, confusion matrices, and classification reports, are 
recorded in a results file for further analysis and comparison.

Overall, the script provides a structured approach to loading, preprocessing, evaluating, and visualizing 
depression and stress detection models using natural language processing techniques and external model 
inference APIs. It aims to provide insights into the effectiveness of different models in identifying 
depression and stress-related text data.

"""

#!pip -q install datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from huggingface_hub import notebook_login
import requests
import datasets
from datasets import load_dataset
import pandas as pd
import time 
import seaborn as sns
import matplotlib.pyplot as plt
#import textwrap

# A custom function to combine 'text' and 'image_text' columns
def combine_text_and_image_text(example):
    eval_prompt = """
Social Media Post:

{0}
""".format(example['old_text'],example['image2text'])
    return {'text': eval_prompt}

def preprocess_text (text):
  # Step 1: Lowercasing
  text = text.lower()

  # Step 2: Cleaning Special Characters and Punctuation
  import string
  special_chars = string.punctuation + '\n\t'  # Special characters to remove

  # Remove special characters and punctuation
  text = text.translate(str.maketrans('', '', special_chars))

  # Step 3: Removing Stop Words
  # Replace with your list of stop words
  stop_words = ["the", "and", "it", "is", "are"]
  words = text.split()
  filtered_text = ' '.join(word for word in words if word not in stop_words)

  # Step 4: Truncate to 128 words
  max_words = 128
  truncated_text = ' '.join(filtered_text.split()[:max_words])

  # Step 5: Print Preprocessed Text
  #print(truncated_text)
  return truncated_text

def evaluate_model(predictions, labels):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return accuracy, precision, recall, f1

def make_api_request (model_name, input_text,label_no_depression):
  url = f"https://api-inference.huggingface.co/models/{model_name}"
  headers = {"Authorization": f"Bearer <add hugging face token>"}
  prediction = 0 #set default to no depression
  response = requests.post(url, headers=headers, json={"inputs": input_text})
  #print (input_text)
  #print(response)
  if response.status_code == 200:
    predictions = response.json()
    for item in predictions:
      #print(item)
      max_score_label = max(item, key=lambda x: x['score'])
      if (max_score_label['label'] != label_no_depression):
        prediction = 1
      else:
        prediction = 0
  elif response.status_code == 429:  # Rate limit exceeded
    retry_after = int(response.headers.get('Retry-After', 5))  # Default to 5 seconds if 'Retry-After' header not provided
    print(f"Rate limit exceeded. Waiting for {retry_after} seconds before retrying.")
    time.sleep(retry_after)
    prediction = make_api_request(url,input_text,label_no_depression)  # Retry the request after waiting
  elif response.status_code == 503:
    print("Service unavailable. Retrying in 60 seconds.")
    time.sleep(60)
    prediction = make_api_request(url,input_text,label_no_depression)  # Retry the request after waiting
  elif response.status_code == 401:
    print("Unauthorized. Check your authentication credentials.")
  elif response.status_code == 403:
    print("Forbidden. You don't have permission to access this resource.")
  elif response.status_code == 404:
    print("Not found. The requested resource does not exist.")
  elif response.status_code == 500:
    print("Internal Server Error. The server encountered an unexpected condition.")
  else:
    print(f"Failed with status code: {response.status_code}, Response: {response.text}")

  return prediction

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

def save_stress_confusion_matrix_plot(cm, title, file_name):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                xticklabels=["Not Stressed", "Stressed"],
                yticklabels=["Not Stressed", "Stressed"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    
####################
###### Main ########
####################

log_file_name = "/content/my_log_file.txt"

# Manually authenticate with the Hugging Face API
print("Please authenticate with the Hugging Face API:")
notebook_login(input("Enter your API token: "))
#token: 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

# Load your dataset from Hugging Face
# Replace 'dataset_name' with the name of your dataset
raw_datasets = load_dataset("nikilas/DepressionIncludingImageText", use_auth_token=True)
# you can also use the data multimidal_dataset.csv in the data folder in github

raw_datasets = raw_datasets.rename_column('text', 'old_text')
#print(raw_datasets.keys())

# Use the .map() method to create a new column 'combined_text'
raw_datasets = raw_datasets.map(combine_text_and_image_text)

# Print the keys to see the new column
print(raw_datasets.keys())

# Define the columns to remove
columns_to_remove = ['old_text', 'image2text']

# Use the .remove_columns() method to remove the specified columns
raw_datasets = raw_datasets.remove_columns(columns_to_remove)

############ Stress ############

# Filter the dataset based on a column's value
stress_dataset = raw_datasets.filter(lambda example: example['class'] == 'Stress')
#print("Stress dataset: Training before split ",len(stress_dataset['train']))

#Split the dataset into training, validation abd test datasets
train_testvalid = stress_dataset['train'].train_test_split(train_size=0.7, seed=42)

# Split the 30% (test + valid) in half test, half valid
test_valid = train_testvalid['test'].train_test_split(train_size=0.5, seed=42)

# gather everyone if you want to have a single DatasetDict
stress_dataset = datasets.DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'val': test_valid['train']})

#print("Stress dataset: Splits available in the DatasetDict:", stress_dataset.keys())

'''
# Iterate over each split and print its details
for split in stress_dataset.keys():
  print ({split: len(stress_dataset[split]) })
  # Print the structure of the Dataset
  print(stress_dataset[split])
'''

#Understand the data distribution for stress
val_df = pd.DataFrame(stress_dataset['val'])

# Assuming you have a dataset in a pandas DataFrame named 'df'
df = val_df

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
plot_file_path = "class_distribution_plot-Stress-ExistingModel.png"
plt.savefig(plot_file_path)
plt.show()

evaluation_data = stress_dataset['val'].select(list(range(128)))

print("Number of data in the val stress dataset",len(stress_dataset['val']))
evaluation_data = stress_dataset['val'].select(list(range(128)))
print ("Evaluation Data - 1st Text", evaluation_data[0]['text'])
print ("Evaluation Data - 1st label ", evaluation_data[0]['label'])
'''
Test a single value works
import textwrap

data = textwrap.shorten(evaluation_data[1]['text'], 128)
model_output = make_api_request("paulagarciaserrano/roberta-depression-detection", data,"not stress")
print (model_output)
'''
models = ["paulagarciaserrano/roberta-depression-detection"]  # Replace with your model choices
modelnames = ["roberta-depression-detection"] # used for filenames, goes tandem with models above

results = {}
results_file_name = "ExistingModel_results.txt"
with open(results_file_name, "a") as results_file:    
    for modelindex, model in enumerate(models):
        predictions = []
        for index, input_text in enumerate(evaluation_data['text']):
            # Process model_output to get the prediction of stress
            prediction = make_api_request(model, preprocess_text(input_text),"not stress")
            predictions.append(prediction)
            # Print the loop counter (index)
            #print(f"Processing text number: {index + 1}, prediction: {prediction}, actual {evaluation_data[index]['label']}")
            results_file.write(f"Processing text number: {index + 1}, prediction: {prediction}, actual {evaluation_data[index]['label']}\n")
    
        accuracy, precision, recall, f1 = evaluate_model(predictions, evaluation_data['label'])
        results[model] = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    
        # Process predictions and true labels
        #predicted_labels = predictions
        #true_labels = [sample['label'] for sample in evaluation_data]
        #evaluate_model(predicted_labels,true_labels)
    
        cm = confusion_matrix(evaluation_data['label'],predictions)
        report = classification_report(evaluation_data['label'],predictions)
        results_file.write(f"CM: {cm}\n")
        results_file.write(f"Report: {report}\n")
        #print(cm)
        #print(report)
        # Save confusion matrix plots for stress
        plot_filename = 'confusion_matrix_stress'+ modelnames[modelindex]+'.png'
        save_stress_confusion_matrix_plot(cm, 'Confusion Matrix - Stress', plot_filename)        

    #print(results)
    results_file.write("Results of Stress Detection:\n")
    for model, result in results.items():
        results_file.write(f"Model: {model}\n")
        results_file.write(f"Accuracy: {result['accuracy']}\n")
        results_file.write(f"Precision: {result['precision']}\n")
        results_file.write(f"Recall: {result['recall']}\n")
        results_file.write(f"F1-score: {result['f1']}\n\n")

############Depression##############
# Filter the dataset based on a column's value
depression_dataset = raw_datasets.filter(lambda example: example['dataset'] == 'SDCNL' or example['dataset'] == 'IDD2018')
print("Depression: Training before split ",len(depression_dataset['train']))
#Split the dataset into training, validation abd test datasets
train_testvalid = depression_dataset['train'].train_test_split(train_size=0.7, seed=42)

# Split the 30% (test + valid) in half test, half valid
test_valid = train_testvalid['test'].train_test_split(train_size=0.5, seed=42)

# gather everyone if you want to have a single DatasetDict
depression_dataset = datasets.DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'val': test_valid['train']})
#print("Depression: Splits available in the DatasetDict:", depression_dataset.keys())
'''
# Iterate over each split and print its details
for split in depression_dataset.keys():
  print ({split: len(depression_dataset[split]) })
  # Print the structure of the Dataset
  print(depression_dataset[split])
'''

#Understand the data distribution for depression
val_df = pd.DataFrame(depression_dataset['val'])

# Assuming you have a dataset in a pandas DataFrame named 'df'
df = val_df

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
plt.show()
plot_file_path = "class_distribution_plot-Depression-ExistingModel.png"
plt.savefig(plot_file_path)

print("Number of data in the val depression dataset",len(depression_dataset['val']))
evaluation_data = depression_dataset['val'].select(list(range(128)))
print ("Evaluation Data - 1st Text", evaluation_data[0]['text'])
print ("Evaluation Data - 1st label ", evaluation_data[0]['label'])

results = {}

results_file_name = "ExistingModel_results.txt"
with open(results_file_name, "a") as results_file:
    for modelindex, model in enumerate(models):
        predictions = []
        for index, input_text in enumerate(evaluation_data['text']):
            # Process model_output to get the prediction of depression
            prediction = make_api_request(model, preprocess_text(input_text),"not depression")
            predictions.append(prediction)
            # Print the loop counter (index)
            #print(f"Processing text number: {index + 1}, prediction: {prediction}, actual {evaluation_data[index]['label']}")
            results_file.write(f"Processing text number: {index + 1}, prediction: {prediction}, actual {evaluation_data[index]['label']}\n")
    
        accuracy, precision, recall, f1 = evaluate_model(predictions, evaluation_data['label'])
        results[model] = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    
        # Process predictions and true labels
        #predicted_labels = predictions
        #true_labels = [sample['label'] for sample in evaluation_data]
        #evaluate_model(predicted_labels,true_labels)
    
        cm = confusion_matrix(evaluation_data['label'],predictions)
        report = classification_report(evaluation_data['label'],predictions)
        results_file.write(f"CM: {cm}\n")
        results_file.write(f"Report: {report}\n")
        #print(cm)
        #print(report)
        # Save confusion matrix plots for depression
        plot_filename = 'confusion_matrix_depression'+ modelnames[modelindex]+'.png'
        save_confusion_matrix_plot(cm, 'Confusion Matrix - Depression', plot_filename)

    #print(results)
    results_file.write("Results of Depression detection:\n")
    for model, result in results.items():
        results_file.write(f"Model: {model}\n")
        results_file.write(f"Accuracy: {result['accuracy']}\n")
        results_file.write(f"Precision: {result['precision']}\n")
        results_file.write(f"Recall: {result['recall']}\n")
        results_file.write(f"F1-score: {result['f1']}\n\n")
