#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 20:48:26 2024

@author: Nikila Swaminathan

The script analyzes social media posts containing text and images to detect 
signs of depression and stress using large language models (LLMs) from OpenAI.
 
It begins by initializing prompt templates for depression and stress detection, along with an API key 
for accessing the OpenAI API. 

The main logic involves processing the data by running the language model 
on the generated text and extracting responses to determine the predicted label for each post. 
This process is repeated for each row in the dataset. 

After processing, the code evaluates the model's performance by calculating accuracy, precision, recall, 
and F1-score metrics. 

Finally, the results are printed for both depression and stress detection, including confusion matrices 
and classification reports. 

Overall, the code automates the process of analyzing social media content to 
identify potential indicators of depression and stress using advanced language models.

"""

#prerequistes
#!pip install cohere tiktoken
#!pip install langchain
#!pip install openai==0.28.1


import pandas as pd
import numpy as np
import re
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from getpass import getpass
import os

def evaluate_model(predictions, labels):
    """
    Evaluate model performance using various metrics.
    """
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return accuracy, precision, recall, f1

def process_data(df, prompt_template, llm_chain):
    """
    Process data and generate predictions.
    """
    df['Predicted'] = np.nan
    df['GPT_Return_word'] = np.nan

    for index, row in df.iterrows():
        print(f"Index: {index}, Row label: {row['label']}")

        return_word = llm_chain.run(row['gen_text'])
        word_trimmed = re.sub(r'^[^:]*:\s*', '', return_word)
        word_trimmed = re.sub(r'^[\s\W]+|[\s\W]+$', '', word_trimmed)

        is_yes = word_trimmed.lower() == "yes"
        is_no = word_trimmed.lower() == "no"
        is_uncertain = word_trimmed.lower() == "uncertain"
        
        if is_yes or is_no:
            predicted = 1 if is_yes else 0
        elif is_uncertain:
            predicted = 1 if row['label'] == 0 else 0
        else:
            predicted = -1
      
        df.at[index, 'Predicted'] = predicted
        df.at[index, 'GPT_Return_word'] = return_word
        print(f"Processing text number: {index + 1}, prediction: {predicted}, actual {row['label']}")
     
    return df

if __name__ == "__main__":
    OPENAI_API_KEY = getpass()
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # Define prompt templates
    dep_template = """Please analyze the following assessment made by a large language model regarding a social media post containing text and images. Determine if the language model has identified signs of depression. {input_text}

    Based on its analysis, respond with a single word: 'Yes', 'No', or 'Uncertain'"""

    stress_template = """Please analyze the following assessment made by a large language model regarding a social media post containing text and images. Determine if the language model has identified signs of stress. {input_text}

    Based on its analysis, respond with a single word: 'Yes', 'No', or 'Uncertain'"""

    # Initialize LLM and LLMChain
    llm = OpenAI()
    dep_prompt = PromptTemplate(template=dep_template, input_variables=["input_text"])
    stress_prompt = PromptTemplate(template=stress_template, input_variables=["input_text"])
    
    #models = ["T5","dolly","gpt","llama70","oasst","vicuna"]
    #models = ["llama","FLAN","mixtral"]
    models = ["mixtral"]
    
    for index, modelname in enumerate(models):
        llm_chain_dep = LLMChain(prompt=dep_prompt, llm=llm)
        llm_chain_stress = LLMChain(prompt=stress_prompt, llm=llm)

        # Process data for depression detection
        dep_df = pd.read_csv(f"intermediate_gentext_{modelname}.csv")
        dep_df = process_data(dep_df, dep_prompt, llm_chain_dep)
        dep_df.to_csv(f"Dep_detected_{modelname}.csv", index=False)
        dep_accuracy, dep_precision, dep_recall, dep_f1 = evaluate_model(dep_df['Predicted'], dep_df['label'])
        print(f"Depression Model: {modelname}, Accuracy: {dep_accuracy}, Precision: {dep_precision}, Recall: {dep_recall}, F1 Score: {dep_f1}")
        cm = confusion_matrix(dep_df['Predicted'],df['label'])
        report = classification_report(dep_df['Predicted'],df['label'])
        print(cm)
        print(report)

        # Process data for stress detection
        stress_df = pd.read_csv(f"intermediate_gentext_{modelname}_Stress.csv")
        stress_df = process_data(stress_df, stress_prompt, llm_chain_stress)
        stress_df.to_csv(f"Stress_detected_{modelname}.csv", index=False)
        stress_accuracy, stress_precision, stress_recall, stress_f1 = evaluate_model(stress_df['Predicted'], stress_df['label'])
        print(f"Stress Model: {modelname}, Accuracy: {stress_accuracy}, Precision: {stress_precision}, Recall: {stress_recall}, F1 Score: {stress_f1}")
        cm = confusion_matrix(stress_df['Predicted'],df['label'])
        report = classification_report(stress_df['Predicted'],df['label'])
        print(cm)
        print(report)