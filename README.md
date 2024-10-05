MentalDisorderDetetion is licensed under the GNU General Public License v3.0. Copyright (C) 2024 Nikila Swaminathan

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

##Multimodal Mental Health Detection

This repository contains Python scripts related to a project on detecting mental health conditions (specifically depression and stress) among adolescents using multimodal deep learning techniques. The project leverages both text and image data from social media to improve the accuracy of detection.

###Repository Contents
The repository includes the following Python scripts:

1) Image2Text_using_LLaVA_model_from_HuggingFace_with_Pipe.py

Description: This script uses the LLAVA (Large Language and Vision Assistant) model from Hugging Face to convert images (selfies) into textual descriptions. The image-to-text conversion captures emotional cues from facial expressions, which are then used for further analysis in detecting mental health indicators.

Purpose: To extract textual features from images that can be combined with text data for multimodal analysis.

2) Benchmark_LLM_Models_RunInReplicate_Stress_Depression.py

Description: This script benchmarks various Large Language Models (LLMs) for their performance in detecting stress and depression from text data. It runs models available on Replicate, including LLaMA-2-70B-Chat, Mixtral-8x7B-Instruct-v0.1, FLAN-T5-XL, and others.

Purpose: To evaluate and compare the effectiveness of different LLMs in mental health detection tasks.

3) Existing_Model_through_Inference_API_MentalRoBERTa.py

Description: This script utilizes the pre-trained MentalRoBERTa model via an inference API to detect mental health conditions from text data. It demonstrates how existing models can be integrated into the pipeline and used for classification tasks.

Purpose: To leverage established models for baseline comparisons and integration into the detection pipeline.

4) Mixtral_finetuned_with_qLoRA.py

Description: This script fine-tunes the Mixtral-8x7B-Instruct-v0.1 model using the qLoRA (Quantized Low-Rank Adapters) technique. The fine-tuned model is optimized for detecting depression and stress from multimodal data (text and image-derived text). It includes code for training the model and evaluating its performance.

Purpose: To enhance the model's ability to detect mental health indicators by fine-tuning it with specialized techniques.

5) TextClassification_gpt3.5turbo.py

Description: This script uses OpenAI's GPT-3.5 Turbo model for text classification tasks related to mental health detection. It demonstrates how to set up prompts and process responses to classify text data for indicators of depression and stress.

Purpose: To utilize state-of-the-art language models for classification tasks through prompt engineering.

###Requirements
To run these scripts, you'll need the following:

Python 3.7 or higher

Required Python libraries (see below)

Access to certain models and APIs (e.g., OpenAI API, Replicate API, Hugging Face models)

Python Libraries
Install the required libraries using pip:

pip install torch transformers datasets bitsandbytes accelerate
pip install openai
pip install replicate
pip install pandas numpy
pip install fastapi uvicorn

Additional libraries may be required depending on the script.

Access and API Keys

OpenAI API: For TextClassification_gpt3.5turbo.py, you'll need an OpenAI API key. Sign up at OpenAI to obtain one.

Replicate API: For scripts using models from Replicate, you'll need a Replicate API key. Sign up at Replicate to obtain one.

Hugging Face Models: Some models may require acceptance of model terms or authentication tokens. Sign up at Hugging Face to access these models.

Usage Instructions
1. Image2Text_using_LLaVA_model_from_HuggingFace_with_Pipe.py
Purpose: Convert images to text descriptions using the LLAVA model.

Usage:

Setup:
Ensure you have the required models and dependencies installed.
Access to the liuhaotian/LLaVA-Lightening-MPT-7B-preview model on Hugging Face.

Modify the Script:
Update the data_dir variable to point to your image dataset directory.
Ensure that the images are in a compatible format (e.g., JPEG, PNG).

Run the Script:
Execute the script in an environment with GPU support for optimal performance.
The script will process images and generate text descriptions, saving results to a specified output file.

2. Benchmark_LLM_Models_RunInReplicate_Stress_Depression.py
Purpose: Benchmark various LLMs for stress and depression detection from text.

Usage:

Setup:
Set your Replicate API key as an environment variable:

export REPLICATE_API_TOKEN='your_replicate_api_token'

Modify the Script:
Provide the dataset of text posts to analyze, ensuring it is properly formatted.

Run the Script:
Execute the script to evaluate the performance of different models.
Results will include metrics such as accuracy, precision, recall, and F1-score.

3. Existing_Model_through_Inference_API_MentalRoBERTa.py
Purpose: Use the MentalRoBERTa model for mental health detection via an inference API.

Usage:

Setup:
Ensure you have access to the MentalRoBERTa inference API.

Modify the Script:
Input the text data you wish to analyze.

Run the Script:
Execute the script to obtain predictions on the input text.
The output will indicate the likelihood of depression or stress indicators.

4. Mixtral_finetuned_with_qLoRA.py
Purpose: Fine-tune the Mixtral model using qLoRA.

Usage:

Setup:
Ensure you have the necessary computational resources (GPU with sufficient VRAM).
Install bitsandbytes for 8-bit optimizers.

Modify the Script:
Update the paths to your training and validation datasets.
Adjust hyperparameters (learning rate, batch size) as needed.

Run the Script:
Execute the script to fine-tune the model.
Monitor training progress through logs or use tools like Weights & Biases.

5. TextClassification_gpt3.5turbo.py
Purpose: Use GPT-3.5 Turbo for text classification.

Usage:

Setup:
Set your OpenAI API key as an environment variable:

export OPENAI_API_KEY='your_openai_api_key'

Modify the Script:
Provide the text data for classification.
Adjust the prompt design if necessary.

Run the Script:
Execute the script to obtain classification results.
The script processes responses and outputs the predictions.

Notes
Data Privacy: Ensure that you have the rights and permissions to use any datasets, especially when they contain sensitive information related to mental health. Anonymize data where appropriate.

Ethical Considerations: Be mindful of ethical guidelines when using AI models for mental health detection. Avoid misuse and ensure compliance with relevant laws and regulations.

Model Access: Some models may have usage restrictions or require agreement to terms and conditions. Check the licensing for each model you use.

Dependencies: Make sure to install all dependencies and check for version compatibility.

## License

This project is licensed under the [GPL3](LICENSE).
