# Text Classification with Convolutional Neural Networks (CNNs)

This project demonstrates how to perform text classification using Convolutional Neural Networks (CNNs) in TensorFlow. 

It focuses on sentiment analysis of movie reviews, classifying them as either positive or negative.

## Description

The project uses the IMDB dataset, which consists of movie reviews that are labeled as either positive (1) or negative (0). 

We implement a CNN model in TensorFlow to automatically classify unseen movie reviews. This approach showcases the power of CNNs not only in image processing but also in natural language processing tasks.

## Table of Contents 

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Real-World Application Scenarios](#real-world-application-scenarios)

## Installation

To run this project, you will need to install Python and TensorFlow. Follow these steps to set up your environment:

1. Install Python (3.8 or later is recommended).
2. Install TensorFlow by running the following command:
   ```
   pip install tensorflow
   ```
3. Clone the GitHub repository:
   ```
   git clone https://github.com/Sorena-Dev/Text-Classification-with-Convolutional-Neural-Networks-CNNs-.git
   ```
4. Navigate to the cloned repository and install any additional requirements (if any):
   ```
   pip install -r requirements.txt
   ```

## Usage

To use this project for text classification:

1. Ensure you have followed the installation steps.
2. Run the Python script `Text Classification with Convolutional Neural Networks (CNNs).py`.
3. The script will automatically download the IMDB dataset, train the CNN model, and evaluate its performance on the test set.
4. To classify new texts, modify the `new_texts` list in the `if __name__ == "__main__":` section of the script with your input texts.


## Features

- Utilizes the TensorFlow framework for building and training a CNN model.
- Employs the IMDB movie reviews dataset for binary sentiment classification.
- Demonstrates preprocessing of text data, model construction, training, evaluation, and prediction.
- Provides an easy-to-understand code structure for beginners in TensorFlow and neural network modeling.

## Real-World Application Scenarios

This model can be adapted for various real-world sentiment analysis tasks, such as:

- Analyzing customer reviews and feedback on products and services.
- Monitoring social media sentiment towards brands, products, or public figures.
- Enhancing content recommendation systems by understanding user preferences expressed in text.
