# Speech-Recognition-0-9
# Digit Speech Recognition using Convolutional Neural Networks (CNN)

This project demonstrates how to build a digit speech recognition system using Convolutional Neural Networks (CNNs) with TensorFlow and Keras. The model is trained on the Google Speech Commands dataset.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Pretrained Model](#pretrained-model)
- [Contributing](#contributing)


## Introduction

Speech recognition is the process of converting spoken words into text. In this project, we focus on recognizing spoken digits (0-9) using deep learning techniques. We utilize MFCC (Mel-frequency cepstral coefficients) features extracted from audio samples as inputs to a CNN model. The CNN learns to classify the spoken digits based on these features.

## Requirements

To run the code in this project, you need the following dependencies:

- Important note to open the DigitSpeak_Final.ipynb you need to Clone the repo on Visual Studio or Download it ,then open it from google colab due to different nbconvert versions
- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- SciPy
- Matplotlib
- Seaborn
- Librosa
- scikit-learn


## Usage
- Dataset Preparation:

Download the Google Speech Commands dataset and extract it into the specified directory.
- Model Training:

Run the train_model.py script to train the CNN model on the prepared dataset.
- Model Evaluation:

After training, evaluate the model's performance using the evaluate_model.py script.
- Prediction:

Use the pre-trained model to make predictions on new audio samples with the predict_digit.py script.
To test the model you can record a .wav file using Online recorder Exp: https://voice-recorder-online.com/
- Pretrained Model
We provide a pre-trained CNN model (digit_speech_recognition_Finalmodel.h5) that you can use to predict digits from audio samples directly. Refer to the Prediction section for usage instructions.

## Contributing
Contributions are welcome! If you have any suggestions, improvements, or bug fixes, feel free to open an issue or submit a pull request.




