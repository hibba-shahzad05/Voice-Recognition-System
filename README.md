# 🎙️ AI Speech Emotion Recognition

## 📌 Project Overview

This project is a Speech Emotion Recognition system developed
using Machine Learning. The system analyzes speech audio and
predicts the emotion expressed in the recording.

## 🎯 Objective

The main objective is to classify speech recordings into
different emotional categories using audio features and
machine learning.

## 🎭 Supported Emotions

The model recognizes eight emotions:

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

## 📊 Dataset

The project uses the RAVDESS emotional speech dataset.

The dataset contains speech recordings from 24 actors.

A total of 1,440 audio files are used in this project.

## 🔊 Feature Extraction

MFCC (Mel-Frequency Cepstral Coefficients) features are
extracted from each audio recording.

40 MFCC features are used for classification.

## 🤖 Machine Learning Model

Random Forest Classifier is used for emotion classification.

Model configuration:

- Number of estimators: 200
- Class weight: Balanced
- Random state: 42

## 📈 Model Performance

The model achieved:

**Accuracy: 63.89%**

## 🧪 Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## 🖥️ Streamlit Application

A Streamlit web application was developed where users can:

1. Upload a WAV speech file
2. Play the uploaded audio
3. Analyze the speech
4. View the predicted emotion
5. View prediction confidence
6. View emotion probabilities

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Librosa
- Scikit-learn
- Random Forest
- Matplotlib
- Streamlit
- Joblib

## 🚀 How to Run

Install dependencies:

```bash
pip install -r requirements.txt