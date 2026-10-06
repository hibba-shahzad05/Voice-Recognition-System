import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import joblib


# --------------------------------------------------
# 1. DATASET PATH
# --------------------------------------------------

DATASET_PATH = "archive"

# Full dataset: 24 actors
ACTORS = [
    f"Actor_{i:02d}"
    for i in range(1, 25)
]


# --------------------------------------------------
# 2. EMOTION LABELS
# --------------------------------------------------

emotion_labels = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


# --------------------------------------------------
# 3. MFCC FEATURE EXTRACTION
# --------------------------------------------------

def extract_features(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        sr=None
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc_scaled = np.mean(
        mfcc.T,
        axis=0
    )

    return mfcc_scaled


# --------------------------------------------------
# 4. LOAD DATASET
# --------------------------------------------------

features = []
labels = []

print("Loading audio files...")

for actor in ACTORS:

    actor_path = os.path.join(
        DATASET_PATH,
        actor
    )

    if not os.path.exists(actor_path):
        print(f"Folder not found: {actor_path}")
        continue

    for file_name in os.listdir(actor_path):

        if file_name.endswith(".wav"):

            file_path = os.path.join(
                actor_path,
                file_name
            )

            try:

                # Filename se emotion code
                emotion_code = file_name.split("-")[2]

                emotion = emotion_labels[
                    emotion_code
                ]

                # Extract MFCC features
                mfcc_features = extract_features(
                    file_path
                )

                features.append(mfcc_features)
                labels.append(emotion)

            except Exception as e:

                print(
                    f"Error processing {file_name}: {e}"
                )


# --------------------------------------------------
# 5. CREATE DATASET
# --------------------------------------------------

X = np.array(features)
y = np.array(labels)

print("\nDataset loaded successfully!")

print("Feature shape:", X.shape)
print("Number of labels:", len(y))

print("\nEmotion distribution:")

print(
    pd.Series(y).value_counts()
)


# --------------------------------------------------
# 6. TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 7. IMPROVED RANDOM FOREST MODEL
# --------------------------------------------------

print("\nTraining improved Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("Model training completed!")


# --------------------------------------------------
# 8. PREDICTION
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 9. MODEL EVALUATION
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nModel Accuracy:", accuracy)

print(
    f"Model Accuracy Percentage: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# --------------------------------------------------
# 10. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(emotion_labels.values())
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=list(emotion_labels.values())
)

disp.plot(
    xticks_rotation=45
)

plt.title(
    "Speech Emotion Recognition - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "emotion_confusion_matrix.png"
)

plt.show()


# --------------------------------------------------
# 11. SAVE MODEL
# --------------------------------------------------

joblib.dump(
    model,
    "emotion_model.pkl"
)

print("\nModel saved successfully!")

print("File: emotion_model.pkl")