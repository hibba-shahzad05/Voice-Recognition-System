import numpy as np
import librosa
import joblib
import os


# --------------------------------------------------
# 1. LOAD TRAINED MODEL
# --------------------------------------------------

MODEL_PATH = "emotion_model.pkl"

model = joblib.load(MODEL_PATH)

print("Emotion recognition model loaded successfully!")


# --------------------------------------------------
# 2. FEATURE EXTRACTION
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
# 3. PREDICT EMOTION
# --------------------------------------------------

def predict_emotion(file_path):

    features = extract_features(file_path)

    features = features.reshape(1, -1)

    prediction = model.predict(features)

    emotion = prediction[0]

    return emotion


# --------------------------------------------------
# 4. AUDIO FILE INPUT
# --------------------------------------------------

audio_file = input(
    "\nEnter the path of your WAV audio file: "
).strip()


# --------------------------------------------------
# 5. CHECK FILE
# --------------------------------------------------

if not os.path.exists(audio_file):

    print("\nAudio file not found!")

else:

    try:

        emotion = predict_emotion(
            audio_file
        )

        print("\n-----------------------------")
        print("Predicted Emotion:", emotion)
        print("-----------------------------")

    except Exception as e:

        print("\nError:", e)