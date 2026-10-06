import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile
import os


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide"
)


# =========================================================
# CUSTOM CSS ONLY
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f5f7fb;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #6b7280;
        margin-bottom: 35px;
    }

    .result-box {
        background-color: white;
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        border: 1px solid #e5e7eb;
        margin-top: 25px;
    }

    .result-text {
        font-size: 40px;
        font-weight: 800;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<h1 class="main-title">🎙️ AI Speech Emotion Recognition</h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'Upload a speech recording and let AI identify the emotion '
    'expressed in the voice.'
    '</p>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return joblib.load("emotion_model.pkl")


try:

    model = load_model()

except Exception as e:

    st.error(f"Could not load emotion_model.pkl: {e}")
    st.stop()


# =========================================================
# FEATURE EXTRACTION
# =========================================================

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


# =========================================================
# EMOTION EMOJIS
# =========================================================

emotion_emoji = {
    "neutral": "😐",
    "calm": "😌",
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "fearful": "😨",
    "disgust": "🤢",
    "surprised": "😲"
}


# =========================================================
# UPLOAD SECTION
# =========================================================

st.subheader("🎧 Upload Your Speech")

st.write(
    "Upload a WAV audio recording to analyze its emotional tone."
)

uploaded_file = st.file_uploader(
    "Choose a WAV audio file",
    type=["wav"]
)


# =========================================================
# AUDIO PREVIEW
# =========================================================

if uploaded_file is not None:

    st.success(
        f"Audio uploaded successfully: {uploaded_file.name}"
    )

    st.audio(
        uploaded_file,
        format="audio/wav"
    )

    st.divider()


    # =====================================================
    # ANALYZE BUTTON
    # =====================================================

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        analyze_button = st.button(
            "🔍 Analyze Emotion",
            use_container_width=True,
            type="primary"
        )


    # =====================================================
    # PREDICTION
    # =====================================================

    if analyze_button:

        try:

            with st.spinner(
                "🎙️ AI is analyzing the speech..."
            ):

                # Create temporary audio file
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".wav"
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getvalue()
                    )

                    temp_path = temp_file.name


                # Extract features
                features = extract_features(
                    temp_path
                )

                features = features.reshape(
                    1,
                    -1
                )


                # Prediction
                prediction = model.predict(
                    features
                )

                emotion = prediction[0]


                # Probabilities
                probabilities = model.predict_proba(
                    features
                )[0]

                classes = model.classes_

                probability_dict = dict(
                    zip(
                        classes,
                        probabilities
                    )
                )

                confidence = (
                    probability_dict[emotion] * 100
                )


                # Remove temporary file
                os.remove(temp_path)


            # =================================================
            # RESULT
            # =================================================

            st.divider()

            emoji = emotion_emoji.get(
                emotion,
                "🎭"
            )

            st.subheader("🎯 Detected Emotion")

            st.success(
                f"{emoji} {emotion.upper()}"
            )


            # =================================================
            # METRICS
            # =================================================

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Predicted Emotion",
                    emotion.capitalize()
                )

            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            with col3:

                st.metric(
                    "MFCC Features",
                    "40"
                )


            # =================================================
            # PROBABILITY CHART
            # =================================================

            st.divider()

            st.subheader(
                "📊 Emotion Probabilities"
            )

            probability_data = {
                name.capitalize():
                round(prob * 100, 2)
                for name, prob
                in probability_dict.items()
            }

            st.bar_chart(
                probability_data
            )


            # =================================================
            # DETAILED EMOTION ANALYSIS
            # =================================================

            st.subheader(
                "🎭 Detailed Analysis"
            )

            sorted_emotions = sorted(
                probability_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )

            for emotion_name, probability in sorted_emotions:

                percentage = probability * 100

                emoji = emotion_emoji.get(
                    emotion_name,
                    "🎭"
                )

                st.write(
                    f"{emoji} **{emotion_name.capitalize()}** "
                    f"— {percentage:.2f}%"
                )

                st.progress(
                    int(min(percentage, 100))
                )


        except Exception as e:

            st.error(
                f"Error processing audio: {e}"
            )


# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

st.subheader("🧠 About the AI Model")

st.write(
    """
    This application uses a Random Forest machine learning
    model for speech emotion recognition.

    The audio is converted into 40 MFCC
    (Mel-Frequency Cepstral Coefficients) features,
    which are then used by the trained model to predict
    the speaker's emotional state.
    """
)


# =========================================================
# MODEL DETAILS
# =========================================================

st.subheader("📌 Model Details")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info("🤖 **Model**\n\nRandom Forest")

with col2:
    st.info("🎵 **Features**\n\n40 MFCC")

with col3:
    st.info("🎙️ **Audio**\n\nWAV")

with col4:
    st.info("🎭 **Emotions**\n\n8 Classes")


# =========================================================
# SUPPORTED EMOTIONS
# =========================================================

st.subheader("🎭 Supported Emotions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info("😐 Neutral")
    st.info("😌 Calm")

with col2:
    st.info("😊 Happy")
    st.info("😢 Sad")

with col3:
    st.info("😠 Angry")
    st.info("😨 Fearful")

with col4:
    st.info("🤢 Disgust")
    st.info("😲 Surprised")


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI Speech Emotion Recognition | "
    "Random Forest + MFCC | CodeAlpha ML Project"
)