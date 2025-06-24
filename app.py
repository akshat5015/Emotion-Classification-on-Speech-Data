import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import pickle
import os
from io import BytesIO

TARGET_TIME = 94
TARGET_ROWS = 32
TARGET_CHANNELS = 2

def ensure_shape(array, target_rows, target_time):
    """Pad/crop array to target shape"""
    if array.shape[1] > target_time:
        array = array[:, :target_time]
    elif array.shape[1] < target_time:
        pad_width = ((0, 0), (0, target_time - array.shape[1]))
        array = np.pad(array, pad_width, mode='constant', constant_values=0)
    if array.shape[0] > target_rows:
        array = array[:target_rows, :]
    elif array.shape[0] < target_rows:
        pad_width = ((0, target_rows - array.shape[0]), (0, 0))
        array = np.pad(array, pad_width, mode='constant', constant_values=0)
    return array

def normalize_features(features):
    """Normalize features with robust handling"""
    if features.size == 0:
        return features
    mean = np.mean(features)
    std = np.std(features)
    if std == 0 or np.isnan(std):
        return features - mean
    return (features - mean) / std

def create_enhanced_features(audio_bytes, target_sr=16000, duration=3.0):
    """Process audio from bytes (Streamlit upload)"""
    try:
        y, sr = librosa.load(BytesIO(audio_bytes), sr=target_sr, duration=duration)
        
        if len(y) == 0:
            return np.zeros((TARGET_ROWS, TARGET_TIME, TARGET_CHANNELS))
        
        target_length = int(duration * sr)
        if len(y) > target_length:
            y = y[:target_length]
        else:
            y = np.pad(y, (0, max(0, target_length - len(y))), mode='constant')
        
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=32, n_fft=1024, hop_length=512)
        channel1 = librosa.power_to_db(mel, ref=np.max)
        channel1 = ensure_shape(channel1, TARGET_ROWS, TARGET_TIME)
        channel1 = normalize_features(channel1)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=1024, hop_length=512)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)
        channel2 = np.vstack([mfcc, chroma, contrast])
        channel2 = ensure_shape(channel2, TARGET_ROWS, TARGET_TIME)
        channel2 = normalize_features(channel2)
        
        return np.stack([channel1, channel2], axis=-1)
    
    except Exception:
        return np.zeros((TARGET_ROWS, TARGET_TIME, TARGET_CHANNELS))

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("C:/Users/aksha/OneDrive/Desktop/mars/emotion_model_v5_baseline_best.keras")

@st.cache_resource
def load_label_encoder():
    with open("C:/Users/aksha/OneDrive/Desktop/mars/emotion_model_v5_baseline_label_encoder.pkl", "rb") as f:
        return pickle.load(f)

st.title("Speech Emotion Recognition")
st.markdown("Upload a WAV file to analyze its emotional content")

uploaded_file = st.file_uploader("Choose audio file", type=["wav"])

if uploaded_file:
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes, format='audio/wav')
    
    if st.button("Analyze Emotion"):
        with st.spinner("Processing audio..."):
            model = load_model()
            le = load_label_encoder()
            
            features = create_enhanced_features(audio_bytes)
            
            if features.shape != (TARGET_ROWS, TARGET_TIME, TARGET_CHANNELS):
                st.error("Feature extraction failed. Please try another file.")
            else:
                features = np.expand_dims(features, axis=0)
                pred = model.predict(features, verbose=0)
                pred_class = np.argmax(pred, axis=1)
                emotion = le.inverse_transform(pred_class)[0]
                confidence = np.max(pred) * 100
                
                st.success(f"**Predicted Emotion:** {emotion.upper()}")
                st.metric("Confidence", f"{confidence:.2f}%")
                
                st.subheader("Confidence Distribution")
                classes = le.classes_
                confidences = {cls: f"{pred[0][i]*100:.1f}%" for i, cls in enumerate(classes)}
                st.table(confidences.items())
