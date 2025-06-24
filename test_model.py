import os
import glob
import tensorflow as tf
import numpy as np
import pickle
import librosa

# Constants
TARGET_TIME = 94
TARGET_ROWS = 32
TARGET_CHANNELS = 2

def ensure_shape(array, target_rows, target_time):
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
    if features.size == 0:
        return features
    mean = np.mean(features)
    std = np.std(features)
    if std == 0 or np.isnan(std):
        return features - mean
    return (features - mean) / std

def create_enhanced_features(filepath, target_sr=16000, duration=3.0):
    try:
        if not os.path.exists(filepath):
            return np.zeros((TARGET_ROWS, TARGET_TIME, TARGET_CHANNELS))
        y, sr = librosa.load(filepath, sr=target_sr, duration=duration)
        if len(y) == 0:
            return np.zeros((TARGET_ROWS, TARGET_TIME, TARGET_CHANNELS))
        target_length = int(duration * sr)
        if len(y) > target_length:
            y = y[:target_length]
        else:
            y = np.pad(y, (0, max(0, target_length - len(y))), mode='constant')
        n_fft = 1024
        hop_length = 512
        n_mels = 32
        # Channel 1
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        channel1 = librosa.power_to_db(mel, ref=np.max)
        channel1 = ensure_shape(channel1, TARGET_ROWS, TARGET_TIME)
        channel1 = normalize_features(channel1)
        # Channel 2
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
        channel2 = np.vstack([mfcc, chroma, contrast])
        channel2 = ensure_shape(channel2, TARGET_ROWS, TARGET_TIME)
        channel2 = normalize_features(channel2)
        features = np.stack([channel1, channel2], axis=-1)
        return features
    except Exception:
        return np.zeros((TARGET_ROWS, TARGET_TIME, TARGET_CHANNELS))

def predict_emotion(model, audio_file, le):
    features = create_enhanced_features(audio_file)
    features = np.expand_dims(features, axis=0)
    predictions = model.predict(features, verbose=0)
    pred_class = np.argmax(predictions)
    confidence = predictions[0][pred_class]
    emotion = le.inverse_transform([pred_class])[0]
    return emotion, confidence

def main():
    # Load model and label encoder
    model = tf.keras.models.load_model("emotion_model_v5_baseline_best.keras")
    with open("emotion_model_v5_baseline_label_encoder.pkl", 'rb') as f:
        le = pickle.load(f)

    print("Choose input type:")
    print("1. Single audio file")
    print("2. Folder containing audio files")
    choice = input("Enter 1 or 2: ").strip()

    if choice == '1':
        file_path = input("Enter the path to the audio file: ").strip()
        if not os.path.isfile(file_path):
            print("File not found")
            return
        emotion, confidence = predict_emotion(model, file_path, le)
        print(f"Predicted emotion: {emotion.upper()} (Confidence: {confidence:.2%})")

    elif choice == '2':
        folder_path = input("Enter the path to the folder: ").strip()
        if not os.path.isdir(folder_path):
            print("Folder not found")
            return
        wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
        if not wav_files:
            print("No WAV files found in the folder")
            return
        print(f"Found {len(wav_files)} WAV files. Processing...")
        for file_path in wav_files:
            emotion, confidence = predict_emotion(model, file_path, le)
            print(f"{os.path.basename(file_path)}: {emotion.upper()} (Confidence: {confidence:.2%})")

    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == '__main__':
    main()
