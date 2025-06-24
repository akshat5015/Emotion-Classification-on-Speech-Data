
# Speech Emotion Recognition

## Project Description

This project implements a deep learning-based Speech Emotion Recognition (SER) system that classifies emotions from short audio clips of speech. The model is trained to identify multiple emotions (such as neutral, calm, happy, sad, angry, fearful, disgust, and surprised) using advanced feature extraction and a convolutional neural network (CNN) architecture. The system is designed for robust performance and can be used via a web interface or command-line scripts.

---

## Pre-processing Methodology

- **Audio Standardization:**  
  All audio files are resampled to 16kHz and trimmed or padded to exactly 3 seconds to ensure consistency.

- **Feature Extraction:**  
  For each audio file, two channels of features are extracted:
  - **Channel 1:** Mel-spectrogram (32 Mel bands, dB scale)
  - **Channel 2:** Stacked features including MFCCs (13 coefficients), chroma, and spectral contrast  
  Each channel is normalized (zero mean, unit variance) and shaped to (32, 94, 2).

- **Targeted Data Augmentation:**  
  To address class imbalance, minority classes are augmented using:
  - Additive noise
  - Time masking
  - Frequency masking
  - Random gain adjustment

- **Label Encoding:**  
  Emotion labels are encoded using `LabelEncoder` for compatibility with the neural network.

---

## Model Pipeline

1. **Input:**  
   Pre-processed features of shape (32, 94, 2) per audio sample.

2. **CNN Architecture:**  
   - Three convolutional blocks with BatchNormalization, MaxPooling, and Dropout
   - Global Average Pooling to reduce parameters
   - Dense layer with ReLU activation and Dropout
   - Output layer: Softmax over emotion classes

3. **Training Strategy:**  
   - Loss: Categorical Crossentropy
   - Optimizer: Adam
   - Early stopping, learning rate reduction, and model checkpointing
   - Dynamic class weighting for imbalanced data

4. **Evaluation:**  
   - Validation split: 20%
   - Model selection based on best validation accuracy

---

## Accuracy Metrics

| Metric                   | Value      | Status |
|--------------------------|------------|--------|
| **Validation Accuracy**  | 82.89%     | PASS   |
| **Weighted F1-score**    | 0.83       | PASS   |
| **All Class Accuracy**   | >75%       | PASS   |



- Detailed classification report and confusion matrix are generated after training.

---

