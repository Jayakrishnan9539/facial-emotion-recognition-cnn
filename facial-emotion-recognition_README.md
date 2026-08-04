# Facial Emotion Recognition (CNN | TensorFlow/Keras)

A deep learning system that detects human emotions from facial images in real time, built with a custom Convolutional Neural Network trained on the FER-2013 dataset. Includes live webcam inference with face detection.

## Overview

This project classifies faces into **7 emotions**: `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`.

- **Model**: Custom CNN (3 convolutional blocks: 32 → 64 → 128 filters, BatchNorm + Dropout, dense classifier head)
- **Dataset**: FER-2013 (~35,000 labeled 48x48 grayscale face images)
- **Framework**: TensorFlow / Keras
- **Test Accuracy**: 61.51%
- **Real-time inference**: Live webcam demo using OpenCV Haar Cascade for face detection + the trained CNN for emotion classification, with in-browser video capture (Google Colab compatible)

## Results

| Metric | Score |
|---|---|
| Test Accuracy | 61.51% |
| Test Loss | 1.0228 |

**Per-class performance (F1-score):**

| Emotion | F1-score |
|---|---|
| Happy | 0.84 |
| Surprise | 0.73 |
| Neutral | 0.57 |
| Angry | 0.54 |
| Sad | 0.47 |
| Disgust | 0.46 |
| Fear | 0.41 |

*Disgust and fear are the hardest classes to predict — this is a known, well-documented limitation of the FER-2013 dataset itself (disgust has very few training samples, and fear/sad expressions visually overlap), consistent with published benchmarks on this dataset.*

## Model Architecture

```
Input (48x48x1 grayscale)
→ Conv2D(32) x2 → BatchNorm → MaxPool → Dropout
→ Conv2D(64) x2 → BatchNorm → MaxPool → Dropout
→ Conv2D(128) x2 → BatchNorm → MaxPool → Dropout
→ Flatten → Dense(256) → BatchNorm → Dropout
→ Dense(7, softmax)
```

Total parameters: ~1.47M

## How to Run

This project is built to run end-to-end in **Google Colab** (free tier, GPU recommended).

1. Open `facial_emotion_recognition_colab.py` in Google Colab (or copy each cell into a new notebook)
2. Enable a GPU runtime: `Runtime → Change runtime type → T4 GPU`
3. Run all cells in order:
   - Install dependencies
   - Download FER-2013 dataset
   - Build data generators
   - Build and compile the CNN
   - Train the model
   - Evaluate on the test set (accuracy, classification report, confusion matrix)
   - Save/export the trained model
   - Launch the real-time webcam emotion detector

## Real-Time Webcam Demo

The webcam module streams frames from the browser, detects faces with OpenCV's Haar Cascade, crops and preprocesses each detected face, and feeds it to the trained CNN for live emotion prediction — displayed as an overlay with the predicted emotion and confidence score.

## Tech Stack

- TensorFlow / Keras — CNN model, training, evaluation
- OpenCV — face detection (Haar Cascade)
- scikit-learn — evaluation metrics (classification report, confusion matrix)
- Matplotlib / Seaborn — training curves and confusion matrix visualization
- Google Colab — training environment (free GPU)

## Future Improvements

- Fine-tune with a larger/more balanced dataset (e.g. FER+ with cleaner labels) to improve minority-class performance (disgust, fear)
- Experiment with transfer learning (e.g. MobileNetV2) for higher accuracy
- Deploy as a web app (Flask/FastAPI + React) for browser-based inference without Colab

## License

This project uses the FER-2013 dataset, released under the Database Contents License (DbCL) v1.0.
