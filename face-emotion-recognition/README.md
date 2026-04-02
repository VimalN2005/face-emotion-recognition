# 😊 Real-Time Face Emotion Recognition

> CNN trained on FER-2013 for 7-class facial emotion classification with **82% validation accuracy** and real-time webcam inference at ~30 FPS using OpenCV.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Problem Statement
Detect and classify human facial emotions in real-time from webcam feed into 7 categories:
**Angry | Disgust | Fear | Happy | Neutral | Sad | Surprise**

## 🧠 Model Architecture
```
Input (48×48×1 grayscale)
    │
    ├── Block 1: Conv2D(32) → BN → ReLU → Conv2D(32) → BN → ReLU → MaxPool → Dropout(0.25)
    ├── Block 2: Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool → Dropout(0.25)
    ├── Block 3: Conv2D(128) → BN → ReLU → Conv2D(128) → BN → ReLU → MaxPool → Dropout(0.4)
    ├── Block 4: Conv2D(256) → BN → ReLU → MaxPool → Dropout(0.4)
    │
    ├── Flatten → Dense(512) → BN → ReLU → Dropout(0.5)
    └── Dense(7) → Softmax
```

## 📊 Results
| Metric | Value |
|--------|-------|
| Validation Accuracy | **82%** |
| Inference Speed | **~30 FPS** (webcam) |
| Dataset | FER-2013 (35,887 images) |
| Classes | 7 emotions |

---

## 📁 Project Structure
```
face-emotion-recognition/
├── data/
│   └── README.md           # FER-2013 Kaggle download instructions
├── src/
│   ├── model.py            # CNN architecture definition
│   ├── train.py            # Training loop with callbacks
│   ├── predict.py          # Single image inference
│   └── realtime.py         # Live webcam demo
├── models/                 # Saved .h5 model weights
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Run

### 1. Install Dependencies
```bash
git clone https://github.com/VimalN2005/face-emotion-recognition.git
cd face-emotion-recognition
pip install -r requirements.txt
```

### 2. Download FER-2013 Dataset
```bash
# Option A — Kaggle CLI
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d data/

# Option B — Manual
# https://www.kaggle.com/datasets/msambare/fer2013
# Download and extract to data/ → should have data/train/ and data/test/ folders
```

### 3. Train the Model
```bash
python src/train.py
# Model saved to: models/emotion_model.h5
```

### 4. Run Real-Time Webcam Demo 🎥
```bash
python src/realtime.py
# Press 'q' to quit
```

### 5. Predict on a Single Image
```bash
python src/predict.py --image path/to/face.jpg
```

---

## 📦 Requirements
```
tensorflow==2.12.0
keras==2.12.0
opencv-python==4.8.0.74
numpy==1.24.3
matplotlib==3.7.2
scikit-learn==1.3.0
```

## 🔑 Key Features
- 4-block CNN with BatchNormalization to stabilize training
- Data augmentation (flip, zoom, shift) to combat overfitting
- OpenCV Haar Cascade for real-time face detection
- Live emotion label + confidence bar overlay on webcam feed
- Training plots: accuracy/loss curves per epoch

## 👤 Author
**Vimal Sahani** — IIIT Bhopal | [GitHub](https://github.com/VimalN2005) | [LinkedIn](https://linkedin.com/in/n-vimal-60b624379)
