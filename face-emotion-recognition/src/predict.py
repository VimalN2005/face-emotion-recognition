"""
predict.py — Single Image Emotion Prediction
Face Emotion Recognition | Vimal Sahani

Usage:
    python src/predict.py --image path/to/face.jpg
"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from src.model import EMOTION_LABELS, IMG_SIZE

MODEL_PATH   = "models/emotion_model.h5"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def predict_image(image_path: str):
    model        = load_model(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    img  = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("[WARNING] No face detected. Predicting on full image.")
        face_roi = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        faces    = [(0, 0, img.shape[1], img.shape[0])]
    else:
        x, y, w, h = faces[0]
        face_roi   = cv2.resize(gray[y:y+h, x:x+w], (IMG_SIZE, IMG_SIZE))

    face_inp = face_roi.astype("float32") / 255.0
    face_inp = np.expand_dims(face_inp, axis=(0, -1))

    preds    = model.predict(face_inp, verbose=0)[0]
    top_idx  = np.argmax(preds)

    print(f"\n{'─'*35}")
    print(f"  Predicted Emotion: {EMOTION_LABELS[top_idx]}")
    print(f"  Confidence: {preds[top_idx]*100:.1f}%")
    print(f"{'─'*35}")
    print("\nAll Probabilities:")
    for i, (label, prob) in enumerate(zip(EMOTION_LABELS, preds)):
        bar = "█" * int(prob * 30)
        print(f"  {label:10s} {prob*100:5.1f}% {bar}")

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for (x, y, w, h) in faces:
        rect = plt.Rectangle((x, y), w, h, fill=False, color="lime", lw=2)
        ax1.add_patch(rect)
    ax1.set_title(f"Detected: {EMOTION_LABELS[top_idx]} ({preds[top_idx]*100:.1f}%)")
    ax1.axis("off")

    colors = ["#e74c3c" if i == top_idx else "#3498db" for i in range(len(EMOTION_LABELS))]
    ax2.barh(EMOTION_LABELS, preds * 100, color=colors)
    ax2.set_xlabel("Confidence (%)")
    ax2.set_title("Emotion Probabilities")
    ax2.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig("prediction_result.png", dpi=150)
    plt.show()
    print("\n✅ Result saved → prediction_result.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()
    predict_image(args.image)
