"""
realtime.py — Live Webcam Emotion Detection (~30 FPS)
Face Emotion Recognition | Vimal Sahani

Run: python src/realtime.py
Controls: Press 'q' to quit
"""

import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from src.model import EMOTION_LABELS, IMG_SIZE

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH   = "models/emotion_model.h5"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Emotion → BGR color mapping
EMOTION_COLORS = {
    "Angry":    (0, 0, 220),
    "Disgust":  (0, 140, 0),
    "Fear":     (180, 0, 180),
    "Happy":    (0, 200, 0),
    "Neutral":  (200, 200, 200),
    "Sad":      (220, 100, 0),
    "Surprise": (0, 200, 220),
}


def draw_emotion_bar(frame, x, y, w, emotion, confidence, color):
    """Draw label box + confidence bar above detected face."""
    label   = f"{emotion}: {confidence*100:.1f}%"
    bar_len = int(w * confidence)

    # Background rectangle for label
    cv2.rectangle(frame, (x, y - 40), (x + w, y), (30, 30, 30), -1)
    cv2.putText(frame, label, (x + 4, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    # Confidence bar
    cv2.rectangle(frame, (x, y - 6), (x + w, y - 2), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y - 6), (x + bar_len, y - 2), color, -1)


def preprocess_face(face_roi):
    """Resize and normalize a face ROI for model inference."""
    face = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)   # (48,48,1)
    face = np.expand_dims(face, axis=0)    # (1,48,48,1)
    return face


def run_webcam():
    """Main real-time loop."""
    print("[INFO] Loading model...")
    model = load_model(MODEL_PATH)

    print("[INFO] Opening camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    fps_timer   = time.time()
    frame_count = 0
    fps         = 0

    print("[INFO] Press 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        for (x, y, w, h) in faces:
            face_roi  = gray[y:y + h, x:x + w]
            face_inp  = preprocess_face(face_roi)
            preds     = model.predict(face_inp, verbose=0)[0]

            emotion_idx = np.argmax(preds)
            emotion     = EMOTION_LABELS[emotion_idx]
            confidence  = float(preds[emotion_idx])
            color       = EMOTION_COLORS[emotion]

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            draw_emotion_bar(frame, x, y, w, emotion, confidence, color)

        # FPS counter
        frame_count += 1
        if time.time() - fps_timer >= 1.0:
            fps         = frame_count
            frame_count = 0
            fps_timer   = time.time()

        cv2.putText(frame, f"FPS: {fps}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
        cv2.putText(frame, "Face Emotion Recognition", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Emotion Detection | Press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera released. Bye!")


if __name__ == "__main__":
    run_webcam()
