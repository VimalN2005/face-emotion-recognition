"""
model.py — 4-Block CNN Architecture for Facial Emotion Recognition
Face Emotion Recognition | Vimal Sahani
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Dropout, Flatten, Dense, Activation,
)
from tensorflow.keras.regularizers import l2


EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
IMG_SIZE       = 48
NUM_CLASSES    = 7


def build_emotion_cnn(input_shape=(48, 48, 1), num_classes=NUM_CLASSES) -> Sequential:
    """
    Build 4-block CNN for 7-class emotion classification.
    Architecture matches FER-2013 benchmark best practices.
    """
    model = Sequential(name="EmotionCNN")

    # ── Block 1 ─────────────────────────────────────────
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape,
                     kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # ── Block 2 ─────────────────────────────────────────
    model.add(Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # ── Block 3 ─────────────────────────────────────────
    model.add(Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # ── Block 4 ─────────────────────────────────────────
    model.add(Conv2D(256, (3, 3), padding="same", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # ── Classifier Head ──────────────────────────────────
    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    return model


def get_model_summary():
    model = build_emotion_cnn()
    model.summary()
    total = model.count_params()
    print(f"\nTotal Parameters: {total:,}")
    return model


if __name__ == "__main__":
    get_model_summary()
