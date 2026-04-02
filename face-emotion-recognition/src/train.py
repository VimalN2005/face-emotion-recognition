"""
train.py — Training Loop with Callbacks & Augmentation
Face Emotion Recognition | Vimal Sahani

Run: python src/train.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping,
    ReduceLROnPlateau, TensorBoard,
)
from src.model import build_emotion_cnn, IMG_SIZE, NUM_CLASSES

# ── Config ──────────────────────────────────────────────────────────────────
TRAIN_DIR   = "data/train"
TEST_DIR    = "data/test"
MODEL_PATH  = "models/emotion_model.h5"
BATCH_SIZE  = 64
EPOCHS      = 80
IMG_SHAPE   = (IMG_SIZE, IMG_SIZE)
LR          = 1e-3


def get_data_generators():
    """Create augmented train generator and validation generator."""

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SHAPE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SHAPE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return train_gen, val_gen


def get_callbacks():
    """Setup training callbacks."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs",   exist_ok=True)

    return [
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        TensorBoard(log_dir="logs", histogram_freq=1),
    ]


def plot_history(history, save_path="models/training_curves.png"):
    """Plot and save accuracy + loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train Acc",  color="royalblue")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc",    color="tomato")
    axes[0].set_title("Accuracy per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train Loss", color="royalblue")
    axes[1].plot(history.history["val_loss"], label="Val Loss",   color="tomato")
    axes[1].set_title("Loss per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Face Emotion Recognition — Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Training curves saved → {save_path}")
    plt.show()


def train():
    print("=" * 50)
    print("  😊 Face Emotion Recognition — Training")
    print("=" * 50)

    train_gen, val_gen = get_data_generators()
    print(f"\nClasses: {train_gen.class_indices}")
    print(f"Train samples: {train_gen.n} | Val samples: {val_gen.n}")

    model = build_emotion_cnn()
    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    print("\n[Training started...]\n")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=get_callbacks(),
    )

    best_val_acc = max(history.history["val_accuracy"])
    print(f"\n✅ Training complete!")
    print(f"   Best Val Accuracy : {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")
    print(f"   Model saved       : {MODEL_PATH}")

    plot_history(history)
    return model, history


if __name__ == "__main__":
    train()
