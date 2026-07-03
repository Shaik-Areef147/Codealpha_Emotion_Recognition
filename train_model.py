"""
train_model.py
---------------
Run this LOCALLY or on Google Colab (NOT on Render — Render free tier has
no GPU and will time out on training). This produces:
    model/emotion_model.h5
    model/label_classes.npy
Commit BOTH of those files to GitHub — app.py loads them at runtime.

HOW TO GET THE DATASET
-----------------------
1. RAVDESS (recommended, cleanest labels):
   https://zenodo.org/record/1188976
   Download "Audio_Speech_Actors_01-24.zip", unzip it.
   You'll get folders Actor_01 ... Actor_24, each full of .wav files.

2. (Optional, to boost accuracy) TESS:
   https://tspace.library.utoronto.ca/handle/1807/24487
   Unzip into a folder of its own.

FOLDER STRUCTURE EXPECTED BY THIS SCRIPT
------------------------------------------
dataset/
├── RAVDESS/
│   ├── Actor_01/*.wav
│   ├── Actor_02/*.wav
│   └── ...
└── TESS/                (optional)
    └── *.wav

HOW TO RUN
-----------
    pip install -r requirements.txt
    python train_model.py --data_dir ./dataset

This will take 10-30 minutes on CPU depending on dataset size.
"""

import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, BatchNormalization, Dropout,
    LSTM, Dense, Flatten
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from feature_extractor import (
    extract_features,
    label_from_ravdess_filename,
    label_from_tess_filename,
    MAX_LEN,
    N_MFCC,
)


def collect_dataset(data_dir):
    features, labels = [], []

    ravdess_dir = os.path.join(data_dir, "RAVDESS")
    if os.path.isdir(ravdess_dir):
        print("Scanning RAVDESS...")
        for root, _, files in os.walk(ravdess_dir):
            for f in files:
                if f.endswith(".wav"):
                    label = label_from_ravdess_filename(f)
                    if label is None:
                        continue
                    path = os.path.join(root, f)
                    feat = extract_features(path)
                    if feat is not None:
                        features.append(feat)
                        labels.append(label)

    tess_dir = os.path.join(data_dir, "TESS")
    if os.path.isdir(tess_dir):
        print("Scanning TESS...")
        for root, _, files in os.walk(tess_dir):
            for f in files:
                if f.endswith(".wav"):
                    label = label_from_tess_filename(f)
                    if label is None:
                        continue
                    path = os.path.join(root, f)
                    feat = extract_features(path)
                    if feat is not None:
                        features.append(feat)
                        labels.append(label)

    return np.array(features), np.array(labels)


def build_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(128, kernel_size=5, activation="relu", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(256, kernel_size=5, activation="relu"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),

        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./dataset")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    print("Extracting features (this takes a while)...")
    X, y = collect_dataset(args.data_dir)
    print(f"Total samples collected: {len(X)}")

    if len(X) == 0:
        print("No data found! Check your --data_dir path and folder structure.")
        return

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    print("Classes found:", list(le.classes_))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
    )

    model = build_model(input_shape=(MAX_LEN, N_MFCC), num_classes=y_categorical.shape[1])
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    loss, acc = model.evaluate(X_test, y_test)
    print(f"\nFinal test accuracy: {acc * 100:.2f}%")

    os.makedirs("model", exist_ok=True)
    model.save("model/emotion_model.h5")
    np.save("model/label_classes.npy", le.classes_)
    print("\nSaved model/emotion_model.h5 and model/label_classes.npy")
    print("Commit BOTH files to GitHub, then deploy app.py to Render.")


if __name__ == "__main__":
    main()
