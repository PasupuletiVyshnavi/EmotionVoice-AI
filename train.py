import os
import glob
import numpy as np
import librosa
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras import layers, models

DATA_PATH = "data/ravdess/Audio_Speech_Actors_01-24"
MODEL_DIR = "models"
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_FRAMES = 174

EMOTIONS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised"
]


def extract_features(file_path):
    audio, _ = librosa.load(
        file_path, sr=SAMPLE_RATE, mono=True
    )
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC
    ).T

    # Make every recording the same length.
    if len(mfcc) < MAX_FRAMES:
        mfcc = np.pad(
            mfcc,
            ((0, MAX_FRAMES - len(mfcc)), (0, 0))
        )
    else:
        mfcc = mfcc[:MAX_FRAMES]

    return mfcc.astype(np.float32)


def load_dataset():
    features, labels = [], []

    files = glob.glob(
        os.path.join(DATA_PATH, "Actor_*", "*.wav")
    )

    if not files:
        raise FileNotFoundError(
            f"No WAV files found in {DATA_PATH}. "
            "Check that the RAVDESS speech ZIP was extracted correctly."
        )

    for file_path in files:
        filename = os.path.basename(file_path)
        parts = filename.split("-")

        if len(parts) != 7:
            continue

        # RAVDESS: modality 03 = audio-only,
        # vocal channel 01 = speech (not song).
        if parts[0] != "03" or parts[1] != "01":
            continue

        emotion_id = int(parts[2])
        label = emotion_id - 1

        try:
            features.append(extract_features(file_path))
            labels.append(label)
        except Exception as exc:
            print(f"Skipping {filename}: {exc}")

    if not features:
        raise ValueError("No valid speech recordings were loaded.")

    return np.array(features), np.array(labels)


def build_model():
    model = models.Sequential([
        layers.Input(shape=(MAX_FRAMES, N_MFCC)),
        layers.Conv1D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(len(EMOTIONS), activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    print("Loading RAVDESS speech dataset...")
    X, y = load_dataset()

    print("Features:", X.shape)
    print("Recordings:", len(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model()
    model.summary()

    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=30,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            )
        ]
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy:.2%}")

    predictions = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(
        y_test,
        predictions,
        labels=list(range(len(EMOTIONS))),
        target_names=EMOTIONS,
        zero_division=0
    ))

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "emotion_model.keras"))
    np.save(
        os.path.join(MODEL_DIR, "emotion_labels.npy"),
        np.array(EMOTIONS)
    )

    print("Model saved in:", MODEL_DIR)

