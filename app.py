
import os
import tempfile
import numpy as np
import librosa
import tensorflow as tf

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "emotion_model.keras"
)
LABELS_PATH = os.path.join(
    BASE_DIR, "models", "emotion_labels.npy"
)
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_FRAMES = 174

model = None
labels = None


def load_model_files():
    global model, labels

    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        return False

    model = tf.keras.models.load_model(MODEL_PATH)
    labels = np.load(LABELS_PATH, allow_pickle=False).tolist()
    return True


def extract_features(file_path):
    audio, _ = librosa.load(
        file_path, sr=SAMPLE_RATE, mono=True
    )

    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC
    ).T

    if len(mfcc) < MAX_FRAMES:
        mfcc = np.pad(
            mfcc,
            ((0, MAX_FRAMES - len(mfcc)), (0, 0))
        )
    else:
        mfcc = mfcc[:MAX_FRAMES]

    return mfcc.astype(np.float32)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or labels is None:
        return jsonify({
            "error": "Model not found. Train the model first."
        }), 503

    if "audio" not in request.files:
        return jsonify({"error": "Please select an audio file."}), 400

    uploaded = request.files["audio"]

    if not uploaded.filename:
        return jsonify({"error": "No file selected."}), 400

    filename = secure_filename(uploaded.filename)

    if not filename.lower().endswith(".wav"):
        return jsonify({
            "error": "Please upload a WAV audio file."
        }), 400

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as temp:
            temp_path = temp.name
            uploaded.save(temp_path)

        features = extract_features(temp_path)
        probabilities = model.predict(
            np.expand_dims(features, axis=0),
            verbose=0
        )[0]

        best_index = int(np.argmax(probabilities))

        scores = {
            labels[i]: round(float(probabilities[i]) * 100, 2)
            for i in range(len(labels))
        }

        return jsonify({
            "emotion": labels[best_index],
            "confidence": round(
                float(probabilities[best_index]) * 100, 2
            ),
            "scores": scores
        })

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Could not process this audio. Please try another WAV file."
        }), 400

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "File is too large. Maximum size is 20 MB."}), 413


if __name__ == "__main__":
# Load the model when the Flask application starts
    if load_model_files():
        print("Emotion model loaded successfully.")
    else:
        print("ERROR: Model files are missing.")
    
    if __name__ == "__main__":
        app.run(
            debug=False,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 5000))
        )
    
