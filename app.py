"""
app.py
------
Flask backend for Speech Emotion Recognition.
Loads the trained model + label classes and serves:
    GET  /              -> frontend UI
    POST /predict        -> accepts an audio file, returns predicted emotion
"""

import os

# Keep TensorFlow lean on Render's free tier (single worker, low RAM)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from feature_extractor import extract_features, MAX_LEN, N_MFCC

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "model", "emotion_model.h5")
LABELS_PATH = os.path.join(APP_DIR, "model", "label_classes.npy")
UPLOAD_DIR = os.path.join(APP_DIR, "uploads")
ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "m4a", "webm"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
CLASSES = np.load(LABELS_PATH, allow_pickle=True)
print("Model loaded. Classes:", list(CLASSES))

EMOJI_MAP = {
    "neutral": "😐",
    "calm": "😌",
    "happy": "😄",
    "sad": "😢",
    "angry": "😠",
    "fearful": "😨",
    "disgust": "🤢",
    "surprised": "😲",
}

COLOR_MAP = {
    "neutral": "#9CA3AF",
    "calm": "#60A5FA",
    "happy": "#FBBF24",
    "sad": "#3B82F6",
    "angry": "#EF4444",
    "fearful": "#A855F7",
    "disgust": "#22C55E",
    "surprised": "#F472B6",
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use wav, mp3, ogg, m4a, or webm."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    try:
        features = extract_features(filepath)
        if features is None:
            return jsonify({"error": "Could not process audio file"}), 400

        features = np.expand_dims(features, axis=0)  # shape (1, MAX_LEN, N_MFCC)
        predictions = model.predict(features, verbose=0)[0]

        top_idx = int(np.argmax(predictions))
        emotion = str(CLASSES[top_idx])
        confidence = float(predictions[top_idx])

        all_probs = [
            {
                "emotion": str(CLASSES[i]),
                "emoji": EMOJI_MAP.get(str(CLASSES[i]), "🎵"),
                "color": COLOR_MAP.get(str(CLASSES[i]), "#8B5CF6"),
                "probability": float(predictions[i]),
            }
            for i in range(len(CLASSES))
        ]
        all_probs.sort(key=lambda x: x["probability"], reverse=True)

        return jsonify({
            "emotion": emotion,
            "emoji": EMOJI_MAP.get(emotion, "🎵"),
            "color": COLOR_MAP.get(emotion, "#8B5CF6"),
            "confidence": round(confidence * 100, 2),
            "all_probabilities": all_probs,
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
