"""
feature_extractor.py
---------------------
Shared MFCC feature extraction logic.
IMPORTANT: This exact function is used in BOTH train_model.py and app.py.
If you change anything here, you must retrain the model, otherwise
predictions will be wrong (train/inference mismatch).
"""

import numpy as np
import librosa

# ---- Fixed hyperparameters (keep identical everywhere) ----
SAMPLE_RATE = 22050
DURATION = 3.5          # seconds — RAVDESS/TESS clips are short, this covers most of them
N_MFCC = 40              # number of MFCC coefficients
MAX_LEN = 130            # fixed number of time frames (pad/truncate to this)


def extract_features(file_path):
    """
    Loads an audio file and returns a fixed-size MFCC feature matrix
    of shape (MAX_LEN, N_MFCC) -> ready to feed into CNN/LSTM.
    """
    try:
        # Load audio, resample to SAMPLE_RATE, trim to DURATION seconds
        audio, sr = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            duration=DURATION,
            res_type="kaiser_fast",
        )

        # Pad short clips with silence so every file has equal length
        target_len = int(SAMPLE_RATE * DURATION)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]

        # Extract MFCCs -> shape (N_MFCC, time_frames)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

        # Transpose -> shape (time_frames, N_MFCC)
        mfcc = mfcc.T

        # Pad or truncate time dimension to MAX_LEN
        if mfcc.shape[0] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode="constant")
        else:
            mfcc = mfcc[:MAX_LEN, :]

        return mfcc.astype(np.float32)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# RAVDESS filename emotion code -> label
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# TESS folder names usually already contain the emotion word, e.g. "OAF_happy"
TESS_EMOTION_KEYWORDS = [
    "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised", "ps"
]


def label_from_ravdess_filename(filename):
    """
    RAVDESS filenames look like: 03-01-06-01-02-01-12.wav
    The 3rd number (index 2) is the emotion code.
    """
    parts = filename.split("-")
    code = parts[2]
    return RAVDESS_EMOTION_MAP.get(code)


def label_from_tess_filename(filename):
    """
    TESS filenames/folders contain the emotion word directly, e.g.
    'YAF_back_happy.wav'. 'ps' means pleasant_surprise -> map to surprised.
    """
    fname = filename.lower()
    if "ps" in fname or "pleasant" in fname:
        return "surprised"
    for emo in ["neutral", "calm", "happy", "sad", "angry", "fear", "disgust", "surprise"]:
        if emo in fname:
            if emo == "fear":
                return "fearful"
            if emo == "surprise":
                return "surprised"
            return emo
    return None
