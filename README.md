# 🎤 Speech Emotion Recognition — Full Deployment Guide

Detects human emotion (happy, sad, angry, neutral, calm, fearful, disgust,
surprised) from a voice clip using MFCC features + a CNN-LSTM deep learning
model, served through Flask and deployed on Render.

---

## 📁 Project Structure

```
speech-emotion-recognition/
├── app.py                    # Flask backend (serves UI + /predict API)
├── train_model.py            # Script to train the model (run locally/Colab)
├── feature_extractor.py      # Shared MFCC extraction logic
├── requirements.txt
├── Procfile                  # Tells Render how to start the app
├── .gitignore
├── templates/
│   └── index.html            # Colorful frontend UI
├── static/
│   ├── style.css
│   └── script.js
├── model/                    # Trained model goes here (you generate this)
│   ├── emotion_model.h5
│   └── label_classes.npy
└── uploads/                  # Temp folder for uploaded audio (auto-cleared)
```

---

## ⚠️ IMPORTANT: Read This First

Render's free tier has **no GPU and limited RAM/CPU** — you **cannot train**
the model on Render. You must **train the model on your own laptop or Google
Colab first**, then commit the resulting `model/emotion_model.h5` file to
GitHub. Render only *runs* the already-trained model.

---

## STEP 1 — Set Up Your Local Project Folder

Create a folder and place all the files from this project inside it, keeping
the exact folder structure shown above (`templates/`, `static/`, `model/`).

```bash
mkdir speech-emotion-recognition
cd speech-emotion-recognition
# copy all provided files into this folder, preserving folder structure
```

---

## STEP 2 — Install Dependencies Locally

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## STEP 3 — Download the Dataset

**RAVDESS** (required):
1. Go to: https://zenodo.org/record/1188976
2. Download **"Audio_Speech_Actors_01-24.zip"**
3. Unzip it into `dataset/RAVDESS/` so you get:
   ```
   dataset/RAVDESS/Actor_01/*.wav
   dataset/RAVDESS/Actor_02/*.wav
   ...
   dataset/RAVDESS/Actor_24/*.wav
   ```

**TESS** (optional, improves accuracy):
1. Go to: https://tspace.library.utoronto.ca/handle/1807/24487
2. Unzip into `dataset/TESS/` (any structure with `.wav` files works — the
   script searches recursively and reads the emotion word from the filename)

---

## STEP 4 — Train the Model

```bash
python train_model.py --data_dir ./dataset --epochs 60
```

- Takes ~10–30 minutes on CPU depending on dataset size.
- When finished you'll see:
  ```
  Final test accuracy: XX.XX%
  Saved model/emotion_model.h5 and model/label_classes.npy
  ```
- Typical accuracy with RAVDESS alone: **65–75%**. Adding TESS usually pushes
  this to **80%+**.

> 💡 Tip: If training is too slow on your laptop, run `train_model.py` in a
> **Google Colab** notebook instead (upload the dataset to Colab, run the
> script, then download `model/emotion_model.h5` and `model/label_classes.npy`
> back to your local `model/` folder).

---

## STEP 5 — Test Locally Before Deploying

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser. Upload a `.wav` file and
confirm you get a colorful result with an emoji and confidence bars.

Press `Ctrl+C` to stop the server once confirmed working.

---

## STEP 6 — Push to GitHub

```bash
git init
git add .
git commit -m "Speech Emotion Recognition - CNN-LSTM Flask app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/speech-emotion-recognition.git
git push -u origin main
```

> ✅ Make sure `model/emotion_model.h5` and `model/label_classes.npy` are
> actually committed (check on GitHub.com that they appear in the `model/`
> folder). If your `.h5` file is very large (>100MB), GitHub will reject it —
> in that case use [Git LFS](https://git-lfs.com/):
> ```bash
> git lfs install
> git lfs track "*.h5"
> git add .gitattributes
> git add model/emotion_model.h5
> git commit -m "Track model with LFS"
> git push
> ```

---

## STEP 7 — Deploy to Render

1. Go to **https://render.com** → sign in with GitHub.
2. Click **New +** → **Web Service**.
3. Select your `speech-emotion-recognition` repository.
4. Fill in the settings exactly as below:

| Setting | Value |
|---|---|
| **Name** | speech-emotion-recognition |
| **Region** | closest to you |
| **Branch** | main |
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app --workers 1 --threads 2 --timeout 120` |
| **Instance Type** | Free |

5. Click **Create Web Service**.
6. Wait for the build to finish (5–10 minutes — TensorFlow + librosa are
   large packages). Watch the **Logs** tab for progress.
7. Once you see `Model loaded. Classes: [...]` in the logs, your app is live
   at the URL Render gives you (e.g. `https://speech-emotion-recognition.onrender.com`).

---

## 🛠 Render Free-Tier Optimizations Already Applied

These are baked into the provided code so the app runs smoothly on Render's
512MB free tier:

- `tensorflow-cpu` instead of full `tensorflow` (much smaller install, no
  wasted GPU-search overhead).
- Single Gunicorn **worker** with 2 **threads** — prevents loading the model
  into memory multiple times (which would crash a free instance).
- `OMP_NUM_THREADS=1` and TensorFlow intra/inter-op threads capped at 1 —
  keeps CPU thread contention low.
- Uploaded audio files are deleted immediately after prediction (`finally`
  block in `app.py`) so disk usage doesn't grow.
- 10MB upload limit prevents oversized files from stalling the free instance.
- Model files are committed directly to Git (not downloaded at runtime),
  avoiding extra network calls on a slow free-tier cold start.

> ℹ️ Free Render web services **spin down after 15 minutes of inactivity**.
> The first request after idling will take 30–60 seconds to "wake up" — this
> is normal and not a bug.

---

## 🧠 How It Works (Quick Explanation)

1. **Feature extraction**: Each audio clip is converted into 40
   Mel-Frequency Cepstral Coefficients (MFCCs) across 130 time frames —
   turning raw sound into a compact numeric fingerprint of pitch/tone.
2. **Model**: Two `Conv1D` layers scan the MFCCs for local acoustic patterns,
   feeding into two `LSTM` layers that learn how those patterns evolve over
   time — this combination captures both texture and rhythm of speech.
3. **Prediction**: The final `Dense(softmax)` layer outputs a probability
   for each of the 8 emotions; the highest one is shown with its emoji.

---

## 🎨 Frontend Features

- Gradient animated background with floating blobs
- Drag-and-drop or click-to-upload audio
- Built-in audio player preview before analyzing
- Emoji + color-coded result per emotion
- Full probability breakdown bar chart for all 8 emotions
- Fully responsive (mobile-friendly)

---

## 🩹 Troubleshooting

| Problem | Fix |
|---|---|
| Build fails on Render with memory error | Make sure `requirements.txt` uses `tensorflow-cpu`, not `tensorflow` |
| "Model not found" error on startup | Confirm `model/emotion_model.h5` and `model/label_classes.npy` were pushed to GitHub |
| Predictions seem random/wrong | Make sure you didn't edit `feature_extractor.py` after training — train and inference must use identical feature settings |
| App sleeps / slow first request | Normal on Render free tier — upgrade to a paid instance to avoid spin-down |
| Large .h5 file rejected by GitHub | Use Git LFS (see Step 6) |

---

## 📈 Ideas to Improve Accuracy Later

- Add TESS and EMO-DB datasets together for more training data
- Add data augmentation (pitch shift, time stretch, noise injection)
- Try adding `Attention` layers on top of the LSTM output
- Balance classes if some emotions have far fewer samples than others

---

Built with 🎯 CNN + LSTM · MFCC · TensorFlow · Flask · Render
