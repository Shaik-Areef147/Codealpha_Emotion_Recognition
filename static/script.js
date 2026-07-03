const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const fileInfo = document.getElementById("fileInfo");
const fileName = document.getElementById("fileName");
const clearBtn = document.getElementById("clearBtn");
const audioPreview = document.getElementById("audioPreview");
const analyzeBtn = document.getElementById("analyzeBtn");

const resultCard = document.getElementById("resultCard");
const loading = document.getElementById("loading");
const resultContent = document.getElementById("resultContent");
const errorContent = document.getElementById("errorContent");
const resultEmoji = document.getElementById("resultEmoji");
const resultEmotion = document.getElementById("resultEmotion");
const resultConfidence = document.getElementById("resultConfidence");
const probBars = document.getElementById("probBars");
const errorMessage = document.getElementById("errorMessage");
const tryAgainBtn = document.getElementById("tryAgainBtn");
const errorRetryBtn = document.getElementById("errorRetryBtn");

let selectedFile = null;

// --- Drag & drop handlers ---
["dragenter", "dragover"].forEach(evt => {
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach(evt => {
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
  });
});

dropZone.addEventListener("drop", (e) => {
  const files = e.dataTransfer.files;
  if (files.length) handleFile(files[0]);
});

dropZone.addEventListener("click", (e) => {
  if (e.target.tagName !== "LABEL") fileInput.click();
});

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file) {
  const allowed = ["audio/wav", "audio/mpeg", "audio/mp3", "audio/ogg", "audio/x-m4a", "audio/webm", "audio/mp4"];
  const ext = file.name.split(".").pop().toLowerCase();
  const allowedExt = ["wav", "mp3", "ogg", "m4a", "webm"];

  if (!allowedExt.includes(ext)) {
    alert("Unsupported file type. Please upload WAV, MP3, OGG, M4A, or WEBM.");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    alert("File too large. Max size is 10MB.");
    return;
  }

  selectedFile = file;
  fileName.textContent = file.name;
  fileInfo.style.display = "flex";

  const url = URL.createObjectURL(file);
  audioPreview.src = url;
  audioPreview.style.display = "block";

  analyzeBtn.disabled = false;
}

clearBtn.addEventListener("click", () => {
  resetUpload();
});

function resetUpload() {
  selectedFile = null;
  fileInput.value = "";
  fileInfo.style.display = "none";
  audioPreview.style.display = "none";
  audioPreview.src = "";
  analyzeBtn.disabled = true;
  resultCard.style.display = "none";
}

tryAgainBtn.addEventListener("click", resetUpload);
errorRetryBtn.addEventListener("click", () => {
  resultCard.style.display = "none";
});

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  resultCard.style.display = "block";
  loading.style.display = "block";
  resultContent.style.display = "none";
  errorContent.style.display = "none";

  const formData = new FormData();
  formData.append("audio", selectedFile);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Prediction failed");
    }

    showResult(data);
  } catch (err) {
    showError(err.message);
  }
});

function showResult(data) {
  loading.style.display = "none";
  resultContent.style.display = "block";

  resultEmoji.textContent = data.emoji;
  resultEmotion.textContent = data.emotion;
  resultConfidence.textContent = `${data.confidence}% confidence`;
  resultEmotion.style.color = data.color;

  probBars.innerHTML = "";
  data.all_probabilities.forEach(item => {
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span class="prob-emoji">${item.emoji}</span>
      <span class="prob-label">${item.emotion}</span>
      <span class="prob-track">
        <span class="prob-fill" style="width:${(item.probability * 100).toFixed(1)}%; background:${item.color}"></span>
      </span>
      <span class="prob-value">${(item.probability * 100).toFixed(1)}%</span>
    `;
    probBars.appendChild(row);
  });
}

function showError(message) {
  loading.style.display = "none";
  errorContent.style.display = "block";
  errorMessage.textContent = message;
}
