const state = {
  selectedFile: null,
  images: {
    original: null,
    roi: null,
  },
  activeView: "original",
};

const els = {
  modelStatus: document.getElementById("modelStatus"),
  fileInput: document.getElementById("fileInput"),
  analyzeButton: document.getElementById("analyzeButton"),
  healthySampleButton: document.getElementById("healthySampleButton"),
  anomalySampleButton: document.getElementById("anomalySampleButton"),
  previewImage: document.getElementById("previewImage"),
  viewer: document.getElementById("viewer"),
  decisionText: document.getElementById("decisionText"),
  riskRing: document.getElementById("riskRing"),
  riskValue: document.getElementById("riskValue"),
  probabilityText: document.getElementById("probabilityText"),
  thresholdText: document.getElementById("thresholdText"),
  predictionLabel: document.getElementById("predictionLabel"),
  confidenceLabel: document.getElementById("confidenceLabel"),
  roiLabel: document.getElementById("roiLabel"),
  checkpointLabel: document.getElementById("checkpointLabel"),
  noteBox: document.getElementById("noteBox"),
};

const steps = [
  document.getElementById("stepInput"),
  document.getElementById("stepPreprocess"),
  document.getElementById("stepModel"),
].filter(Boolean);

function setModelStatus(text, mode) {
  els.modelStatus.classList.remove("online", "offline");
  if (mode) els.modelStatus.classList.add(mode);
  els.modelStatus.querySelector("span:last-child").textContent = text;
}

function setActiveStep(index) {
  steps.forEach((step, i) => {
    step.classList.toggle("active", i <= index);
  });
}

function setImage(view) {
  state.activeView = view;
  document.querySelectorAll(".tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });

  const src = state.images[view];
  if (src) {
    els.previewImage.src = src;
    els.viewer.classList.add("has-image");
  }
}

function setBusy(isBusy) {
  els.analyzeButton.disabled = isBusy || !state.selectedFile;
  els.healthySampleButton.disabled = isBusy;
  els.anomalySampleButton.disabled = isBusy;
  els.analyzeButton.textContent = isBusy ? "Analyzing..." : "Analyze";
}

function percent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return `${Math.round(Number(value) * 100)}%`;
}

function riskColor(probability) {
  if (probability >= 0.65) return "#b42318";
  if (probability >= 0.4) return "#b7791f";
  return "#1f7a3e";
}

function updateResult(data) {
  const probability = Number(data.probability);
  const degrees = Math.max(0, Math.min(1, probability)) * 360;
  const color = riskColor(probability);

  els.riskRing.style.background = `conic-gradient(${color} ${degrees}deg, #e6ded2 ${degrees}deg)`;
  els.riskValue.textContent = percent(probability);
  els.probabilityText.textContent = data.probability_text || percent(probability);
  els.thresholdText.textContent = `Decision threshold: ${data.threshold?.toFixed?.(3) || data.threshold || "0.500"}`;
  els.decisionText.textContent = data.prediction === 1 ? "Anomaly suspected" : "No anomaly suspected";
  els.predictionLabel.textContent = data.prediction === 1 ? "Anomaly" : "Healthy / empty";
  els.confidenceLabel.textContent = data.confidence || "--";
  els.roiLabel.textContent = data.roi_box || "--";
  els.checkpointLabel.textContent = data.checkpoint_name || data.model_status || "--";

  els.noteBox.className = "note-box";
  if (data.prediction === 1) {
    els.noteBox.classList.add("alert");
    els.noteBox.textContent = "Model flagged this image as suspicious. Review the original image and ROI/model-input view together.";
  } else {
    els.noteBox.classList.add("ok");
    els.noteBox.textContent = "Model did not flag a strong anomaly pattern. This remains a research prototype, not a diagnosis.";
  }
  if (data.sample_note) {
    els.noteBox.textContent = `${data.sample_note} ${els.noteBox.textContent}`;
  }

  state.images.original = data.original_image || state.images.original;
  state.images.roi = data.roi_image || data.original_image || state.images.original;
  setImage("original");
  setActiveStep(2);
}

async function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function analyzeSelectedFile() {
  if (!state.selectedFile) return;

  setBusy(true);
  setActiveStep(1);

  const dataUrl = await fileToDataUrl(state.selectedFile);
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      filename: state.selectedFile.name,
      data_url: dataUrl,
    }),
  });

  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Prediction failed");
  updateResult(data);
}

async function runSample(caseName) {
  setBusy(true);
  setActiveStep(1);

  const response = await fetch(`/api/sample?case=${encodeURIComponent(caseName)}`);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Sample prediction failed");
  state.selectedFile = null;
  state.images.original = null;
  state.images.roi = null;
  updateResult(data);
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    if (data.model_loaded) {
      setModelStatus(`Model ready: ${data.checkpoint_name}`, "online");
      els.thresholdText.textContent = `Decision threshold: ${Number(data.threshold).toFixed(3)}`;
    } else {
      setModelStatus("Server running without checkpoint", "offline");
    }
  } catch {
    setModelStatus("Start server.py to run inference", "offline");
  }
}

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => setImage(button.dataset.view));
});

els.fileInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;

  state.selectedFile = file;
  setActiveStep(0);
  els.analyzeButton.disabled = false;

  if (file.type.startsWith("image/")) {
    state.images.original = await fileToDataUrl(file);
    state.images.roi = null;
    setImage("original");
  } else {
    state.images.original = null;
    state.images.roi = null;
    els.viewer.classList.remove("has-image");
    els.decisionText.textContent = "NIfTI selected";
    els.probabilityText.textContent = "Ready for backend preprocessing";
  }
});

els.analyzeButton.addEventListener("click", () => {
  analyzeSelectedFile()
    .catch((error) => {
      els.noteBox.className = "note-box alert";
      els.noteBox.textContent = error.message;
    })
    .finally(() => setBusy(false));
});

els.healthySampleButton.addEventListener("click", () => {
  runSample("healthy")
    .catch((error) => {
      els.noteBox.className = "note-box alert";
      els.noteBox.textContent = error.message;
    })
    .finally(() => setBusy(false));
});

els.anomalySampleButton.addEventListener("click", () => {
  runSample("anomaly")
    .catch((error) => {
      els.noteBox.className = "note-box alert";
      els.noteBox.textContent = error.message;
    })
    .finally(() => setBusy(false));
});

checkHealth();
