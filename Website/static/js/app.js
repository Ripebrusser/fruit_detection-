/* FruitVision front-end: upload handling, API call, result rendering. */
(function () {
  "use strict";

  const MAX_BYTES = 8 * 1024 * 1024;
  const ALLOWED = ["image/jpeg", "image/png", "image/webp", "image/bmp"];

  const el = (id) => document.getElementById(id);
  const dropzone = el("dropzone");
  const fileInput = el("file-input");
  const dropzoneEmpty = el("dropzone-empty");
  const preview = el("preview");
  const previewImage = el("preview-image");
  const previewCaption = el("preview-caption");
  const form = el("upload-form");
  const errorBox = el("form-error");
  const analyseButton = el("analyse-button");
  const clearButton = el("clear-button");
  const results = el("results");

  let selectedFile = null;
  let objectUrl = null;
  let lastUploadName = null;

  /* ------------------------------------------------------------- helpers */
  const pct = (value) => `${(Number(value) * 100).toFixed(1)}%`;
  const clampPct = (value) => Math.min(100, Math.max(0, Number(value) * 100));

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
  }

  function clearError() {
    errorBox.textContent = "";
    errorBox.hidden = true;
  }

  function revokePreview() {
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      objectUrl = null;
    }
  }

  function validate(file) {
    if (!file) return "Please choose an image file.";
    if (file.type && !ALLOWED.includes(file.type)) {
      return "Unsupported file type. Please upload a JPG, PNG, WEBP or BMP image.";
    }
    if (!file.type && !/\.(jpe?g|png|webp|bmp)$/i.test(file.name)) {
      return "Unsupported file type. Please upload a JPG, PNG, WEBP or BMP image.";
    }
    if (file.size > MAX_BYTES) {
      return `That image is ${(file.size / (1024 * 1024)).toFixed(1)} MB. The maximum is 8 MB.`;
    }
    if (file.size === 0) return "That file is empty.";
    return null;
  }

  /* ------------------------------------------------------------- preview */
  function setFile(file) {
    clearError();
    const problem = validate(file);
    if (problem) {
      showError(problem);
      return;
    }
    selectedFile = file;
    revokePreview();
    objectUrl = URL.createObjectURL(file);
    previewImage.src = objectUrl;
    previewImage.alt = `Preview of ${file.name}`;
    previewCaption.textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
    preview.hidden = false;
    dropzoneEmpty.hidden = true;
    dropzone.classList.add("has-image");
    analyseButton.disabled = false;
    clearButton.disabled = false;
  }

  function resetAll() {
    selectedFile = null;
    revokePreview();
    fileInput.value = "";
    preview.hidden = true;
    previewImage.removeAttribute("src");
    previewCaption.textContent = "";
    dropzoneEmpty.hidden = false;
    dropzone.classList.remove("has-image", "is-dragover");
    analyseButton.disabled = true;
    clearButton.disabled = true;
    clearError();
    results.hidden = true;
    setLoading(false);
    if (lastUploadName) {
      // Best-effort cleanup of the stored copy on the server.
      fetch(`/api/uploads/${encodeURIComponent(lastUploadName)}`, { method: "DELETE" }).catch(() => {});
      lastUploadName = null;
    }
    dropzone.focus({ preventScroll: true });
  }

  /* -------------------------------------------------------------- upload */
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) setFile(fileInput.files[0]);
  });

  ["dragenter", "dragover"].forEach((type) => {
    dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      dropzone.classList.add("is-dragover");
    });
  });
  ["dragleave", "dragend"].forEach((type) => {
    dropzone.addEventListener(type, () => dropzone.classList.remove("is-dragover"));
  });
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-dragover");
    const file = event.dataTransfer && event.dataTransfer.files ? event.dataTransfer.files[0] : null;
    if (file) setFile(file);
  });

  // Prevent the browser from opening a dropped file elsewhere on the page.
  ["dragover", "drop"].forEach((type) => {
    window.addEventListener(type, (event) => {
      if (!dropzone.contains(event.target)) event.preventDefault();
    });
  });

  clearButton.addEventListener("click", resetAll);

  function setLoading(isLoading) {
    analyseButton.classList.toggle("is-loading", isLoading);
    analyseButton.disabled = isLoading || !selectedFile;
    analyseButton.querySelector(".btn-label").textContent = isLoading ? "Analysing…" : "Analyse image";
    clearButton.disabled = isLoading || !selectedFile;
    if (!isLoading && !selectedFile) analyseButton.disabled = true;
  }

  /* --------------------------------------------------------------- render */
  function renderResult(data) {
    lastUploadName = (data.image_url || "").split("/").pop() || null;

    el("result-image").src = data.image_url;
    el("result-image").alt = `Analysed image: ${data.image_name}`;
    const info = data.image_info || {};
    el("result-image-meta").textContent =
      `${info.width || "?"}×${info.height || "?"} px · brightness ${Math.round(info.brightness || 0)}/255`;

    el("results-meta").textContent =
      `Analysed at ${new Date().toLocaleTimeString()} · inference ${Math.round(data.inference_ms)} ms`;

    const uncertain = data.uncertainty && data.uncertainty.rejected;
    el("result-fruit").textContent = uncertain ? "Uncertain" : data.fruit;
    el("result-class").textContent = uncertain
      ? `${data.uncertainty.message} Closest supported class: ${data.predicted_class}.`
      : `Predicted class: ${data.predicted_class}`;

    el("result-confidence").textContent = pct(data.identification_confidence);
    el("confidence-bar").style.width = `${clampPct(data.identification_confidence)}%`;
    el("result-confidence-note").textContent =
      "Share of the model's probability assigned to this fruit (all eight fruits consider fresh and rotten).";

    const conditionBadge = el("result-condition");
    conditionBadge.textContent = uncertain
      ? "Uncertain"
      : data.condition === "fresh" ? "Looks fresh / good" : "Looks rotten / bad";
    conditionBadge.className = "condition-badge " + (uncertain ? "" : data.condition);

    el("fresh-value").textContent = pct(data.fresh_probability);
    el("rotten-value").textContent = pct(data.rotten_probability);
    el("fresh-bar").style.width = `${clampPct(data.fresh_probability)}%`;
    el("rotten-bar").style.width = `${clampPct(data.rotten_probability)}%`;

    const edibility = data.edibility || {};
    const edibilityLabel = el("result-edibility");
    edibilityLabel.textContent = edibility.label || "Visual assessment is uncertain";
    edibilityLabel.className =
      "edibility-label " +
      (edibility.status === "appears_fresh" ? "fresh" : edibility.status === "shows_spoilage" ? "rotten" : "uncertain");
    el("result-edibility-note").textContent = edibility.note || "";
    el("result-disclaimer").textContent = edibility.disclaimer || "";

    el("result-shelf").textContent = data.shelf_life.message;
    el("result-shelf-reason").textContent = data.shelf_life.reason || "";

    const list = el("probability-list");
    list.innerHTML = "";
    Object.entries(data.class_probabilities || {})
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .forEach(([name, value]) => {
        const li = document.createElement("li");
        const label = document.createElement("span");
        label.className = "prob-name";
        label.textContent = name;
        const score = document.createElement("span");
        score.className = "prob-value";
        score.textContent = pct(value);
        li.append(label, score);
        list.appendChild(li);
      });

    const warnings = data.warnings || [];
    const warningBox = el("warnings");
    const warningList = el("warning-list");
    warningList.innerHTML = "";
    if (warnings.length) {
      warnings.forEach((text) => {
        const li = document.createElement("li");
        li.textContent = text;
        warningList.appendChild(li);
      });
      warningBox.hidden = false;
    } else {
      warningBox.hidden = true;
    }

    el("model-footnote").textContent =
      `${data.model.name} · input ${data.model.input_size}×${data.model.input_size} · ` +
      `temperature ${Number(data.model.temperature).toFixed(2)} · ${data.notes.scope}`;

    results.hidden = false;
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /* ----------------------------------------------------------------- call */
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!selectedFile) {
      showError("Please choose an image first.");
      return;
    }
    clearError();
    setLoading(true);

    const body = new FormData();
    body.append("image", selectedFile, selectedFile.name);

    try {
      const response = await fetch("/api/analyze", { method: "POST", body });
      let payload;
      try {
        payload = await response.json();
      } catch (parseError) {
        throw new Error(`The server returned an unreadable response (HTTP ${response.status}).`);
      }
      if (!response.ok || !payload.ok) {
        const detail = payload && payload.detail ? ` (${payload.detail})` : "";
        throw new Error((payload && payload.error ? payload.error : "Analysis failed.") + detail);
      }
      renderResult(payload);
    } catch (error) {
      showError(error.message || "Something went wrong while analysing the image.");
    } finally {
      setLoading(false);
      analyseButton.disabled = !selectedFile;
      clearButton.disabled = !selectedFile;
    }
  });

  /* ----------------------------------------------------------- model card */
  async function loadModelSummary() {
    const target = el("model-summary");
    try {
      const response = await fetch("/api/model-info");
      if (!response.ok) throw new Error("unavailable");
      const info = await response.json();
      const accuracy = (info.test_accuracy * 100).toFixed(1);
      const f1 = (info.macro_f1 * 100).toFixed(1);
      const condition = (info.condition_level_accuracy * 100).toFixed(1);
      const n = info.test_split ? info.test_split.n : "—";
      target.textContent =
        `On ${n} held-out test images the 16-class model reaches ${accuracy}% accuracy ` +
        `(macro F1 ${f1}%). Fresh-versus-rotten accuracy, which is what the main result shows, is ${condition}%.`;
    } catch (error) {
      target.textContent =
        "Evaluation metrics are not available yet. Run the evaluation script to populate the model card.";
    }
  }

  loadModelSummary();
})();
