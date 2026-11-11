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
  function setChip(node, label, variant) {
    node.textContent = label;
    node.className = `chip chip-compact ${variant}`;
  }

  function renderVerdict(data) {
    const verdict = data.verdict || {};
    const state = verdict.state || "identified";
    const banner = el("verdict-banner");

    let tone;
    let kicker;
    let detail;
    if (state === "unidentified") {
      tone = "is-unknown";
      kicker = "Identification failed";
      detail =
        (data.uncertainty && data.uncertainty.message
          ? data.uncertainty.message
          : "Unable to identify this fruit confidently.") +
        " No condition assessment is offered, because naming the wrong fruit would make one meaningless.";
    } else if (state === "identified_condition_uncertain") {
      tone = "is-uncertain";
      kicker = "Fruit identified · condition unclear";
      detail = `This is a ${String(verdict.fruit).toLowerCase()}, but fresh and rotten probabilities are too close to call from this photograph.`;
    } else {
      tone = data.condition === "fresh" ? "is-fresh" : "is-rotten";
      kicker = "Assessment complete";
      detail = `The model identifies this as a ${String(verdict.fruit).toLowerCase()} and reads the condition as ${data.condition}.`;
    }

    banner.className = `verdict-banner ${tone}`;
    el("verdict-kicker").textContent = kicker;
    el("verdict-headline").textContent = verdict.headline || "—";
    el("verdict-detail").textContent = detail;

    setChip(
      el("chip-fruit"),
      state === "unidentified" ? "Fruit not identified" : "Fruit identified",
      state === "unidentified" ? "chip-warn" : "chip-ok"
    );
    setChip(
      el("chip-condition"),
      state === "unidentified" ? "Condition not assessed" : state === "identified_condition_uncertain" ? "Condition uncertain" : "Condition confident",
      state === "unidentified" ? "chip-muted" : state === "identified_condition_uncertain" ? "chip-warn" : "chip-ok"
    );
  }

  function renderResult(data) {
    lastUploadName = (data.image_url || "").split("/").pop() || null;

    const state = (data.verdict && data.verdict.state) || "identified";
    const unidentified = state === "unidentified";
    const conditionUncertain = state === "identified_condition_uncertain";
    const fruitName = (data.verdict && data.verdict.fruit) || data.fruit;

    el("result-image").src = data.image_url;
    el("result-image").alt = `Analysed image: ${data.image_name}`;
    const info = data.image_info || {};
    el("result-image-meta").textContent =
      `${info.width || "?"}×${info.height || "?"} px · brightness ${Math.round(info.brightness || 0)}/255`;

    el("results-meta").textContent =
      `Analysed at ${new Date().toLocaleTimeString()} · inference ${Math.round(data.inference_ms)} ms`;

    renderVerdict(data);

    /* ---- identification card ---- */
    el("result-fruit").textContent = unidentified ? "Not identified" : fruitName;
    if (unidentified) {
      el("result-class").textContent =
        `Closest supported class: ${data.predicted_class} (${pct(data.identification_confidence)}). ` +
        "Below the acceptance level, so no fruit is claimed.";
    } else {
      el("result-class").textContent = `Predicted class: ${data.predicted_class}`;
    }
    setChip(
      el("identification-status"),
      unidentified ? "Uncertain" : "Identified",
      unidentified ? "chip-warn" : "chip-ok"
    );
    el("result-confidence").textContent = pct(data.identification_confidence);
    el("confidence-bar").style.width = `${clampPct(data.identification_confidence)}%`;
    el("result-confidence-note").textContent =
      "Summed probability of this fruit's fresh and rotten classes — the eight fruits together account for 100%.";

    /* ---- condition card ---- */
    const conditionCard = el("condition-card");
    conditionCard.classList.toggle("is-muted", unidentified);
    el("result-condition").textContent = unidentified
      ? "Not assessed"
      : conditionUncertain
        ? "Uncertain — too close to call"
        : data.condition === "fresh"
          ? "Looks fresh"
          : "Looks rotten";
    el("result-condition").className =
      "condition-badge " + (unidentified ? "" : conditionUncertain ? "uncertain" : data.condition);

    const showValues = !unidentified;
    el("fresh-value").textContent = showValues ? pct(data.fresh_probability) : "—";
    el("rotten-value").textContent = showValues ? pct(data.rotten_probability) : "—";
    el("fresh-bar").style.width = showValues ? `${clampPct(data.fresh_probability)}%` : "0%";
    el("rotten-bar").style.width = showValues ? `${clampPct(data.rotten_probability)}%` : "0%";
    el("result-condition-confidence").textContent = showValues ? pct(data.condition_confidence) : "—";
    el("condition-confidence-bar").style.width = showValues
      ? `${clampPct(data.condition_confidence)}%`
      : "0%";

    setChip(
      el("condition-status"),
      unidentified ? "Not assessed" : conditionUncertain ? "Uncertain" : "Confident",
      unidentified ? "chip-muted" : conditionUncertain ? "chip-warn" : "chip-ok"
    );

    /* ---- spoilage assessment ---- */
    const edibility = data.edibility || {};
    const edibilityLabel = el("result-edibility");
    edibilityLabel.textContent = edibility.label || "Visual assessment is uncertain";
    edibilityLabel.className = "edibility-label " + (
      edibility.status === "appears_fresh" ? "fresh"
        : edibility.status === "shows_spoilage" ? "rotten"
          : edibility.status === "not_assessed" ? "muted" : "uncertain"
    );
    el("result-edibility-note").textContent = edibility.reason || edibility.note || "";
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
  function tile(value, label, note) {
    const wrapper = document.createElement("div");
    wrapper.className = "model-tile";
    const strong = document.createElement("span");
    strong.className = "model-tile-value";
    strong.textContent = value;
    const caption = document.createElement("span");
    caption.className = "model-tile-label";
    caption.textContent = label;
    wrapper.append(strong, caption);
    if (note) {
      const small = document.createElement("span");
      small.className = "model-tile-note";
      small.textContent = note;
      wrapper.appendChild(small);
    }
    return wrapper;
  }

  async function loadModelSummary() {
    const target = el("model-summary");
    const tiles = el("model-tiles");
    try {
      const response = await fetch("/api/model-info");
      if (!response.ok) throw new Error("unavailable");
      const info = await response.json();
      const n = info.test_split ? info.test_split.n : "—";

      target.remove();
      tiles.innerHTML = "";
      tiles.append(
        tile(`${(info.test_accuracy * 100).toFixed(1)}%`, "16-class accuracy", `${n} held-out images`),
        tile(`${(info.condition_level_accuracy * 100).toFixed(1)}%`, "fresh vs rotten accuracy", "what the result card shows"),
        tile(`${(info.macro_f1 * 100).toFixed(1)}%`, "macro F1", "all 16 classes weighted equally"),
        tile("No", "cross-section", "not validated")
      );

      const gate = info.gate || {};
      if (gate.min_fruit_confidence !== undefined) {
        el("gate-summary").textContent =
          `Fruit acceptance level ${(gate.min_fruit_confidence * 100).toFixed(0)}%, condition ` +
          `${(gate.condition_min_confidence * 100).toFixed(0)}%, fitted on the validation split. ` +
          `On the test split the gate declined to name a fruit for ` +
          `${(gate.test_rejection_rate * 100).toFixed(1)}% of images.`;
      }

      const calibration = info.calibration || {};
      if (calibration.temperature) {
        el("metrics-note").textContent =
          `Temperature scaling T=${Number(calibration.temperature).toFixed(2)} fitted on validation. ` +
          `Expected calibration error ${(calibration.ece_after * 100).toFixed(2)}% after calibration ` +
          `(${(calibration.ece_before * 100).toFixed(2)}% before).`;
      }
    } catch (error) {
      target.textContent =
        "Evaluation metrics are not available yet. Run the evaluation script to populate this section.";
    }
  }

  loadModelSummary();
})();
