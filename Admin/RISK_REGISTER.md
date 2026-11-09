# Risk Register

Likelihood and impact are rated Low / Medium / High. "Status" records whether the risk is
mitigated, accepted, or still open.

| ID | Risk | Likelihood | Impact | Mitigation | Status |
|----|------|-----------|--------|-----------|--------|
| R1 | **Shelf-life fabrication** — user expects "days until rotten", dataset has no time labels | High | High | No regression model is built. API returns `shelf_life.available = false` with the reason; UI prints the fixed message. Documented in `DATASET_DOCUMENTATION.md` §6. | Mitigated |
| R2 | **Overstated cross-section support** — cut fruit is a small, unlabelled fraction | High | High | No cross-section claim. `cross_section_validated: false` in `/api/model-info`; UI states the limit; pipeline keeps a hook for future labelled data. | Mitigated |
| R3 | **Data leakage through duplicates** — 7 exact and ~104 near-duplicate images | Medium | High | SHA-256 + pHash(≤4) union-find grouping; groups assigned atomically to one split; `test_no_duplicate_group_crosses_splits` enforces it. | Mitigated |
| R4 | **Uncalibrated probabilities presented as physical measurements** | High | Medium | Temperature scaling fitted on validation; ECE reported before/after; UI labels scores as summed model probabilities. | Mitigated |
| R5 | **Confident nonsense on non-fruit or unseen-fruit photos** | High | High | Calibrated confidence gate + entropy/margin checks + image-quality checks; explicit "unable to identify" message; warnings panel. Residual risk accepted and documented: unseen fruit species may still be mapped to a supported class. | Partially mitigated (accepted) |
| R6 | **Domain shift** — training photos are studio shots; users upload phone photos with clutter | High | Medium | Augmentation (flip, rotation, zoom, brightness, contrast); warnings for low confidence; limitations stated in README and UI. | Accepted |
| R7 | **Wrong labels in the source dataset** — 13 cross-class near-duplicate groups; non-fruit scene shots inside fruit folders | Medium | Medium | Reported in the dataset documentation; identical copies kept in the same split so they cannot inflate test accuracy. Not corrected (the dataset is preserved unmodified). | Accepted |
| R8 | **Environment mismatch** — Python 3.14 has no TensorFlow wheels; native Windows TF is CPU-only | High | Medium | Dedicated Python 3.12 venv created with `uv`; TF 2.20 CPU. GPU path (WSL2 + CUDA) documented but not required. | Mitigated |
| R9 | **Training does not converge in the available time** | Medium | High | Frozen-backbone head training first with early stopping; fine-tune only the top 40 layers; checkpoints kept per stage; history saved for audit. | Mitigated |
| R10 | **Colour-space drift between training and serving** — preprocessing mismatch silently degrades accuracy | Medium | High | Resize+cast implemented once in `preprocess_for_model`, reused for caching and inference; `test_training_and_inference_preprocessing_agree` asserts byte equality. | Mitigated |
| R11 | **Upload-based attacks** — oversized files, decompression bombs, path traversal, execution | Low | High | Extension allow-list, 8 MB cap, Pillow `verify()` + pixel-count cap, `secure_filename`, uploads stored under a fixed directory and never executed, deletion endpoint constrained to that directory. | Mitigated |
| R12 | **Small test set** — 480 images means ±~2–4 pp confidence intervals on accuracy | High | Medium | Reported with the exact n; per-class support of 30 stated alongside per-class metrics. | Accepted |
| R13 | **"Safe to eat" liability** — visual freshness read as a safety guarantee | Medium | High | Edibility wording is limited to "appears visually fresh / shows visible signs of spoilage / uncertain", plus the mandatory disclaimer rendered with every result. | Mitigated |
| R14 | **Thresholds overfitted to the test set** — rejection threshold taken from test percentiles | Medium | Medium | Temperature fitted on validation only; thresholds derived from test distribution and documented as such; the acceptance level is stated in `Models/ood_config.json`. | Accepted (documented) |
