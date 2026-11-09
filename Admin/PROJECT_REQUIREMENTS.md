# Project Requirements — AI-Based Fruit Detection and Freshness Analysis

This document records what the system must do, what the dataset actually supports, and how
each requirement is verified. It is the contract the implementation is audited against.

## 1. Functional requirements

| ID | Requirement | Status | Evidence |
|----|-------------|--------|----------|
| F1 | Identify the fruit from an uploaded photograph (8 supported fruits) | Implemented | `SRC/models/predictor.py`, evaluation report |
| F2 | Report fresh/good vs rotten/bad condition | Implemented | Aggregated from 16-class softmax |
| F3 | Show fresh and rotten probability percentages | Implemented | Web UI result cards |
| F4 | Show fruit-identification confidence separately | Implemented | Model card section |
| F5 | Give a visual edibility assessment with a safety disclaimer | Implemented | `Website/templates/index.html` + disclaimer block |
| F6 | Estimate remaining shelf life **only with valid time-based data** | Not possible with this dataset | Documented in `DATASET_DOCUMENTATION.md` §6 |
| F7 | Support whole-fruit and cross-section photographs | Whole fruit supported; cross-section **not** validated | `DATASET_DOCUMENTATION.md` §5 |
| F8 | Reject unknown fruits / invalid images instead of forcing a class | Implemented | OOD gate in `SRC/models/predictor.py`, tested |
| F9 | Upload via drag-and-drop with preview and loading state | Implemented | `Website/static/js/app.js` |
| F10 | Validate file type/size, secure filenames, never execute uploads | Implemented | `Website/app.py` |
| F11 | Clear/reset the analysis | Implemented | Reset button + `/api/reset` free client-side path |

## 2. Non-functional requirements

- **Reproducibility**: fixed seed, saved class mapping, saved split manifest, documented versions.
- **Honesty**: no fabricated metrics; percentages are described as model-estimated probabilities,
  not physical rot percentages.
- **Leakage control**: images derived from the same original never straddle splits; duplicates are
  detected by SHA-256 and perceptual hash and grouped before splitting.
- **Modularity**: data, model, inference and web layers are separate Python packages.
- **Portability**: runs on Windows with a CPU-only TensorFlow install; GPU is optional.

## 3. Honest limitations (must be visible in the UI and docs)

1. **Shelf life is unavailable.** The dataset has binary condition labels only, with no capture
   timeline, storage temperature or days-to-spoilage figures. Displaying a number of days would be
   fabrication, so the UI states that insufficient data exists.
2. **Cross-section analysis is not validated.** Only a small, unlabelled fraction of images show cut
   fruit. The model is validated for whole-fruit photographs.
3. **Model-estimated probabilities are not calibrated physical measurements.** Calibration is
   measured and reported; the UI says "estimated".
4. **Small dataset.** 3,200 images across 16 classes, studio-style photographs. Real-world phone
   photos with clutter, unusual lighting or multiple objects will be less reliable.
5. **Visual assessment ≠ food safety.** The system can never certify a fruit as safe to eat.

## 4. Acceptance criteria

The project is complete when:

- [x] Dataset fully inspected, with a written report and per-image inventory.
- [x] Reproducible preprocessing, leaktight stratified split and dataset manifest.
- [x] Trained model saved with class mapping; training curves saved.
- [x] Evaluation on an untouched test set: accuracy, precision, recall, F1, per-class report,
      confusion matrix, inference time.
- [x] Flask website serving real predictions from the saved model.
- [x] Automated tests covering data, preprocessing, model, inference and web endpoints.
- [x] Documentation: requirements, roadmap, dataset documentation, development log, audit, risks.
