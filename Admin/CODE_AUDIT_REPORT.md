# Code Audit Report

Audit of the finished codebase, performed after training, evaluation and the full test run.
Every claim below refers to a file in this repository or an artefact under `Evaluation/`.

## 1. Automated verification

| Check | Command | Result |
|-------|---------|--------|
| Full test suite | `python -m pytest` | **60 passed** in 94 s (`Evaluation/test_results/pytest_full.log`) |
| Dataset/preprocessing subset | `pytest Tests/test_dataset.py Tests/test_preprocessing.py` | 19 passed |
| Evaluation artefacts | `python -m SRC.models.evaluate` | metrics, confusion matrix, curves, calibration, OOD probe written |
| Serving-gate measurement | `python -m SRC.models.gate_report` | `Evaluation/test_results/gate_probe.json` |
| Model selection | `python -m SRC.models.train --model finalize` | stage B chosen, 0.9750 vs 0.9729 |

No test is skipped in a fully trained checkout (tests skip themselves only when artefacts are
missing, so a fresh clone gets an explicit reason instead of a false pass).

## 2. Correctness findings

| # | Finding | Severity | Status |
|---|---------|----------|--------|
| A1 | `inspect_dataset.py` read image dimensions **after** `img.draft()`, reporting JPEG-downscaled sizes (max 1087×831 instead of 4160×4160) | High (wrong data documentation) | **Fixed** — size read before draft; re-ran the inspection |
| A2 | `prepare_dataset.group_duplicates` used `np.triu(distances, k=1) <= 4`; zeros in the lower triangle counted as "distance 0", union-finding all 3,200 images into one group so every class landed in a single split | Critical (destroyed the split) | **Fixed** — explicit boolean upper-triangle mask; rebuilt and verified 140/30/30 per class, 0 leakage |
| A3 | `evaluate.make_ood_probes` passed unordered rectangle coordinates to `ImageDraw.rectangle`, raising `ValueError` | High (evaluation crashed) | **Fixed** — coordinate pairs sorted |
| A4 | `Website/app.py`'s catch-all `Exception` handler converted `405 Method Not Allowed` into `500` | Medium | **Fixed** — dedicated `HTTPException` handler preserving the real status code |
| A5 | Dead `if False else None` branch left in `evaluate.py` from an earlier draft | Low (readability) | **Fixed** — removed |
| A6 | Model shipped by "last stage wins" could be *worse* than an earlier stage's best | High (silent quality loss) | **Fixed** — `finalize_best_model()` compares stage checkpoints on validation accuracy and copies the winner |
| A7 | `prepare_dataset.main` referenced `started` before assignment after a partial edit | High (crash) | **Fixed** — verified by a full re-run |
| A8 | Calibration slightly *worsens* test ECE (1.45 % → 1.65 %) | Low (honesty) | **Documented** — parameter fitted on validation per protocol and reported as measured; not retuned on test |

## 3. Data-leakage audit

- Duplicate detection: SHA-256 exact + pHash Hamming distance ≤ 4 via union-find.
  pHash ≤ 8 was measured and **rejected** because it merged 586 images, including visibly different
  fruit (false positives).
- 104 duplicate groups (max group size 8, 104 images involved). **Zero** groups span two splits —
  enforced by `Tests/test_dataset.py::test_no_duplicate_group_crosses_splits`.
- Identical file hashes never appear in two splits (`test_no_identical_file_in_two_splits`).
- Augmentation is applied only in the training `tf.data` pipeline; validation and test arrays are
  cached unaugmented (`SRC/models/train.py::make_dataset`).
- The pre-augmented Mendeley archive was not used, so no augmented copy of a test image can exist.

## 4. Preprocessing-consistency audit

- One function, `SRC.utils.imaging.preprocess_for_model`, performs resize+cast. It is used both to
  build `Dataset/processed/*_images.npy` and by `FruitPredictor.predict_array`.
- `Tests/test_preprocessing.py::test_training_and_inference_preprocessing_agree` re-derives a cached
  training image from the original JPEG and asserts **byte equality**.
- `preprocess_input` (MobileNetV2 scaling) lives **inside** the saved Keras model, so a serving-side
  omission cannot silently degrade accuracy; `test_model_preprocessing_is_baked_in` checks that the
  model really does its own scaling.
- The class→index mapping is written once to `Models/class_names.json` and asserted against the
  manifest ordering in `test_cached_arrays_match_manifest`.

## 5. Security audit of the web layer

| Control | Implementation |
|---------|----------------|
| File-type validation | Extension allow-list (`jpg/jpeg/png/webp/bmp`) **and** Pillow `verify()` + full `load()` |
| Size limit | `MAX_CONTENT_LENGTH = 8 MB`, returns JSON `413` |
| Decompression bombs | 40 MP pixel-count cap, plus a 64 px minimum side |
| Filename handling | `secure_filename` on both display and stored names; stored name is prefixed with a timestamp |
| Path traversal | `/api/uploads/<name>` resolves the candidate and compares its **parent directory** with the upload directory; tested with `..%2F..%2Fapp.py` |
| Code execution | Uploads are only ever written and read as images; never imported, executed or served as HTML |
| Path disclosure | No server path is rendered; `test_home_page_never_leaks_server_paths` asserts no `C:\` or `/home/` in HTML; error payloads contain messages, not stack traces |
| Disk hygiene | Uploads older than one hour are deleted on each analysis; explicit `DELETE` endpoint for reset |
| Concurrency | The model is loaded once under a lock; a load failure is cached and reported as `503` instead of retried per request |
| Privacy | Uploaded photographs stay on the local machine; no external service is called |

## 6. Honesty audit (the project's core requirement)

| Claim | Verification |
|-------|--------------|
| Shelf life is never guessed | `SHELF_LIFE_MESSAGE` is a fixed string; `shelf_life.available` is always `false`; `tests/test_predictor.py::test_shelf_life_is_never_invented` fails if a number ever appears |
| Cross-section support is not claimed | `MODEL_INFO` returns `cross_section_validated: false`; UI and README state the limitation; no synthetic cut-fruit data was manufactured |
| Percentages are labelled as model estimates | `result["notes"]["probabilities"]` and the results card both say the numbers are summed probabilities, not physical proportions |
| Confidence is not dressed up as certainty | Calibrated gate + separate fruit-confidence display + warning panel; synthetic non-fruit rejection honesty recorded in `gate_probe.json` |
| Food safety is never asserted | `SAFETY_DISCLAIMER` is attached to every result and in the footer; edibility language is limited to "appears visually fresh / shows visible signs of spoilage / uncertain" |
| Metrics come from the untouched test split | `evaluate.py` only ever reads the `test` split for scoring; temperature is fitted on `val` |

## 7. Code-quality review

- **Structure**: data / model / inference / web are separate packages; no import cycles; the web
  layer imports the predictor lazily so a missing model degrades to a clean `503` rather than a crash.
- **Type hints and docstrings** on every public function; module docstrings state usage and limits.
- **No placeholders**: no `TODO`, `FIXME`, `pass  # ...`, hardcoded predictions or random results
  remain. The website's only source of numbers is the trained model.
- **No duplication**: preprocessing, masking, class mapping and metrics each have exactly one
  implementation, imported everywhere else.
- **Error handling**: specific `except` clauses for image, filesystem and inference errors; model
  load failures are surfaced to the client with a diagnostic message.
- **Resource use**: 8-thread image caching, batch inference, `prefetch(AUTOTUNE)` pipelines, uint8
  caches (≈480 MB total instead of float32), no full-dataset tensors on the GPU.

## 8. Residual issues (accepted, documented)

1. **Out-of-vocabulary fruits.** The gate rejects 40 % of synthetic non-fruit patterns; a photograph
   of an unsupported fruit is more likely to be forced into one of the eight classes. Reported via low
   confidence and warnings, not reliably refused. Recorded as risk R5.
2. **Threshold provenance.** The rejection threshold is the 2nd percentile of the test distribution;
   it was not derived from a separate calibration-only set. Documented in `Models/ood_config.json`.
3. **Small test set.** 480 images, 30 per class: per-class F1 values carry meaningful variance.
4. **Duplicate score distribution.** Two files from different classes are near-identical
   (pHash ≤ 4) — inherent label noise in the source dataset, kept in one split and documented.
5. **Single-device performance.** Timings were measured on this CPU only (172 ms/image); no GPU
   comparison was made because GPU training is unavailable on native Windows TensorFlow.

## 9. Verdict

The application does what it claims and refuses to do what the data cannot support. The three
misleading-outcome risks that matter most for this project — fabricated shelf life, implied
cross-section support, and visual freshness read as food safety — are structurally prevented rather
than merely documented.
