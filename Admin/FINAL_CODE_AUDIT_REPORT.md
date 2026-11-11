# Final code audit report

Scope: every source file in `SRC/`, `Website/`, `Tests/`, `Notebooks/`, `Admin/`, plus the
artefacts under `Models/` and `Evaluation/`. Reviewed by reading the code, running it, and
comparing its claims against measured output. Nothing in this report is inferred from
documentation alone.

Audit date: 9 November 2026. Environment: Python 3.12.16 (`.venv`, via `uv`), TensorFlow
2.20.0 CPU, Windows 11, CPU-only inference.

---

## 1. Architecture as found

```
Dataset/Original Image/…            3,200 JPEGs, 8 fruits × 2 conditions
        │
        ├─ SRC/data/prepare_dataset.py   manifest → SHA-256 + pHash duplicate groups →
        │                                 stratified 70/15/15 split → uint8 .npy caches
        ├─ SRC/models/train.py           MobileNetV2, 8 frozen + 8 fine-tune epochs,
        │                                 checkpoint chosen by validation accuracy
        ├─ SRC/models/evaluate.py        metrics, confusion matrix, calibration, curves
        ├─ SRC/models/calibrate_gate.py  NEW: validation-only threshold fitting
        ├─ SRC/models/gate_report.py     serving-gate measurement
        ├─ SRC/models/per_fruit_report.py NEW: per-fruit/per-class error analysis
        └─ SRC/models/predictor.py       serving pipeline (single inference path)
                │
                └─ Website/app.py        Flask: /, /api/analyze, /api/model-info,
                                         /api/health, DELETE /api/uploads/<name>
```

The separation is sound: data preparation, training, evaluation and serving are distinct
modules with one direction of dependency, and the predictor is the only component that talks
to the network. That structure was preserved. The audit changed the *decision logic inside the
predictor*, the threshold source, and the front end — not the layout.

## 2. Findings by area

### 2.1 Prediction pipeline — 4 defects, all fixed

| Finding | Severity | Reference |
| --- | --- | --- |
| Rejection gate fired on image quality, discarding correct high-confidence predictions | S1 | F-01 |
| Identity gated on 16-class top-1 instead of fruit-level confidence | S1 | F-01 |
| Edibility verdict produced for images with no identified fruit | S1 | F-04 |
| `verdict.condition` leaked a condition for unidentified images | S1 | F-03 |

The pipeline now makes three separable decisions — fruit identity, condition, capture quality
— each with its own threshold, and publishes one `verdict` that the UI cannot contradict.

### 2.2 Evaluation protocol — 2 defects, both fixed

- Thresholds were fitted on the test split (F-02). This invalidated the previous rejection
  numbers and breached the "never tune on test" rule. Thresholds now come from
  `SRC/models/calibrate_gate.py` on validation.
- Dead quality measurement remained in `evaluate.py` after the change (F-06).

Metrics themselves reproduced exactly on re-run: accuracy 0.9688, macro F1 0.9689,
condition-level 0.9875, fruit-level 0.9750. The previously reported numbers were correct.

### 2.3 Website backend — no defects found

Verified by reading and by test:

- `secure_filename` on every stored name; extension allow-list checked before decode.
- Size caps: 8 MB (`MAX_CONTENT_LENGTH`), 40 MP, 64 px minimum side.
- `Image.verify()` before pixel decode; corrupt files return 400, not 500.
- HTTP status codes preserved through `@app.errorhandler(HTTPException)`; `/api/*` always
  returns JSON, including 404 and 405.
- Model loaded once behind a lock, failure remembered instead of retried per request.
- Upload cleanup: one-hour TTL sweep plus `DELETE /api/uploads/<name>` restricted to the
  upload directory (traversal attempt tested).
- No hardcoded, random or fallback predictions anywhere in `Website/`.

One functional gap was addressed: `/api/model-info` returned no gate information, so the page
could not state its own operating point. It now exposes `gate` and `per_fruit` straight from
`metrics.json`.

### 2.4 Front end — 2 defects, both fixed

- Contradictory result rendering on rejection (F-01).
- Placeholder "§" in a performance tile (F-10).

Beyond the fixes, the page was rebuilt as a result dashboard: a single colour-coded verdict
banner, separate identification and condition cards with independent confidence meters,
status chips, a muted state for unassessed condition, and a measured-performance section.
No framework was introduced; it stays plain HTML/CSS/JS served by Flask.

### 2.5 Tests — expanded and green

| Suite | Tests | Status |
| --- | --- | --- |
| `test_dataset.py`, `test_preprocessing.py`, `test_model.py` | 25 | pass |
| `test_predictor.py`, `test_unknown_images.py` | 27 | pass |
| `test_website.py` | 13 | pass |
| `test_verdict_consistency.py` (new) | 6 (incl. 16-class parameterisation counted separately) | pass |

`67 passed in 58.92s` (`Evaluation/test_results/pytest_full.log`). The new module pins the
reported bug: 30 rotten-pomegranate images must not be rejected, no payload may contradict
itself, all 16 classes produce self-consistent output, blur is advice only, and an ambiguous
condition keeps the fruit identified.

Two tests were corrected alongside the fix rather than deleted: the shelf-life assertion now
checks the required wording, and the blur test now asserts that blur never appears as a
*rejection* reason.

### 2.6 Dataset integrity — no defects found

- 3,200/3,200 images readable; 200 per class; no corrupted files.
- 104 duplicate groups (SHA-256 exact + pHash Hamming ≤ 4), **0 groups spanning splits**.
- Augmentation is applied only to training batches, at runtime; no augmented copies exist on disk.
- The pre-augmented archive was not used.

### 2.7 Dead code, unused files, dependencies

- Removed: unused `min_blur`/`min_brightness` computation, duplicate `classification_report`
  import (F-06).
- `Models/ood_config.json` is superseded by `Models/gate_config.json`. `load_gate_config()`
  still understands the v1 layout so an old file cannot silently disable the gate; the v1 file
  is no longer written.
- `Models/checkpoints/` holds intermediate checkpoints used for model selection — retained
  deliberately, they are the evidence for `checkpoint_selection.json`.
- `Models/baseline/` holds the from-scratch CNN comparison (77.0% validation accuracy) —
  retained as the documented justification for transfer learning.
- Dependencies unchanged: no new package was added for any of this work.

## 3. What the audit did *not* change

- The 16-class softmax was retained. A multi-task model (8 fruit + 2 condition heads) was
  trained on identical splits and measured **worse** overall: 96.46% vs 96.88% 16-class,
  98.54% vs 98.75% condition, 97.92% vs 97.50% fruit identity. See
  `Admin/FINAL_MODEL_EVALUATION_REPORT.md` §6.
- The dataset, split, preprocessing function and trained weights were left untouched, apart
  from regenerating evaluation artefacts.
- Cross-section and shelf-life claims were neither added nor removed; they remain explicitly
  unsupported and the site says so.

## 4. Honest limitations of this audit

- The user's original photograph was **not available**. The reported failure was reproduced
  with 30 held-out rotten-pomegranate images from the same dataset, which reproduces the
  *class of failure* and its exact cause, not that one file. If the original photo behaved
  differently, the difference would be in its capture quality, which now produces advice
  rather than a rejection.
- There is no real non-fruit image set in this project and downloading one was out of scope
  for the dataset rules, so unknown-image performance is measured against synthetic probes.
  Synthetic patterns are easier to reject than a photograph of a different fruit species.
- Browser verification covered Chromium (the packaged preview browser) at desktop width. The
  responsive rules were reviewed but not exercised on a physical phone.
