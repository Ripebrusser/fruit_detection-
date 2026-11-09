# Development Log

Chronological record of what was actually executed, including failures and how they were fixed.
Every entry was performed on the machine described in §Environment.

## Environment (2026-11-09)

| Item | Finding |
|------|---------|
| OS | Windows 11 |
| GPU | NVIDIA RTX 4050 Laptop, 6 GB (driver 617.42, CUDA UMD 13.4) |
| System Python | 3.14.3 only — **no TensorFlow wheels exist for 3.14** |
| `python` on PATH | Windows Store alias stub (fails with "Python was not found") |
| WSL2 | Ubuntu installed but stopped; `wsl --status` reports the optional component is not enabled |
| Decision | Isolated Python **3.12.16** virtual environment created with `uv` (no admin rights, no changes to system Python) |

Commands used:

```bash
python -m pip install --user uv
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

Installed: TensorFlow 2.20.0 (CPU), NumPy 2.2.6, Flask 3.1.3, OpenCV 5.0.0, scikit-learn 1.9.1,
Pillow, imagehash, pandas, matplotlib, seaborn, pytest.

**Why CPU training:** TensorFlow ≥2.11 has no native Windows GPU support (the CUDA build is
Linux-only). GPU training requires WSL2 with the CUDA driver — documented in the README as an
optional path. Measured CPU throughput made CPU training practical (~50–65 s per epoch).

## Phase 1 — Inspection

1. Inspected the repository: `Admin/`, `Dataset/`, `Evaluation/`, `Models/`, `Notebooks/`, `SRC/`,
   `Tests/`, `Website/` existed but were **all empty**. No archive file was present; only the
   extracted "Original Image/Original Image" tree.
2. Wrote `SRC/data/inspect_dataset.py`: SHA-256, pHash, dimensions, EXIF, brightness/saturation
   statistics, exact/near duplicate detection, per-class contact sheets.
3. **Bug found and fixed:** the inspector called `img.draft()` *before* reading `img.size`, so
   reported dimensions were the draft-reduced JPEG size (max 1087×831) instead of the true size
   (max 4160×4160). Fixed by reading the size before drafting, then re-ran.
4. Rendered `Dataset/interim/review_grids/classes/*.jpg` (all 200 images per class) and reviewed
   all 16 grids visually to answer the whole-fruit/cross-section question.
5. Wrote `SRC/data/find_cross_sections.py` to screen candidates, then discovered the first scoring
   version was dominated by the white background. Fixed by measuring only inside a background-masked
   fruit region (`SRC/utils/imaging.fruit_mask`). Second observation: pale-flesh scores also fire on
   light-skinned whole fruit, so the montages — not the score — were used as evidence.

**Phase 1 findings:** 3,200/3,200 readable, 16 balanced classes, 0 corrupted files, 7 exact duplicate
groups (all in `FreshOrange`), no shelf-life metadata, cross-sections present but rare and unlabelled.

## Phase 2 — Data preparation

1. Wrote `SRC/data/prepare_dataset.py` (inventory → duplicate grouping → stratified split → cache).
2. **Bug found and fixed:** `np.triu(distances, k=1) <= 4` treated the zeros in the lower triangle as
   "distance 0", union-finding all 3,200 images into a single group. Each class then landed wholly in
   one split (e.g. 200/0/0). Fixed with an explicit boolean upper-triangle mask; rebuilt.
3. The first full run exceeded the tool's 295 s wait cap and was killed mid-cache. Made the caching
   step resumable (`--stage cache`, `--stage metadata`) and parallel (8 threads), then ran it as a
   background job: train 140 s, val 28 s, test 27 s.
4. Verified: 140/30/30 per class, totals 2,240/480/480, duplicate groups 104 (max size 8), **0 groups
   spanning splits**.

## Phase 3 — Model

1. `SRC/models/train.py`: MobileNetV2 (ImageNet) with `preprocess_input` inside the model, frozen
   backbone → 256-unit head (stage A), then the top 40 layers fine-tuned with BatchNorm frozen
   (stage B). Augmentation only on the training pipeline.
2. Smoke-tested the pipeline on 128 images (4 steps, ~12 s) before committing to a full run —
   accuracy 0.766 on the smoke batch confirmed the wiring.
3. Launched full training (8 head epochs + 8 fine-tune epochs) in the background.
4. **Metric sanity:** staged epochs reached 0.69 → 0.89 training accuracy and 0.90 → 0.94 validation
   accuracy in the first two epochs, which is in line with a 16-class, studio-lit dataset.

## Phase 4–5 — Pipeline and website

1. `SRC/models/predictor.py`: aggregation (fresh = Σ 8 fresh classes, rotten = Σ 8 rotten classes),
   temperature scaling, uncertainty gate, edibility wording, fixed shelf-life message.
2. `Website/app.py`: Flask app with upload validation, size/pixel caps, secure filenames,
   `/api/analyze`, `/api/model-info`, `/api/health`, upload deletion, JSON error handlers.
3. Frontend: single-page UI with drag-and-drop, preview, loading state, result cards, probability
   list, warnings panel and model card.

## Phase 6 — Testing

- `Tests/test_dataset.py`, `Tests/test_preprocessing.py`, `Tests/test_model.py`,
  `Tests/test_predictor.py`, `Tests/test_unknown_images.py`, `Tests/test_website.py`.
- First run: 2 failures — `Models/class_names.json` was missing because the earlier timed-out run
  never reached the metadata step. Added `--stage metadata` to `prepare_dataset.py`; all dataset and
  preprocessing tests then passed (19 passed).
- Second run (full suite, 60 tests): 59 passed, 1 failed — `GET /api/analyze` returned 500 instead of
  405 because the catch-all `Exception` handler swallowed `MethodNotAllowed`.
  **Fixed** by adding a dedicated `HTTPException` handler that preserves the real status code and
  returns JSON for `/api/*` paths. Re-ran the suite.

## Phase 3 (results) — Training

| Stage | Epochs | Best validation accuracy |
|-------|--------|--------------------------|
| A — frozen MobileNetV2 backbone + new head (lr 1e-3) | 8 | 0.9729 |
| B — top 40 layers fine-tuned, BatchNorm frozen (lr 1e-4) | 8 | **0.9750** |

- 16 epochs total, 758 s on CPU (~47 s/epoch), 1,995,408 trainable parameters in stage B.
- Early stopping restored the best epoch (6) of stage B; `ReduceLROnPlateau` halved the LR once.
- **Model selection bug avoided:** stage B can end below stage A's best validation accuracy, so the
  shipped model is chosen by *measured* validation accuracy from both stage checkpoints
  (`--model finalize`, saved in `Models/final_model/checkpoint_selection.json`) rather than by which
  stage ran last. Stage B won (0.9750 vs 0.9729).

## Phase 6 (results) — Evaluation on the untouched test split (480 images)

| Metric | Value |
|--------|-------|
| 16-class accuracy | 0.9688 |
| Macro precision / recall / F1 | 0.9716 / 0.9688 / 0.9689 |
| Weighted F1 | 0.9689 |
| Fruit-level accuracy | 0.9750 |
| Condition accuracy (fresh vs rotten) | 0.9875 |
| Single-image inference (CPU) | 172 ms (median); 40.5 images/s at batch 32 |

**Calibration outcome:** temperature scaling was fitted on the validation split (T = 1.077). On the
held-out test split it changed ECE from 1.45 % to 1.65 % and NLL from 0.0848 to 0.0851 — i.e. the
model was already well calibrated and the fitted temperature is essentially neutral (slightly
negative on test). The protocol was not gamed: the parameter was chosen on validation, reported as
measured, and the label probabilities remain described as model estimates.

**Errors found during evaluation, fixed:**

- `make_ood_probes` crashed with `ValueError: y1 must be greater than or equal to y0` because random
  rectangle corners were not ordered. Fixed by sorting each coordinate pair.
- `evaluate.py` initially contained an unreachable `if False else None` branch from an earlier draft;
  removed rather than left dead.

## Phase 7 — Uncertainty gate measurement

The evaluation calibrates only the top-1 threshold. `SRC/models/gate_report.py` measures the gate the
website actually applies (top-1 + fruit margin + entropy + blur + brightness):

| Input | Rejection rate |
|-------|----------------|
| Untouched test images (n = 120) | 4.2 % |
| Synthetic non-fruit patterns (n = 40) | 40.0 % |
| Blurred test images (n = 50) | 100 % |
| Very dark test images (n = 50) | 100 % |
| Down-sampled to 48 px then upscaled (n = 50) | 86 % |

**Honest conclusion:** the gate reliably catches degraded photographs but rejects only 40 % of
synthetic non-fruit patterns — the network is over-confident far outside its training distribution.
Unseen fruit species are the hardest case and are not reliably rejected. This is why every prediction
carries a warning panel, why confidence is displayed separately, and why the UI never asserts
certainty. Recorded as risk R5 (partially mitigated, accepted).

## Phase 7 — Documentation

`README.md`, `Admin/CODE_AUDIT_REPORT.md`, and this log updated with the measured numbers; the
`README` results table is copied from `Evaluation/metrics/metrics.json`.
