# Project Roadmap

Seven phases, executed in order. Each phase lists the artefacts it produces and the gate that must
pass before the next phase starts.

| Phase | Goal | Deliverables | Gate |
|-------|------|--------------|------|
| 1 | Inspect project + dataset | `Admin/DATASET_DOCUMENTATION.md`, `Dataset/interim/inspection_summary.json`, `Dataset/interim/image_inventory.csv`, review grids | Every image accounted for; cross-section and shelf-life availability decided from evidence |
| 2 | Data preparation | `Dataset/dataset_manifest.csv`, `Dataset/processed/*.npy`, augmentation pipeline, `SRC/data/prepare_dataset.py` | Split counts match the documented 70/15/15; no duplicate group straddles splits |
| 3 | Model development | `Models/final_model/*`, `Models/class_names.json`, baseline CNN comparison, training curves | Validation accuracy plateau reached; best checkpoint restored |
| 4 | Prediction pipeline | `SRC/models/predictor.py` (fruit + condition + OOD gate), `Models/calibration.json` | Inference preprocessing identical to training; OOD behaviour measured on non-fruit images |
| 5 | Website | `Website/app.py`, templates, CSS/JS | Real predictions rendered; upload validation; clear/reset |
| 6 | Testing | `Tests/*`, `Evaluation/*` | `pytest` green; evaluation plots and reports saved |
| 7 | Final audit + docs | `README.md`, `Admin/CODE_AUDIT_REPORT.md`, `Admin/DEVELOPMENT_LOG.md`, `Admin/RISK_REGISTER.md` | Every claim traceable to an artefact |

## Key decisions taken up front

- **Environment**: native Windows TensorFlow is CPU-only from version 2.11 onward. Python 3.14 has no
  TensorFlow wheels either, so the project runs in a dedicated Python 3.12 virtual environment
  (`.venv`) with `tensorflow` 2.20 CPU. WSL2 with CUDA remains an option for GPU training and is
  documented, but is not required.
- **Architecture**: single 16-class MobileNetV2 transfer-learning model. Fruit identity and condition
  fall out of the class names, and condition probabilities are summed across the eight classes of
  each condition. A multi-task variant was considered unnecessary for this dataset size.
- **Leakage**: duplicate groups (identical file hash, or perceptual-hash near duplicates confirmed by
  review) are assigned to a single split as a group.
- **Shelf life**: not implemented — no time-based labels exist. The API returns an explicit
  `unavailable` status with the reason, and the UI renders that message.
