# Final model evaluation report

All numbers below were produced by this repository on 9 November 2026, on the untouched test
split, after the audit fixes. No figure is carried over from documentation.

- Model: `Models/final_model/fruit_model.keras` — MobileNetV2 (ImageNet) + GAP → 256 → 16-way
  softmax, 8 frozen epochs then 8 fine-tune epochs.
- Test split: 480 images, 30 per class, never used for training, threshold fitting or model
  selection.
- Artefacts: `Evaluation/metrics/metrics.json`, `metrics/classification_report.{txt,csv}`,
  `confusion_matrices/confusion_matrix.png`, `test_results/{ood_probe,gate_probe,
  gate_calibration,predictions,misclassified}.csv/json`,
  `metrics/{per_fruit_report.json,per_fruit_report.md,architecture_comparison.md}`.

## 1. Headline metrics

| Metric | Value |
| --- | --- |
| 16-class accuracy | **96.88%** (465 / 480) |
| Macro precision / recall / F1 | 97.16% / 96.88% / **96.89%** |
| Weighted F1 | 96.89% |
| Fruit identity accuracy (8-way) | 97.50% (468 / 480) |
| Fresh-vs-rotten accuracy (2-way) | **98.75%** (474 / 480) |
| Cross-entropy before / after calibration | 0.0848 / 0.0851 |

The fresh-vs-rotten figure is the one the result card depends on; the 16-class figure is the
stricter view that also requires the right fruit.

## 2. Per-class results

Full table: `Evaluation/metrics/classification_report.csv`.

| Class | Precision | Recall | F1 |
| --- | --- | --- | --- |
| FreshApple | 1.000 | 0.867 | 0.929 |
| RottenApple | 0.966 | 0.933 | 0.949 |
| FreshBanana | 1.000 | 1.000 | 1.000 |
| RottenBanana | 1.000 | 1.000 | 1.000 |
| FreshOrange | 1.000 | 0.933 | 0.966 |
| RottenOrange | 0.968 | 1.000 | 0.984 |
| FreshGrape | 1.000 | 1.000 | 1.000 |
| RottenGrape | 1.000 | 0.933 | 0.966 |
| FreshGuava | 1.000 | 1.000 | 1.000 |
| RottenGuava | 0.968 | 0.967 | 0.967 |
| FreshJujube | 1.000 | 1.000 | 1.000 |
| RottenJujube | 0.933 | 1.000 | 0.966 |
| **FreshPomegranate** | 0.833 | **1.000** | 0.909 |
| **RottenPomegranate** | **1.000** | **0.933** | 0.966 |
| FreshStrawberry | 1.000 | 1.000 | 1.000 |
| RottenStrawberry | 1.000 | 1.000 | 1.000 |

`FreshPomegranate` has the lowest precision because four `FreshApple` photographs were labelled
as fresh pomegranate. It classifies every one of its own 30 images correctly. **The reported
rotten-pomegranate failure was therefore not a model weakness**: 28/30 correct, precision 1.000
on the class. It was the serving gate (see §5).

## 3. Per-fruit aggregation

From `Evaluation/metrics/per_fruit_report.md`. Fruit-level precision counts a false positive
whenever the model names that fruit for a photograph of a different one.

| Fruit | Images | Fruit correct | Condition acc. | Fresh acc. | Rotten acc. | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Apple | 60 | 54 | 1.000 | 1.000 | 1.000 | 0.982 | 0.900 | 0.939 |
| Banana | 60 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Grape | 60 | 59 | 0.967 | 1.000 | 0.933 | 1.000 | 0.983 | 0.992 |
| Guava | 60 | 59 | 0.983 | 1.000 | 0.967 | 0.967 | 0.983 | 0.975 |
| Jujube | 60 | 59 | 1.000 | 1.000 | 1.000 | 0.952 | 0.983 | 0.967 |
| Orange | 60 | 58 | 0.967 | 0.933 | 1.000 | 0.983 | 0.967 | 0.975 |
| Pomegranate | 60 | 59 | 0.983 | 1.000 | 0.967 | 0.922 | 0.983 | 0.952 |
| Strawberry | 60 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Apple is the weakest fruit (0.900 recall): four fresh apples are read as fresh pomegranate and
two rotten apples as rotten guava. Pomegranate has the lowest fruit precision (0.922) for the
same reason — it absorbs those apple images.

**Where the condition is wrong, it is wrong across fruits.** Every fruit scores ≥ 0.967
condition accuracy; the 15 class-level errors are predominantly *fruit* confusions, not
fresh/rotten confusions. That is the behaviour the two-part gate is designed around.

Misclassified images are listed in `Evaluation/test_results/misclassified.csv` with their
confidence, so they can be inspected individually. The dataset provides no evidence of
label errors among them; they are visually plausible confusions (rotten apple and rotten guava
share dark wrinkled skin; fresh apple and fresh pomegranate share red, glossy surfaces).

## 4. Calibration and timing

- Temperature scaling, T = **1.077**, fitted by minimising NLL on the validation split.
- ECE 1.45% → **1.65%** after calibration; NLL 0.0848 → 0.0851. Temperature scaling does not
  improve this model's calibration — it is already well calibrated and the single-parameter fit
  slightly over-smooths. It is applied anyway because the gate thresholds were fitted on
  calibrated probabilities, and applying it consistently avoids a train/serve mismatch.
- Inference: **83.5 ms** mean single-image, 65.5 ms median, 88.6 ms p95 on CPU;
  92.1 images/s at batch 32.

## 5. Serving gate (identity vs condition)

Fitted on validation by `SRC/models/calibrate_gate.py`, stored in `Models/gate_config.json`:

| Parameter | Value | Rule |
| --- | --- | --- |
| Fruit confidence floor | 0.750 | largest synthetic-OOD rejection subject to ≤2% validation false rejection |
| Fruit margin floor | 0.300 | 1st percentile of validation fruit margin |
| Condition confidence floor | 0.673 | 1st percentile of validation condition confidence |
| Temperature | 1.077 | NLL minimisation on validation |

Measured behaviour:

| Group | Identity rejected | Condition uncertain | Mean fruit confidence |
| --- | --- | --- | --- |
| Test split, n = 120 sample | **4.2%** | 0.8% | 0.98 |
| Test split, all 480 | **4.58%** | 1.25% | — |
| Validation, all 480 | 1.87% (the fitted target) | — | — |
| Synthetic non-fruit, n = 40 | **62.5%** | 0.0% | 0.72 |
| Synthetic non-fruit, n = 120 | **71.7%** | — | — |
| Flat colour, n = 5 | 40.0% | 0.0% | 0.78 |
| Blurred test images, n = 50 | 40.0% | — | — |
| Very dark test images, n = 50 | 80.0% | — | — |
| Down-sampled test images, n = 50 | 18.0% | — | — |

**Before → after** (same 480-image test split):

| Measure | First release | Now |
| --- | --- | --- |
| Genuine test images rejected | 11.5% (55/480) | **4.58% (22/480)** |
| … of those, rejected purely for blur | 47 | **0** |
| Synthetic non-fruit rejected | 40% | **71.7%** (n = 120) |
| Rotten pomegranates rejected | 14/30 | **2/30** |
| Contradictory result payloads | frequent | **0/480** |

Both error rates improved at once, which the old gate could not do because its blur threshold
rejected genuine fruit and its 16-class confidence threshold was too coarse to reject patterns.

## 6. Architecture comparison: one 16-class head vs two heads

Trained for this audit with the identical backbone, augmentation, split and 8 + 8 epoch
schedule, selected on validation loss, evaluated once on the test split
(`SRC/models/train_multitask.py`, 537 s CPU):

| Metric | 16-class single head | Multi-task (8 + 2) | Difference |
| --- | --- | --- | --- |
| 16-class accuracy | **0.9688** | 0.9646 | −0.0042 |
| Fresh vs rotten accuracy | **0.9875** | 0.9854 | −0.0021 |
| Fruit identity accuracy | 0.9750 | **0.9792** | +0.0042 |

The multi-task model is marginally better at *which fruit* and worse at *everything the site
reports as its main answer*. The 16-class model also has one interpretive advantage: both
numbers come from the same distribution, so the fresh score and the fruit score can never
describe different objects.

**Decision: keep the 16-class single head.** The difference is within noise (≈2 images out of
480), it does not improve the headline metric, and replacing a validated model to gain nothing
measurable would be churn. The comparison is retained as evidence that the choice was tested
rather than assumed.

## 7. Limitations

- **Cross-section support is not validated.** The dataset has no cut/whole label. A visual
  audit of the class grids found cut fruit only as a small minority in apple, pomegranate and
  orange sets. Whole-fruit and cross-section performance cannot be separated on a labelled
  basis, so no claim is made and the UI states it in the hero and in the About section.
- **Shelf life is not predictable** from this dataset. There are no storage ages, capture
  timelines or days-to-spoilage labels; EXIF dates cluster in Mar–Apr 2022 with no storage
  meaning. The site displays "Remaining shelf life cannot currently be estimated from the
  available dataset."
- **Unknown fruit species are the weak point.** The softmax always produces an answer; only
  confidence separates "a fruit I know, seen badly" from "a fruit I have never seen".
  Synthetic probes are the only negatives available here, and they are easier than reality.
- **Apple is the weakest fruit (0.900 recall)**, driven by fresh-apple → fresh-pomegranate
  confusions.
- **CPU-only measurement.** Timings describe a laptop CPU, not a deployment target.
