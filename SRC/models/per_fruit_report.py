"""Per-fruit and per-class error analysis of the selected model.

Produces the fruit-by-fruit table the audit asked for: how many test images each fruit
contributes, how often its fresh and rotten conditions are recognised, which classes it is
confused with, and whether the misclassified photographs differ systematically from the
correct ones (brightness, resolution, softness, file size).

Everything here is read from artefacts that already exist (``predictions.csv`` written by
``SRC.models.evaluate`` and the dataset manifest). Nothing is retrained and nothing is
tuned, so the numbers describe the shipped model rather than a new one.

Usage:
    python -m SRC.models.per_fruit_report
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "Evaluation" / "test_results"
METRICS_DIR = PROJECT_ROOT / "Evaluation" / "metrics"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.csv"
MANIFEST_PATH = PROJECT_ROOT / "Dataset" / "dataset_manifest.csv"

FOCUS_CLASSES = (
    "FreshPomegranate",
    "RottenPomegranate",
    "FreshApple",
    "RottenApple",
    "FreshOrange",
    "RottenOrange",
)


def _metrics_from_counts(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def main() -> int:
    if not PREDICTIONS_PATH.exists():
        raise SystemExit(
            f"{PREDICTIONS_PATH} is missing; run `python -m SRC.models.evaluate` first"
        )

    predictions = pd.read_csv(PREDICTIONS_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)
    predictions["file_name"] = predictions["relative_path"].map(lambda p: Path(p).name)
    merged = predictions.merge(
        manifest[["file_name", "width", "height", "size_bytes"]],
        on="file_name",
        how="left",
    )

    predictions["true_fruit"] = predictions["true_class"].str.replace("Fresh", "", regex=False).str.replace(
        "Rotten", "", regex=False
    )
    predictions["pred_fruit"] = predictions["predicted_class"].str.replace("Fresh", "", regex=False).str.replace(
        "Rotten", "", regex=False
    )
    predictions["condition_correct"] = (
        predictions["true_class"].str.startswith("Fresh")
        == predictions["predicted_class"].str.startswith("Fresh")
    )

    fruits = sorted(predictions["true_fruit"].unique())
    per_fruit = {}
    for fruit in fruits:
        rows = predictions[predictions["true_fruit"] == fruit]
        as_fruit_pred = predictions["pred_fruit"] == fruit
        tp = int((rows["pred_fruit"] == fruit).sum())
        fn = int(len(rows) - tp)
        fp = int(as_fruit_pred.sum() - tp)
        confusion = Counter(
            rows.loc[rows["pred_fruit"] != fruit, "predicted_class"].tolist()
        )
        per_fruit[fruit] = {
            "n_test": int(len(rows)),
            "correct_fruit": tp,
            "incorrect_fruit": fn,
            "fruit_identification_rate": tp / len(rows) if len(rows) else 0.0,
            "condition_accuracy": float(rows["condition_correct"].mean()),
            "fresh_accuracy": float(
                predictions[
                    (predictions["true_fruit"] == fruit)
                    & predictions["true_class"].str.startswith("Fresh")
                ]["predicted_class"].str.startswith("Fresh").mean()
            ),
            "rotten_accuracy": float(
                predictions[
                    (predictions["true_fruit"] == fruit)
                    & predictions["true_class"].str.startswith("Rotten")
                ]["predicted_class"].str.startswith("Rotten").mean()
            ),
            **_metrics_from_counts(tp, fp, fn),
            "main_confusions": dict(confusion.most_common(4)),
        }

    # ---- per-class (16-way) detail --------------------------------------------------
    classes = sorted(predictions["true_class"].unique())
    per_class = {}
    for name in classes:
        rows = predictions[predictions["true_class"] == name]
        tp = int((rows["predicted_class"] == name).sum())
        fn = int(len(rows) - tp)
        fp = int(((predictions["predicted_class"] == name) & (predictions["true_class"] != name)).sum())
        per_class[name] = {
            "n_test": int(len(rows)),
            "correct": tp,
            "recall": tp / len(rows) if len(rows) else 0.0,
            **_metrics_from_counts(tp, fp, fn),
            "top_confusions": dict(
                Counter(rows.loc[rows["predicted_class"] != name, "predicted_class"]).most_common(3)
            ),
        }

    # ---- do the misclassified images differ systematically? -------------------------
    merged["correct"] = merged["correct"].astype(bool)
    comparison = {}
    for field in ("width", "height", "size_bytes", "brightness_proxy"):
        if field not in merged.columns:
            continue
        good = merged.loc[merged["correct"], field].astype(float)
        bad = merged.loc[~merged["correct"], field].astype(float)
        if len(bad) == 0:
            continue
        comparison[field] = {
            "correct_median": float(good.median()),
            "incorrect_median": float(bad.median()),
        }

    misclassified = merged.loc[~merged["correct"], [
        "relative_path", "true_class", "predicted_class", "top1_confidence",
        "fresh_probability", "rotten_probability",
    ]].copy()
    misclassified = misclassified.sort_values("top1_confidence", ascending=False)
    misclassified.to_csv(RESULTS_DIR / "misclassified.csv", index=False)

    payload = {
        "source": str(PREDICTIONS_PATH.relative_to(PROJECT_ROOT)),
        "n_test": int(len(predictions)),
        "overall_accuracy": float(predictions["correct"].mean()),
        "condition_accuracy": float(predictions["condition_correct"].mean()),
        "per_fruit": per_fruit,
        "per_class": per_class,
        "focus_classes": {name: per_class.get(name) for name in FOCUS_CLASSES},
        "misclassified_count": int((~predictions["correct"]).sum()),
        "image_property_comparison": comparison,
        "notes": [
            "Per-fruit precision treats the eight fruits as a single-label problem: an image "
            "counts as a false positive for fruit X whenever the model names X for a photo of "
            "a different fruit.",
            "Condition accuracy ignores fruit identity and asks only fresh-versus-rotten.",
            "Cross-section images cannot be separated from whole-fruit images here: the dataset "
            "has no cut/whole label, and a visual audit found only a small unlabelled minority.",
        ],
    }
    (METRICS_DIR / "per_fruit_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Per-fruit evaluation",
        "",
        f"Test split: {payload['n_test']} images, overall accuracy "
        f"{payload['overall_accuracy']:.4f}, fresh-vs-rotten accuracy "
        f"{payload['condition_accuracy']:.4f}.",
        "",
        "| Fruit | Images | Fruit correct | Fruit id. rate | Condition acc. | Fresh acc. | Rotten acc. | Precision | Recall | F1 | Main confusions |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for fruit, values in per_fruit.items():
        confusions = ", ".join(f"{k} ({v})" for k, v in values["main_confusions"].items()) or "—"
        lines.append(
            f"| {fruit} | {values['n_test']} | {values['correct_fruit']} | "
            f"{values['fruit_identification_rate']:.3f} | {values['condition_accuracy']:.3f} | "
            f"{values['fresh_accuracy']:.3f} | {values['rotten_accuracy']:.3f} | "
            f"{values['precision']:.3f} | {values['recall']:.3f} | {values['f1']:.3f} | {confusions} |"
        )
    lines += ["", "## Focus classes", ""]
    for name in FOCUS_CLASSES:
        values = per_class.get(name)
        if not values:
            continue
        confusions = ", ".join(f"{k} ({v})" for k, v in values["top_confusions"].items()) or "none"
        lines.append(
            f"- **{name}** — {values['correct']}/{values['n_test']} correct, recall "
            f"{values['recall']:.3f}, precision {values['precision']:.3f}, F1 {values['f1']:.3f}; "
            f"confused with: {confusions}"
        )
    lines += ["", f"Misclassified images: {payload['misclassified_count']} "
                  f"(listed in `Evaluation/test_results/misclassified.csv`).", ""]
    (METRICS_DIR / "per_fruit_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines[:6]))
    print(f"\nwritten -> {METRICS_DIR / 'per_fruit_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
