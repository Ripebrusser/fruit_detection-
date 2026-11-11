"""Evaluate the trained model on the untouched test split and calibrate the gate.

Produces, under ``Evaluation/``:

* ``metrics/metrics.json``          — headline numbers (accuracy, P/R/F1, timings)
* ``metrics/classification_report.csv/.txt`` — per-class precision/recall/F1/support
* ``metrics/confusion_matrix.csv``  — raw counts
* ``confusion_matrices/confusion_matrix.png`` — annotated heat map
* ``figures/training_accuracy.png``, ``figures/training_loss.png``
* ``figures/calibration.png``       — reliability diagram before/after temperature scaling
* ``test_results/predictions.csv``  — one row per test image
* ``test_results/ood_probe.json``   — rejection behaviour on synthetic non-fruit images
* ``Models/ood_config.json``        — thresholds + temperature used at serving time

Usage:
    python -m SRC.models.evaluate
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from scipy.optimize import minimize_scalar  # noqa: E402

from SRC.data.prepare_dataset import split_frame  # noqa: E402
from SRC.models.predictor import DEFAULT_MODEL_PATH, CLASS_NAMES_PATH  # noqa: E402
from SRC.utils.imaging import quality_checks  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "Dataset" / "processed"
EVAL_DIR = PROJECT_ROOT / "Evaluation"
METRICS_DIR = EVAL_DIR / "metrics"
CM_DIR = EVAL_DIR / "confusion_matrices"
RESULTS_DIR = EVAL_DIR / "test_results"
FIGURES_DIR = EVAL_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "Models"

SEED = 42


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.load(PROCESSED_DIR / f"{split}_images.npy"),
        np.load(PROCESSED_DIR / f"{split}_labels.npy"),
    )


def log_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.log(np.clip(probabilities, 1e-12, 1.0))


def softmax_from(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def calibrate_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Fit a single temperature on the validation split by minimising NLL."""
    logits = log_probabilities(probabilities)

    def nll(log_temp: float) -> float:
        temp = float(np.exp(log_temp))
        scaled = softmax_from(logits / temp)
        picked = scaled[np.arange(len(labels)), labels]
        return float(-np.log(np.clip(picked, 1e-12, 1.0)).mean())

    result = minimize_scalar(nll, bounds=(np.log(0.25), np.log(8.0)), method="bounded")
    return float(np.exp(result.x))


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if mask.sum() == 0:
            continue
        error += (mask.sum() / len(labels)) * abs(correct[mask].mean() - confidences[mask].mean())
    return float(error)


def plot_reliability(probabilities: np.ndarray, labels: np.ndarray, out_path: Path, title: str) -> None:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)
    bins = np.linspace(0, 1, 11)
    centres, accuracies = [], []
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if mask.sum() > 0:
            centres.append((lower + upper) / 2)
            accuracies.append(correct[mask].mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")
    ax.plot(centres, accuracies, marker="o", label="model")
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curves(history_path: Path) -> dict:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    info = {"accuracy_plot": None, "loss_plot": None}
    if not history_path.exists():
        return info
    history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = range(1, len(history.get("loss", [])) + 1)
    for metric, filename in (("accuracy", "training_accuracy.png"), ("loss", "training_loss.png")):
        fig, ax = plt.subplots(figsize=(6, 4))
        if metric in history:
            ax.plot(epochs, history[metric], label=f"train {metric}")
        if f"val_{metric}" in history:
            ax.plot(epochs, history[f"val_{metric}"], label=f"validation {metric}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.set_title(f"Training and validation {metric} (MobileNetV2)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        target = FIGURES_DIR / filename
        fig.savefig(target, dpi=150)
        plt.close(fig)
        info[f"{metric}_plot"] = str(target.relative_to(PROJECT_ROOT))
    return info


def make_ood_probes(count: int = 40) -> tuple[np.ndarray, list[str]]:
    """Synthetic non-fruit images plus degraded fruit images.

    These are deliberately simple: smooth gradients, noise, flat colour, checkerboards
    and geometric shapes. They represent "not a supported fruit photo" far better than
    another fruit dataset would, and they require no external download.
    """
    rng = np.random.default_rng(SEED)
    size = 224
    images: list[np.ndarray] = []
    labels: list[str] = []

    for _ in range(count):
        kind = rng.choice(["noise", "gradient", "flat", "checker", "shapes", "text"])
        if kind == "noise":
            images.append(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))
        elif kind == "gradient":
            ramp = np.linspace(0, 255, size, dtype=np.float32)
            channel = rng.integers(0, 3)
            canvas = np.zeros((size, size, 3), dtype=np.float32)
            for c in range(3):
                canvas[..., c] = ramp if c == channel else ramp.T * rng.uniform(0.2, 1.0)
            images.append(canvas.astype(np.uint8))
        elif kind == "flat":
            images.append(np.full((size, size, 3), rng.integers(0, 256), dtype=np.uint8))
        elif kind == "checker":
            cell = int(rng.integers(8, 40))
            grid = ((np.add.outer(np.arange(size) // cell, np.arange(size) // cell)) % 2) * 255
            images.append(np.stack([grid] * 3, axis=-1).astype(np.uint8))
        elif kind == "shapes":
            canvas = Image.new("RGB", (size, size), tuple(int(v) for v in rng.integers(180, 255, 3)))
            from PIL import ImageDraw

            draw = ImageDraw.Draw(canvas)
            for _ in range(6):
                # ImageDraw requires x1 >= x0 and y1 >= y0, so order each pair explicitly.
                x0, x1 = sorted(int(v) for v in rng.integers(0, size, 2))
                y0, y1 = sorted(int(v) for v in rng.integers(0, size, 2))
                draw.rectangle([x0, y0, x1, y1], fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
            images.append(np.asarray(canvas))
        else:  # text-like stripes
            canvas = np.full((size, size, 3), 245, dtype=np.uint8)
            for row in range(10, size - 10, 14):
                canvas[row:row + 6, 12:size - 12] = rng.integers(0, 60)
            images.append(canvas)
        labels.append(f"synthetic_{kind}")
    return np.stack(images), labels


def degrade(images: np.ndarray, kind: str) -> np.ndarray:
    import cv2

    out = []
    for image in images:
        if kind == "blur":
            out.append(cv2.GaussianBlur(image, (31, 31), 0))
        elif kind == "dark":
            out.append((image.astype(np.float32) * 0.06).astype(np.uint8))
        elif kind == "tiny":
            small = cv2.resize(image, (48, 48), interpolation=cv2.INTER_AREA)
            out.append(cv2.resize(small, (224, 224), interpolation=cv2.INTER_LINEAR))
        else:
            out.append(image)
    return np.stack(out)


def main() -> int:
    for directory in (METRICS_DIR, CM_DIR, RESULTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    import tensorflow as tf

    model = tf.keras.models.load_model(DEFAULT_MODEL_PATH)
    meta = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    class_names: list[str] = meta["class_names"]
    fruits: list[str] = meta["fruits"]

    x_test, y_test = load_split("test")
    x_val, y_val = load_split("val")
    print(f"test {x_test.shape}, val {x_val.shape}")

    started = time.perf_counter()
    test_probs = model.predict(x_test, batch_size=32, verbose=0)
    batch_seconds = time.perf_counter() - started
    val_probs = model.predict(x_val, batch_size=32, verbose=0)

    y_pred = test_probs.argmax(axis=1)
    accuracy = float(accuracy_score(y_test, y_pred))
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    report_dict = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    report_text = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    (METRICS_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
    pd.DataFrame(report_dict).transpose().to_csv(METRICS_DIR / "classification_report.csv")

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(METRICS_DIR / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=class_names, yticklabels=class_names, ax=ax, annot_kws={"size": 8},
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix — 16-class test set (n={len(y_test)}, accuracy={accuracy:.1%})")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(CM_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # ---- aggregation level metrics (what the website actually reports) ----------
    fruit_of = lambda idx: idx // 2  # noqa: E731 — CLASS_NAMES is fruit-major, fresh/rotten pairs
    fruit_true = np.array([fruit_of(i) for i in y_test])
    fruit_pred = test_probs.reshape(len(test_probs), len(fruits), 2).sum(axis=2).argmax(axis=1)
    condition_true = y_test % 2
    condition_pred = test_probs.reshape(len(test_probs), len(fruits), 2).sum(axis=1).argmax(axis=1)

    # ---- calibration ------------------------------------------------------------
    temperature = calibrate_temperature(val_probs, y_val)
    calibrated_test = softmax_from(log_probabilities(test_probs) / temperature)
    ece_before = expected_calibration_error(test_probs, y_test)
    ece_after = expected_calibration_error(calibrated_test, y_test)
    plot_reliability(test_probs, y_test, FIGURES_DIR / "calibration_before.png",
                     f"Reliability before calibration (ECE={ece_before:.3f})")
    plot_reliability(calibrated_test, y_test, FIGURES_DIR / "calibration_after.png",
                     f"Reliability after temperature scaling T={temperature:.2f} (ECE={ece_after:.3f})")
    nll_before = float(-np.log(np.clip(test_probs[np.arange(len(y_test)), y_test], 1e-12, 1)).mean())
    nll_after = float(-np.log(np.clip(calibrated_test[np.arange(len(y_test)), y_test], 1e-12, 1)).mean())

    # ---- inference timing -------------------------------------------------------
    single_times = []
    for i in range(min(40, len(x_test))):
        t0 = time.perf_counter()
        model.predict(x_test[i:i + 1], verbose=0)
        single_times.append((time.perf_counter() - t0) * 1000.0)

    # Quality reference values now live in Models/gate_config.json, fitted on validation
    # by ``python -m SRC.models.calibrate_gate``; that split is also used for the gate.

    # ---- out-of-distribution behaviour ------------------------------------------
    ood_images, ood_labels = make_ood_probes()
    degraded = {
        "blur": degrade(x_test[:100], "blur"),
        "dark": degrade(x_test[:100], "dark"),
        "tiny": degrade(x_test[:100], "tiny"),
    }

    # ---- serving gate: thresholds come from validation, never from this split ------
    from SRC.models.predictor import load_gate_config

    gate = load_gate_config()
    fruit_gate = gate.get("fruit", {})
    gate_min_confidence = float(fruit_gate.get("min_confidence", 0.0))
    gate_min_margin = float(fruit_gate.get("min_margin", 0.0))

    def fruit_decisions(probabilities: np.ndarray):
        """Fruit-level confidence and margin, exactly as the serving gate computes them."""
        fruit = probabilities.reshape(len(probabilities), len(fruits), 2).sum(axis=2)
        ordered = np.sort(fruit, axis=1)
        confidence = ordered[:, -1]
        margin = ordered[:, -1] - ordered[:, -2]
        accepted = (confidence >= gate_min_confidence) & (margin >= gate_min_margin)
        return confidence, margin, accepted

    test_conf, test_margin, test_accepted = fruit_decisions(calibrated_test)
    ood_probs = softmax_from(
        log_probabilities(model.predict(ood_images, batch_size=32, verbose=0)) / temperature
    )
    degraded_scores = {
        name: softmax_from(log_probabilities(model.predict(batch, batch_size=32, verbose=0)) / temperature)
        for name, batch in degraded.items()
    }

    ood_conf, _, ood_accepted = fruit_decisions(ood_probs)
    ood_rejected = float((~ood_accepted).mean())
    degraded_rejection = {
        name: float((~fruit_decisions(probs)[2]).mean()) for name, probs in degraded_scores.items()
    }
    in_distribution_rejected = float((~test_accepted).mean())

    ood_report = {
        "gate": "fruit identity confidence (sum of the fruit's fresh and rotten classes) "
                "plus a fruit-margin floor; thresholds fitted on the validation split only",
        "min_fruit_confidence": gate_min_confidence,
        "min_fruit_margin": gate_min_margin,
        "in_distribution_test_rejection_rate": in_distribution_rejected,
        "in_distribution_test_confidence": {
            "median": float(np.median(test_conf)),
            "p01": float(np.percentile(test_conf, 1)),
        },
        "synthetic_non_fruit": {
            "n": int(len(ood_images)),
            "kinds": pd.Series(ood_labels).value_counts().to_dict(),
            "rejection_rate": ood_rejected,
            "mean_fruit_confidence": float(ood_conf.mean()),
            "test_mean_fruit_confidence": float(test_conf.mean()),
        },
        "degraded_fruit": {name: {"rejection_rate": rate} for name, rate in degraded_rejection.items()},
        "caveat": "Synthetic probes are a sanity check, not a benchmark. Unseen fruit species are "
                  "the hardest case and are NOT reliably rejected; the site reports the closest "
                  "supported fruit, the gate status and the confidence instead of pretending the "
                  "image is certainly one of the eight fruits.",
    }
    (RESULTS_DIR / "ood_probe.json").write_text(json.dumps(ood_report, indent=2), encoding="utf-8")

    # ---- per-fruit detail (reuses the report and matrix computed above) --------------
    per_fruit = {}
    for fruit_idx, fruit_name in enumerate(fruits):
        fresh_idx, rotten_idx = fruit_idx * 2, fruit_idx * 2 + 1
        mask = (y_test == fresh_idx) | (y_test == rotten_idx)
        fresh_mask = y_test == fresh_idx
        rotten_mask = y_test == rotten_idx
        per_fruit[fruit_name] = {
            "n_test": int(mask.sum()),
            "n_fresh": int(fresh_mask.sum()),
            "n_rotten": int(rotten_mask.sum()),
            "fresh_accuracy": float((y_pred[fresh_mask] == fresh_idx).mean()),
            "rotten_accuracy": float((y_pred[rotten_mask] == rotten_idx).mean()),
            "fruit_level_accuracy": float((fruit_pred[mask] == fruit_idx).mean()),
        }
    metrics_detail = {
        "report": report_dict,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "validated_gate": test_accepted.tolist(),
        "per_fruit": per_fruit,
    }
    (METRICS_DIR / "per_class_report.json").write_text(json.dumps(metrics_detail, indent=2), encoding="utf-8")

    # ---- per-image predictions --------------------------------------------------
    frame = split_frame("test")
    predictions = pd.DataFrame(
        {
            "relative_path": frame["relative_path"],
            "true_class": [class_names[i] for i in y_test],
            "predicted_class": [class_names[i] for i in y_pred],
            "correct": y_pred == y_test,
            "top1_confidence": test_probs.max(axis=1),
            "fresh_probability": test_probs[:, ::2].sum(axis=1),
            "rotten_probability": test_probs[:, 1::2].sum(axis=1),
        }
    )
    predictions.to_csv(RESULTS_DIR / "predictions.csv", index=False)

    curves = plot_training_curves(MODELS_DIR / "final_model" / "history.json")

    metrics = {
        "model": "MobileNetV2 transfer learning (16-class softmax)",
        "test_split": {"n": int(len(y_test)), "per_class": int(len(y_test) / len(class_names))},
        "accuracy": accuracy,
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "fruit_level_accuracy": float(accuracy_score(fruit_true, fruit_pred)),
        "condition_level_accuracy": float(accuracy_score(condition_true, condition_pred)),
        "calibration": {
            "temperature": temperature,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "nll_before": nll_before,
            "nll_after": nll_after,
            "method": "single-parameter temperature scaling fitted on the validation split",
        },
        "timing": {
            "batch32_seconds_480_images": batch_seconds,
            "batch32_images_per_second": float(len(x_test) / batch_seconds),
            "single_image_ms_mean": float(np.mean(single_times)),
            "single_image_ms_median": float(np.median(single_times)),
            "single_image_ms_p95": float(np.percentile(single_times, 95)),
            "device": "CPU (TensorFlow " + tf.__version__ + ")",
        },
        "out_of_distribution": ood_report,
        "gate": {
            "min_fruit_confidence": gate_min_confidence,
            "min_fruit_margin": gate_min_margin,
            "condition_min_confidence": float((gate.get("condition") or {}).get("min_confidence", 0.0)),
            "calibrated_on": gate.get("calibrated_on"),
            "test_rejection_rate": in_distribution_rejected,
        },
        "per_class": report_dict,
        "per_fruit": per_fruit,
        "training_curves": curves,
        "seed": SEED,
    }
    (METRICS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps({k: metrics[k] for k in (
        "accuracy", "f1_macro", "fruit_level_accuracy", "condition_level_accuracy",
    )}, indent=2))
    print("calibration:", {k: round(v, 4) for k, v in metrics["calibration"].items() if isinstance(v, float)})
    print("OOD:", {"gate_conf": round(gate_min_confidence, 3), "gate_margin": round(gate_min_margin, 3),
                   "synthetic_rejected": round(ood_rejected, 3),
                   "in_dist_rejected": round(in_distribution_rejected, 3), **degraded_rejection})
    print("timing:", metrics["timing"]["single_image_ms_mean"], "ms/image")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
