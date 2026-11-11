"""Calibrate the serving gate on the validation split only.

Why this module exists
----------------------
The first release derived its acceptance threshold from the *test* split, which both
violated the "never tune on test" rule and produced a gate that rejected genuine,
confidently classified fruit photographs: 55 of 480 test images were rejected, 47 of
them on a blur threshold alone, including images the network classified at 99.9%
confidence. Fourteen of thirty rotten pomegranates were rejected while thirteen of them
were near-certain and correctly named, which is exactly the contradiction the website
showed: "Unable to identify this fruit confidently" on one card and a confident rotten
score on another.

The replacement gate separates three questions that were previously fused:

1. **Fruit identity** — is this a supported fruit at all? Scored with the summed
   probability of that fruit's two classes (fresh + rotten), so a close fresh/rotten
   split can no longer hide a well-identified fruit.
2. **Condition** — is the fresh/rotten split decisive? Has its own threshold and its own
   confidence; a weak condition never rejects the image.
3. **Capture quality** — blur, exposure and resolution only produce warnings. They are
   descriptive measurements, not evidence that the fruit is a different species.

Thresholds are fitted on the validation split, targeting a false-rejection rate of at
most 2% on genuine fruit photographs, and the resulting trade-off against synthetic
non-fruit probes is measured and reported rather than assumed.

Usage:
    python -m SRC.models.calibrate_gate
"""
from __future__ import annotations

import json

import numpy as np

from SRC.models.evaluate import (
    DEFAULT_MODEL_PATH,
    RESULTS_DIR,
    calibrate_temperature,
    degrade,
    load_split,
    make_ood_probes,
)
from SRC.utils.imaging import quality_checks

PROJECT_ROOT = DEFAULT_MODEL_PATH.parents[2]
MODELS_DIR = PROJECT_ROOT / "Models"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
GATE_CONFIG_PATH = MODELS_DIR / "gate_config.json"

TARGET_FALSE_REJECTION = 0.02


def _fruit_matrix(probabilities: np.ndarray, fruits: list[str]) -> np.ndarray:
    """Collapse the 16-class softmax into eight (fresh, rotten) pairs."""
    return probabilities.reshape(len(probabilities), len(fruits), 2).sum(axis=2)


def _condition_matrix(probabilities: np.ndarray, fruits: list[str]) -> np.ndarray:
    return probabilities.reshape(len(probabilities), len(fruits), 2).sum(axis=1)


def _summarise(scores: np.ndarray) -> dict:
    return {
        "min": float(scores.min()),
        "p01": float(np.percentile(scores, 1)),
        "median": float(np.median(scores)),
        "mean": float(scores.mean()),
        "max": float(scores.max()),
    }


def choose_fruit_threshold(val_conf: np.ndarray, ood_conf: np.ndarray) -> tuple[float, list[dict]]:
    """Pick the acceptance threshold with the best measured trade-off.

    Constraint: at most ``TARGET_FALSE_REJECTION`` of genuine validation photographs may
    be rejected. Among the thresholds that satisfy it, the one that rejects the most
    synthetic non-fruit probes wins; ties break toward the more conservative threshold.
    """
    grid = np.round(np.arange(0.20, 0.991, 0.01), 4)
    rows: list[dict] = []
    best: tuple[float, float] | None = None
    for threshold in grid:
        false_rejection = float((val_conf < threshold).mean())
        ood_rejection = float((ood_conf < threshold).mean())
        rows.append(
            {
                "threshold": float(threshold),
                "val_false_rejection_rate": false_rejection,
                "synthetic_ood_rejection_rate": ood_rejection,
            }
        )
        if false_rejection > TARGET_FALSE_REJECTION:
            continue
        if best is None or (ood_rejection, float(threshold)) > (best[1], best[0]):
            best = (float(threshold), ood_rejection)

    if best is None:  # pragma: no cover - only if validation is pathologically uncertain
        return float(np.percentile(val_conf, 1.0)), rows

    # Blend with the 1st percentile so the operating point tracks the data as it changes.
    chosen = max(best[0], float(np.percentile(val_conf, 1.0)))
    return round(float(chosen), 4), rows


def main() -> int:
    import tensorflow as tf

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    fruits: list[str] = meta["fruits"]

    model = tf.keras.models.load_model(DEFAULT_MODEL_PATH)
    x_val, y_val = load_split("val")
    val_probs = model.predict(x_val, batch_size=32, verbose=0)

    # ---- temperature: fitted on validation, applied to both splits -----------------
    temperature = calibrate_temperature(val_probs, y_val)
    log_val = np.log(np.clip(val_probs, 1e-12, 1.0))
    log_val -= log_val.max(axis=1, keepdims=True)
    exp_val = np.exp(log_val / temperature)
    cal_val = exp_val / exp_val.sum(axis=1, keepdims=True)

    val_fruit = _fruit_matrix(cal_val, fruits)
    val_cond = _condition_matrix(cal_val, fruits)
    val_fruit_conf = val_fruit.max(axis=1)
    val_fruit_sorted = np.sort(val_fruit, axis=1)
    val_fruit_margin = val_fruit_sorted[:, -1] - val_fruit_sorted[:, -2]
    val_cond_conf = val_cond.max(axis=1)

    # ---- synthetic non-fruit probes ------------------------------------------------
    ood_images, ood_labels = make_ood_probes(120)
    ood_probs = model.predict(ood_images, batch_size=32, verbose=0)
    log_ood = np.log(np.clip(ood_probs, 1e-12, 1.0))
    log_ood -= log_ood.max(axis=1, keepdims=True)
    exp_ood = np.exp(log_ood / temperature)
    cal_ood = exp_ood / exp_ood.sum(axis=1, keepdims=True)
    ood_fruit_conf = _fruit_matrix(cal_ood, fruits).max(axis=1)

    fruit_threshold, sweep = choose_fruit_threshold(val_fruit_conf, ood_fruit_conf)

    # ---- margin and condition thresholds -------------------------------------------
    fruit_margin_threshold = round(float(np.clip(np.percentile(val_fruit_margin, 1.0), 0.05, 0.30)), 4)
    condition_threshold = round(float(np.clip(np.percentile(val_cond_conf, 1.0), 0.60, 0.95)), 4)

    # ---- capture-quality reference values (warnings only) ---------------------------
    quality_rows = [quality_checks(image) for image in x_val]
    blur = np.array([q["blur_variance"] for q in quality_rows])
    brightness = np.array([q["brightness"] for q in quality_rows])
    quality_reference = {
        "min_blur_variance": round(float(np.percentile(blur, 1.0)), 3),
        "min_brightness": round(float(np.percentile(brightness, 1.0)), 3),
        "max_brightness": 250.0,
        "note": "Reference values only: quality measurements produce warnings, never rejections.",
    }

    # ---- measured behaviour of the chosen operating point ---------------------------
    degraded = {kind: degrade(x_val[:80], kind) for kind in ("blur", "dark", "tiny")}
    degraded_rates = {}
    for kind, batch in degraded.items():
        probs = model.predict(batch, batch_size=32, verbose=0)
        logs = np.log(np.clip(probs, 1e-12, 1.0))
        logs -= logs.max(axis=1, keepdims=True)
        exp = np.exp(logs / temperature)
        calibrated = exp / exp.sum(axis=1, keepdims=True)
        fruit = _fruit_matrix(calibrated, fruits)
        deck = np.sort(fruit, axis=1)
        conf = deck[:, -1]
        accepted = (conf >= fruit_threshold) & ((deck[:, -1] - deck[:, -2]) >= fruit_margin_threshold)
        degraded_rates[kind] = {
            "n": int(len(batch)),
            "rejection_rate": float((~accepted).mean()),
            "mean_fruit_confidence": float(conf.mean()),
        }

    reviewed = (val_fruit_conf >= fruit_threshold) & (val_fruit_margin >= fruit_margin_threshold)
    report = {
        "temperature": round(float(temperature), 6),
        "fruit_threshold": fruit_threshold,
        "fruit_margin_threshold": fruit_margin_threshold,
        "condition_threshold": condition_threshold,
        "target_false_rejection": TARGET_FALSE_REJECTION,
        "validation_false_rejection_rate": float((~reviewed).mean()),
        "synthetic_non_fruit_rejection_rate": float((ood_fruit_conf < fruit_threshold).mean()),
        "validation_fruit_confidence": _summarise(val_fruit_conf),
        "validation_condition_confidence": _summarise(val_cond_conf),
        "ood_fruit_confidence": _summarise(ood_fruit_conf),
        "degraded_fruit": degraded_rates,
        "synthetic_kinds": {k: int(v) for k, v in
                            zip(*np.unique(ood_labels, return_counts=True))},
        "threshold_sweep": sweep,
        "method": "Thresholds fitted on the validation split only; the test split is "
                  "untouched here and is used once, later, to report the final gate metrics.",
    }
    (RESULTS_DIR / "gate_calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    config = {
        "version": 2,
        "temperature": round(float(temperature), 6),
        "fruit": {
            "min_confidence": fruit_threshold,
            "min_margin": fruit_margin_threshold,
        },
        "condition": {"min_confidence": condition_threshold},
        "quality": quality_reference,
        "calibrated_on": {
            "split": "validation",
            "n": int(len(x_val)),
            "target_false_rejection": TARGET_FALSE_REJECTION,
            "temperature": "minimised NLL on the validation split",
            "fruit_threshold": "largest synthetic-OOD rejection subject to <=2% validation "
                               "false rejection",
            "condition_threshold": "1st percentile of validation condition confidence, "
                                   "clipped to [0.60, 0.95]",
        },
        "measured_on_validation": {
            "false_rejection_rate": report["validation_false_rejection_rate"],
            "synthetic_non_fruit_rejection_rate": report["synthetic_non_fruit_rejection_rate"],
        },
    }
    GATE_CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"temperature          : {temperature:.4f}")
    print(f"fruit threshold      : {fruit_threshold:.3f}  (margin {fruit_margin_threshold:.3f})")
    print(f"condition threshold  : {condition_threshold:.3f}")
    print(f"val false rejection  : {report['validation_false_rejection_rate']:.2%}")
    print(f"synthetic OOD reject : {report['synthetic_non_fruit_rejection_rate']:.2%}")
    for kind, values in degraded_rates.items():
        print(f"degraded {kind:6s}      : rejected {values['rejection_rate']:.2%}")
    print(f"written -> {GATE_CONFIG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
