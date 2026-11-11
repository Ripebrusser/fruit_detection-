"""Measure the *serving* gate exactly as a visitor experiences it.

``SRC.models.calibrate_gate`` fits the thresholds on validation; ``SRC.models.evaluate``
reports them on the test split. This script exercises the real predictor on four groups so
the numbers describe the deployed behaviour rather than a proxy:

* untouched test images — the false-rejection rate on genuine fruit photographs,
* synthetic non-fruit probes (noise, gradients, flat colour, checkerboards, shapes, text),
* blurred, very dark and down-sampled real photographs,
* flat colour images that contain no fruit at all.

Identity and condition are counted separately, because they are separate decisions: an
image can be confidently identified while its fresh/rotten split stays uncertain.

Output: ``Evaluation/test_results/gate_probe.json``

Usage:
    python -m SRC.models.gate_report
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from SRC.models.evaluate import degrade, make_ood_probes
from SRC.models.predictor import FruitPredictor, load_gate_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "Evaluation" / "test_results"


def measure(predictor: FruitPredictor, images: np.ndarray) -> dict:
    """Identity and condition outcomes for a batch of images."""
    unidentified = 0
    condition_uncertain = 0
    confidences, condition_confidences = [], []
    reasons: dict[str, int] = {}

    for image in images:
        result = predictor.analyze(image)
        confidences.append(result["identification_confidence"])
        condition_confidences.append(result["condition_confidence"])
        if result["verdict"]["state"] == "unidentified":
            unidentified += 1
            for reason in result["uncertainty"]["reasons"]:
                key = reason.split("(")[0].strip()[:60]
                reasons[key] = reasons.get(key, 0) + 1
        elif result["verdict"]["state"] == "identified_condition_uncertain":
            condition_uncertain += 1

    n = len(images)
    return {
        "n": n,
        "unidentified_rate": unidentified / n if n else 0.0,
        "identified_condition_uncertain_rate": condition_uncertain / n if n else 0.0,
        "mean_fruit_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "median_fruit_confidence": float(np.median(confidences)) if confidences else 0.0,
        "mean_condition_confidence": float(np.mean(condition_confidences)) if condition_confidences else 0.0,
        "identity_rejection_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    predictor = FruitPredictor()
    config = load_gate_config()
    test_images = np.load(PROJECT_ROOT / "Dataset" / "processed" / "test_images.npy")

    probes, labels = make_ood_probes()
    sample = np.asarray(test_images[:120])

    kinds, counts = np.unique(labels, return_counts=True)
    report = {
        "gate": {
            "version": config.get("version"),
            "fruit_min_confidence": config["fruit"]["min_confidence"],
            "fruit_min_margin": config["fruit"]["min_margin"],
            "condition_min_confidence": config["condition"]["min_confidence"],
            "temperature": config.get("temperature"),
            "calibrated_on": config.get("calibrated_on"),
        },
        "in_distribution_test_sample": measure(predictor, sample),
        "synthetic_non_fruit": measure(predictor, probes),
        "degraded_fruit": {
            "blur": measure(predictor, degrade(sample[:50], "blur")),
            "dark": measure(predictor, degrade(sample[:50], "dark")),
            "tiny": measure(predictor, degrade(sample[:50], "tiny")),
        },
        "flat_colour": measure(
            predictor,
            np.stack([np.full((224, 224, 3), value, dtype=np.uint8) for value in (0, 64, 128, 200, 255)]),
        ),
        "probe_kinds": dict(zip(kinds.tolist(), counts.tolist())),
        "summary": (
            "Identity and condition are gated separately. Blur, darkness and resolution are "
            "reported as advice and never discard an image, so a genuine photograph with a soft "
            "focus keeps its prediction instead of being told the fruit is unrecognisable. "
            "Synthetic patterns are rejected far more often than by the first release, but unseen "
            "fruit species remain the hard case: the eight-fruit softmax always has an answer, so "
            "the site reports the closest fruit together with the gate status and confidence."
        ),
    }
    (RESULTS_DIR / "gate_probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k != "summary"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
