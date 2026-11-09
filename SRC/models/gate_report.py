"""Measure the *serving* uncertainty gate (not just the top-1 threshold).

``SRC.models.evaluate`` calibrates the top-1 probability threshold. The website also
applies the fruit-margin, entropy and image-quality checks, so the rejection rate a
user actually experiences is different. This script measures the full predictor gate on:

* synthetic non-fruit probes (noise, gradients, flat colour, checkerboards, shapes, text),
* blurred / very dark / down-sampled versions of real test images,
* a sample of untouched test images (the false-rejection rate).

Output: ``Evaluation/test_results/gate_probe.json``

Usage:
    python -m SRC.models.gate_report
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from SRC.models.evaluate import degrade, make_ood_probes
from SRC.models.predictor import FruitPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "Evaluation" / "test_results"


def measure(predictor: FruitPredictor, images: np.ndarray) -> dict:
    """Rejection statistics for a batch of images under the full analyser gate."""
    rejected, confidences, reasons = 0, [], {}
    for image in images:
        result = predictor.analyze(image)
        confidences.append(result["identification_confidence"])
        if result["uncertainty"]["rejected"]:
            rejected += 1
            for reason in result["uncertainty"]["reasons"]:
                key = reason.split("(")[0].strip()[:60]
                reasons[key] = reasons.get(key, 0) + 1
    n = len(images)
    return {
        "n": n,
        "rejection_rate": rejected / n if n else 0.0,
        "mean_fruit_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "rejection_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    predictor = FruitPredictor()
    test_images = np.load(PROJECT_ROOT / "Dataset" / "processed" / "test_images.npy")

    probes, labels = make_ood_probes()
    sample = np.asarray(test_images[:120])

    report = {
        "gate": {
            "min_top1_probability": predictor.config["min_top1_probability"],
            "min_fruit_margin": predictor.config["min_fruit_margin"],
            "max_entropy_ratio": predictor.config["max_entropy_ratio"],
            "min_blur_variance": predictor.config["min_blur_variance"],
            "min_brightness": predictor.config["min_brightness"],
            "temperature": predictor.config["temperature"],
        },
        "in_distribution_test_sample": measure(predictor, sample),
        "synthetic_non_fruit": measure(predictor, probes),
        "degraded_fruit": {
            "blur": measure(predictor, degrade(sample[:50], "blur")),
            "dark": measure(predictor, degrade(sample[:50], "dark")),
            "tiny": measure(predictor, degrade(sample[:50], "tiny")),
        },
        "probe_kinds": labels[:5],
        "summary": (
            "The full gate combines a calibrated top-1 threshold with fruit-margin, entropy and "
            "image-quality checks. It reliably flags degraded photographs but only rejects a "
            "minority of synthetic non-fruit patterns: the model is over-confident on inputs very "
            "far from the training distribution. That is why every result also carries a warning "
            "panel and why the UI never presents a confident answer as certain."
        ),
    }
    (RESULTS_DIR / "gate_probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k != "summary"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
