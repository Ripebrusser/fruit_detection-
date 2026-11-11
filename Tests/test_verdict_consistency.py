"""Regression tests for the reported rotten-pomegranate failure.

The bug was not a wrong prediction: on 30 held-out rotten pomegranates the model named the
right fruit 29 times and the right class 28 times. The failure was that the serving gate
rejected the whole image — mostly on a blur measurement — while the result page went on to
show a confident rotten score. One photograph, two contradictory answers.

These tests pin down the fixed contract:

* a confidently recognised fruit is never discarded because of capture quality,
* identity and condition are separate decisions with separate confidences,
* an unidentified image never displays a condition as if it were known,
* every one of the sixteen classes produces a self-consistent payload.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from SRC.models.predictor import UNKNOWN_MESSAGE

pytestmark = pytest.mark.usefixtures("predictor")

STATES = {"identified", "identified_condition_uncertain", "unidentified"}


def _rows(manifest, split: str, class_name: str | None = None) -> list[dict]:
    rows = [r for r in manifest if r["split"] == split]
    if class_name:
        rows = [r for r in rows if r["class_name"] == class_name]
    return rows


def test_rotten_pomegranate_is_not_rejected(predictor, manifest) -> None:
    """Measured before the fix: 14 of 30 rejected. After: 2 of 30, both genuine errors."""
    rows = _rows(manifest, "test", "RottenPomegranate")
    assert len(rows) == 30

    results = [predictor.analyze(Image.open(r["relative_path"])) for r in rows]
    identified = [r for r in results if r["verdict"]["state"] != "unidentified"]
    named_pomegranate = [r for r in identified if r["verdict"]["fruit"] == "Pomegranate"]

    assert len(identified) >= 27, f"only {len(identified)}/30 rotten pomegranates were identified"
    assert len(named_pomegranate) >= 27, f"only {len(named_pomegranate)}/30 named Pomegranate"

    for result in results:
        assert result["verdict"]["state"] in STATES


def test_no_payload_contradicts_itself(predictor, manifest) -> None:
    """An unidentified fruit must never carry a condition verdict or an edibility claim."""
    sample = _rows(manifest, "test")[::12]
    for row in sample:
        result = predictor.analyze(Image.open(row["relative_path"]))
        verdict = result["verdict"]
        assert verdict["state"] in STATES

        if verdict["state"] == "unidentified":
            assert verdict["fruit"] is None
            assert verdict["condition"] is None
            assert result["edibility"]["status"] == "not_assessed"
            assert result["uncertainty"]["rejected"] is True
            assert result["uncertainty"]["message"] == UNKNOWN_MESSAGE
        else:
            assert verdict["fruit"] == result["fruit"]
            assert result["fruit"].lower() in verdict["headline"].lower()
            assert result["uncertainty"]["rejected"] is False

        if verdict["state"] == "identified_condition_uncertain":
            assert verdict["condition"] is None
            assert result["edibility"]["status"] == "uncertain"
        if verdict["state"] == "identified":
            assert verdict["condition"] == result["condition"]
            assert result["edibility"]["status"] in {"appears_fresh", "shows_spoilage"}


def test_every_class_produces_a_consistent_payload(predictor, manifest) -> None:
    for class_name in predictor.class_names:
        row = _rows(manifest, "test", class_name)[0]
        result = predictor.analyze(Image.open(row["relative_path"]), source_name=row["file_name"])

        assert result["ok"] is True
        assert result["predicted_class"] in predictor.class_names
        assert result["fruit"] in predictor.fruits
        assert result["verdict"]["state"] in STATES
        assert result["fresh_probability"] + result["rotten_probability"] == pytest.approx(1.0, abs=1e-6)
        assert result["identification_confidence"] == pytest.approx(
            result["fruit_probabilities"][result["fruit"]], abs=1e-6
        )
        assert result["shelf_life"]["available"] is False


def test_blur_is_advice_not_a_rejection_reason(predictor, sample_rgb: np.ndarray) -> None:
    """Quality measurements describe the photograph, not which fruit it contains."""
    import cv2

    blurred = cv2.GaussianBlur(sample_rgb, (41, 41), 0)
    result = predictor.analyze(blurred)
    reasons = " ".join(result["uncertainty"]["reasons"]).lower()

    assert "blur" not in reasons
    assert "focus" not in reasons
    assert any("soft" in w.lower() or "focus" in w.lower() for w in result["warnings"])


def test_uncertain_condition_keeps_the_fruit_identified(
    predictor, sample_rgb: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 52/48 fresh-rotten split must not hide a fruit the model recognises."""
    fake = {
        "class_probabilities": {name: 0.0 for name in predictor.class_names},
        "top_class": "FreshApple",
        "top_class_probability": 0.52,
        "runner_up_probability": 0.48,
        "class_margin": 0.04,
        "fruit_probabilities": {fruit: 0.0 for fruit in predictor.fruits} | {"Apple": 0.99},
        "predicted_fruit": "Apple",
        "fruit_confidence": 0.99,
        "fruit_margin": 0.90,
        "fresh_probability": 0.52,
        "rotten_probability": 0.48,
        "entropy": 1.0,
        "entropy_ratio": 0.35,
        "inference_ms": 12.0,
        "temperature": 1.0,
    }
    fake["class_probabilities"]["FreshApple"] = 0.52
    fake["class_probabilities"]["RottenApple"] = 0.47

    monkeypatch.setattr(predictor, "predict_array", lambda rgb: dict(fake))
    result = predictor.analyze(sample_rgb)

    assert result["verdict"]["state"] == "identified_condition_uncertain"
    assert result["verdict"]["fruit"] == "Apple"
    assert result["verdict"]["condition"] is None
    assert result["edibility"]["status"] == "uncertain"
    assert result["fruit_identification"]["status"] == "identified"


def test_condition_confidence_is_reported_separately(predictor, sample_rgb: np.ndarray) -> None:
    result = predictor.analyze(sample_rgb)
    assert result["condition_confidence"] == pytest.approx(
        max(result["fresh_probability"], result["rotten_probability"]), abs=1e-9
    )
    assert 0.0 <= result["condition_confidence"] <= 1.0
    assert result["condition_assessment"]["status"] in {"confident", "uncertain"}


def test_noise_is_rejected_without_a_condition_claim(predictor) -> None:
    noise = np.random.default_rng(11).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    result = predictor.analyze(noise)
    if result["verdict"]["state"] == "unidentified":
        assert result["verdict"]["fruit"] is None
        assert result["edibility"]["status"] == "not_assessed"
        assert result["uncertainty"]["message"] == UNKNOWN_MESSAGE
    else:
        assert result["warnings"], "an odd image must still raise a warning"
