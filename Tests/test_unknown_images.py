"""Behaviour on images the model was never trained to classify.

These tests do not assert a specific prediction for noise or a mango: they assert the
*contract* — the system must either reject the image with the documented message or
attach an explicit uncertainty warning, and it must never present a confident answer
without a safety disclaimer.
"""
from __future__ import annotations

import numpy as np
import pytest

from SRC.models.predictor import UNKNOWN_MESSAGE

pytestmark = pytest.mark.usefixtures("predictor")


def _noise(seed: int = 0, shape=(224, 224, 3)) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def test_pure_noise_is_rejected_or_warned(predictor) -> None:
    result = predictor.analyze(_noise(1))
    assert result["ok"] is True
    if result["uncertainty"]["rejected"]:
        assert result["uncertainty"]["message"] == UNKNOWN_MESSAGE
    assert result["warnings"], "a noise image must produce at least one warning"


def test_noise_generally_less_confident_than_real_fruit(predictor, test_images) -> None:
    noise_confidences = [predictor.analyze(_noise(seed))["identification_confidence"] for seed in range(6)]
    fruit_confidences = [
        predictor.analyze(np.asarray(test_images[i]))["identification_confidence"] for i in range(6)
    ]
    assert np.mean(noise_confidences) < np.mean(fruit_confidences)


def test_flat_colour_image_is_flagged(predictor) -> None:
    flat = np.full((224, 224, 3), 200, dtype=np.uint8)
    result = predictor.analyze(flat)
    assert result["ok"] is True
    assert result["uncertainty"]["rejected"] or result["warnings"]


def test_very_dark_image_is_reported(predictor, sample_rgb: np.ndarray) -> None:
    dark = (sample_rgb.astype(np.float32) * 0.02).astype(np.uint8)
    result = predictor.analyze(dark)
    reasons = " ".join(result["uncertainty"]["reasons"]).lower()
    assert "dark" in reasons or result["warnings"]


def test_blurry_image_is_reported(predictor, sample_rgb: np.ndarray) -> None:
    """Blur is advice about the photograph, not a reason to discard the prediction."""
    import cv2

    blurry = cv2.GaussianBlur(sample_rgb, (41, 41), 0)
    result = predictor.analyze(blurry)
    warnings = " ".join(result["warnings"]).lower()
    reasons = " ".join(result["uncertainty"]["reasons"]).lower()
    assert "soft" in warnings or "focus" in warnings
    assert "blur" not in reasons


def test_multi_object_image_still_returns_valid_payload(predictor, sample_rgb: np.ndarray) -> None:
    import cv2

    small = cv2.resize(sample_rgb, (96, 96))
    tiled = np.tile(small, (2, 2, 1))[:192, :192]
    result = predictor.analyze(tiled)
    assert result["ok"] is True
    assert 0.0 <= result["fresh_probability"] <= 1.0


def test_unseen_use_case_is_not_claimed_to_be_safe(predictor) -> None:
    """Whatever happens, an unrecognised image never yields a safety guarantee."""
    result = predictor.analyze(_noise(3))
    assert "cannot guarantee food safety" in result["edibility"]["disclaimer"]
    assert result["shelf_life"]["available"] is False
