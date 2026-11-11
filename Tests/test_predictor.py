"""Prediction pipeline contract: scores, disclaimer and unavailable features."""
from __future__ import annotations

import numpy as np
import pytest

from SRC.models.predictor import SAFETY_DISCLAIMER, SHELF_LIFE_MESSAGE, UNKNOWN_MESSAGE


def test_result_contract(predictor, sample_rgb: np.ndarray) -> None:
    result = predictor.analyze(sample_rgb, source_name="unit-test.png")

    for key in (
        "ok", "fruit", "condition", "fresh_probability", "rotten_probability",
        "identification_confidence", "condition_confidence", "predicted_class",
        "class_probabilities", "fruit_probabilities", "verdict", "fruit_identification",
        "condition_assessment", "edibility", "shelf_life", "uncertainty", "warnings",
        "image_info", "model", "inference_ms", "notes",
    ):
        assert key in result, key
    assert result["ok"] is True
    assert result["fruit"] in predictor.fruits
    assert result["predicted_class"] in predictor.class_names


def test_probabilities_are_normalised(predictor, sample_rgb: np.ndarray) -> None:
    result = predictor.analyze(sample_rgb)
    assert len(result["class_probabilities"]) == 16
    assert np.isclose(sum(result["class_probabilities"].values()), 1.0, atol=1e-5)
    assert np.isclose(sum(result["fruit_probabilities"].values()), 1.0, atol=1e-5)
    assert np.isclose(result["fresh_probability"] + result["rotten_probability"], 1.0, atol=1e-5)


def test_condition_aggregation_matches_class_scores(predictor, sample_rgb: np.ndarray) -> None:
    result = predictor.analyze(sample_rgb)
    probabilities = result["class_probabilities"]
    fresh = sum(p for name, p in probabilities.items() if name.startswith("Fresh"))
    rotten = sum(p for name, p in probabilities.items() if name.startswith("Rotten"))
    assert np.isclose(fresh, result["fresh_probability"], atol=1e-5)
    assert np.isclose(rotten, result["rotten_probability"], atol=1e-5)
    expected = "fresh" if fresh >= rotten else "rotten"
    assert result["condition"] == expected


def test_identification_confidence_is_fruit_level(predictor, sample_rgb: np.ndarray) -> None:
    result = predictor.analyze(sample_rgb)
    assert np.isclose(
        result["identification_confidence"],
        result["fruit_probabilities"][result["fruit"]],
        atol=1e-6,
    )


def test_shelf_life_is_never_invented(predictor, sample_rgb: np.ndarray) -> None:
    shelf = predictor.analyze(sample_rgb)["shelf_life"]
    assert shelf["available"] is False
    assert shelf["days_estimate"] is None
    assert shelf["message"] == SHELF_LIFE_MESSAGE
    assert "cannot currently be estimated" in shelf["message"].lower()
    assert "day" not in shelf["message"].lower()


def test_safety_disclaimer_present(predictor, sample_rgb: np.ndarray) -> None:
    edibility = predictor.analyze(sample_rgb)["edibility"]
    assert edibility["disclaimer"] == SAFETY_DISCLAIMER
    assert "cannot guarantee food safety" in edibility["disclaimer"]
    assert edibility["status"] in {"appears_fresh", "shows_spoilage", "uncertain"}
    assert edibility["status"] != "safe_to_eat"


def test_probability_note_is_honest(predictor, sample_rgb: np.ndarray) -> None:
    notes = predictor.analyze(sample_rgb)["notes"]
    assert "not the physical proportion" in notes["probabilities"]
    assert "whole-fruit" in notes["scope"]


def test_uncertainty_payload_shape(predictor, sample_rgb: np.ndarray) -> None:
    uncertainty = predictor.analyze(sample_rgb)["uncertainty"]
    assert isinstance(uncertainty["rejected"], bool)
    assert isinstance(uncertainty["reasons"], list)
    if uncertainty["rejected"]:
        assert uncertainty["message"] == UNKNOWN_MESSAGE


def test_rgba_and_grayscale_inputs_are_accepted(predictor, sample_rgb: np.ndarray) -> None:
    rgba = np.dstack([sample_rgb, np.full(sample_rgb.shape[:2], 255, dtype=np.uint8)])
    assert predictor.analyze(rgba)["ok"] is True
    gray = np.asarray(sample_rgb).mean(axis=2).astype(np.uint8)
    assert predictor.analyze(gray)["ok"] is True


def test_pil_image_input_supported(predictor, sample_rgb: np.ndarray) -> None:
    from PIL import Image

    result = predictor.analyze(Image.fromarray(sample_rgb))
    assert result["ok"] is True


def test_inference_time_reported(predictor, sample_rgb: np.ndarray) -> None:
    result = predictor.analyze(sample_rgb)
    assert result["inference_ms"] > 0
    assert result["model"]["input_size"] == 224


def test_float_array_is_clipped_not_crashing(predictor, sample_rgb: np.ndarray) -> None:
    floats = sample_rgb.astype(np.float32) * 1.5
    assert predictor.analyze(floats)["ok"] is True


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_repeated_calls_are_deterministic(predictor, sample_rgb: np.ndarray, seed: int) -> None:
    first = predictor.analyze(sample_rgb)["class_probabilities"]
    second = predictor.analyze(sample_rgb)["class_probabilities"]
    assert first == pytest.approx(second, abs=1e-6)
