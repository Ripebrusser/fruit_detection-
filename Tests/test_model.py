"""Model artefact and raw-network checks."""
from __future__ import annotations

import json

import numpy as np
import pytest

from Tests.conftest import MODEL_PATH, PROJECT_ROOT

CLASS_NAMES_PATH = PROJECT_ROOT / "Models" / "class_names.json"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="trained model missing — run SRC.models.train"
)


@pytest.fixture(scope="module")
def model():
    import tensorflow as tf

    return tf.keras.models.load_model(MODEL_PATH)


def test_model_file_and_layout() -> None:
    assert MODEL_PATH.is_file()
    assert MODEL_PATH.stat().st_size > 1_000_000
    meta = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    assert len(meta["class_names"]) == 16
    assert meta["input_size"] == 224


def test_model_io_shapes(model) -> None:
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 16)


def test_model_output_is_probability_distribution(model, sample_rgb: np.ndarray) -> None:
    from SRC.utils.imaging import preprocess_for_model

    batch = preprocess_for_model(sample_rgb, 224)[None, ...]
    probabilities = model.predict(batch, verbose=0)[0]
    assert probabilities.shape == (16,)
    assert np.all(probabilities >= 0.0)
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-4)


def test_model_preprocessing_is_baked_in(model, sample_rgb: np.ndarray) -> None:
    """The saved model must accept raw 0-255 pixels and do its own scaling."""
    from SRC.utils.imaging import preprocess_for_model

    batch = preprocess_for_model(sample_rgb, 224)[None, ...]
    raw = model.predict(batch, verbose=0)
    doubled = model.predict(np.clip(batch * 2.0, 0, 255), verbose=0)
    # Different input scaling must change the output: preprocessing lives in the model,
    # so feeding rescaled data is a different (and wrong) input, not a no-op.
    assert not np.allclose(raw, doubled, atol=1e-3)


def test_model_is_better_than_chance_on_test_batch(model, test_images, test_labels) -> None:
    predictions = model.predict(np.asarray(test_images[:96]).astype("float32"), verbose=0)
    accuracy = float((predictions.argmax(axis=1) == test_labels[:96]).mean())
    assert accuracy > 0.5, f"model accuracy on the first 96 test images is only {accuracy:.2f}"
