"""Preprocessing, segmentation and image-quality helpers."""
from __future__ import annotations

import numpy as np
import pytest

from SRC.utils.imaging import (
    background_colour,
    fruit_mask,
    preprocess_for_model,
    quality_checks,
)


def test_preprocess_shape_dtype_and_range(sample_rgb: np.ndarray) -> None:
    out = preprocess_for_model(sample_rgb, 224)
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 255.0


def test_preprocess_is_deterministic(sample_rgb: np.ndarray) -> None:
    first = preprocess_for_model(sample_rgb, 224)
    second = preprocess_for_model(sample_rgb, 224)
    assert np.array_equal(first, second)


def test_preprocess_resizes_any_resolution() -> None:
    for size in ((640, 480), (300, 900), (224, 224)):
        image = np.random.default_rng(0).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        out = preprocess_for_model(image, 224)
        assert out.shape == (224, 224, 3)


def test_non_rgb_input_is_rejected_by_preprocess() -> None:
    grayscale = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError):
        preprocess_for_model(grayscale, 224)


def test_background_and_mask_on_synthetic_image() -> None:
    image = np.full((200, 200, 3), 250, dtype=np.uint8)
    image[70:130, 70:130] = (40, 120, 40)  # a dark green "fruit" on white
    bg = background_colour(image)
    assert np.all(bg > 240)
    mask = fruit_mask(image)
    assert mask.dtype == bool
    assert 0.05 < mask.mean() < 0.4
    assert mask[100, 100]


def test_mask_handles_uniform_image() -> None:
    flat = np.full((120, 120, 3), 128, dtype=np.uint8)
    assert fruit_mask(flat).sum() == 0


def test_quality_checks_keys_and_types(sample_rgb: np.ndarray) -> None:
    quality = quality_checks(sample_rgb)
    assert set(quality) == {"width", "height", "brightness", "blur_variance", "mask_fraction"}
    assert all(isinstance(value, float) for key, value in quality.items() if key not in ("width", "height"))
    assert 0.0 <= quality["brightness"] <= 255.0
    assert quality["blur_variance"] >= 0.0


def test_dark_image_has_low_brightness() -> None:
    dark = np.full((224, 224, 3), 5, dtype=np.uint8)
    assert quality_checks(dark)["brightness"] < 10


def test_training_and_inference_preprocessing_agree() -> None:
    """The cached training arrays must equal what inference produces."""
    from Tests.conftest import PROCESSED_DIR, PROJECT_ROOT
    import pandas as pd
    from PIL import Image, ImageOps

    cached_path = PROCESSED_DIR / "train_images.npy"
    manifest_path = PROJECT_ROOT / "Dataset" / "dataset_manifest.csv"
    if not cached_path.exists():
        pytest.skip("cached arrays missing")
    cached = np.load(cached_path, mmap_mode="r")
    frame = (
        pd.read_csv(manifest_path)
        .query("split == 'train'")
        .sort_values(["class_name", "file_name"])
        .reset_index(drop=True)
    )
    row = frame.iloc[7]
    with Image.open(PROJECT_ROOT / row["relative_path"]) as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    recomputed = preprocess_for_model(rgb, 224).astype(np.uint8)
    assert np.array_equal(recomputed, np.asarray(cached[7]))
