"""Shared image utilities: background estimation, fruit segmentation, colour stats.

These helpers are used by dataset inspection, preprocessing and inference so that
every stage looks at an image the same way.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_rgb(path, size: tuple[int, int] | None = None) -> np.ndarray:
    """Load an image as an RGB uint8 array, applying EXIF orientation."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        if size is not None:
            img = img.resize(size, Image.BILINEAR)
        return np.asarray(img)


def background_colour(rgb: np.ndarray, border: int = 8) -> np.ndarray:
    """Median colour of the image border, used as the background estimate."""
    strips = [
        rgb[:border, :, :].reshape(-1, 3),
        rgb[-border:, :, :].reshape(-1, 3),
        rgb[:, :border, :].reshape(-1, 3),
        rgb[:, -border:, :].reshape(-1, 3),
    ]
    return np.median(np.concatenate(strips, axis=0), axis=0)


def fruit_mask(rgb: np.ndarray, tolerance: float = 38.0) -> np.ndarray:
    """Boolean mask of pixels that differ from the estimated background.

    A pixel belongs to the fruit when its Euclidean RGB distance from the border
    median exceeds ``tolerance``. The largest connected component is kept so that
    stray specks and shadows are ignored.
    """
    bg = background_colour(rgb)
    distance = np.linalg.norm(rgb.astype(np.float32) - bg, axis=2)
    mask = distance > tolerance
    try:
        import cv2

        mask_u8 = (mask.astype(np.uint8)) * 255
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if count > 1:
            # Rank components by area (label 0 is the background component).
            areas = stats[1:, cv2.CC_STAT_AREA]
            keep = 1 + int(np.argmax(areas))
            mask = labels == keep
            # Keep secondary components larger than 15% of the main one (multi-fruit shots).
            secondary = [
                idx for idx in range(1, count)
                if idx != keep and stats[idx, cv2.CC_STAT_AREA] >= 0.15 * stats[keep, cv2.CC_STAT_AREA]
            ]
            if secondary:
                mask = np.isin(labels, [keep, *secondary])
    except Exception:  # pragma: no cover - OpenCV optional at inspection time
        pass
    return mask


def fruit_colour_stats(rgb: np.ndarray) -> dict:
    """Colour and composition statistics computed inside the fruit mask."""
    mask = fruit_mask(rgb)
    if mask.sum() < 64:  # fall back to the whole frame for odd images
        mask = np.ones(rgb.shape[:2], dtype=bool)
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.float32)
    h, s, v = hsv[..., 0][mask], hsv[..., 1][mask], hsv[..., 2][mask]
    selected = rgb[mask].astype(np.float32)
    return {
        "mask_fraction": float(mask.mean()),
        "mean_hue": float(h.mean()),
        "mean_saturation": float(s.mean()),
        "mean_value": float(v.mean()),
        "mean_r": float(selected[:, 0].mean()),
        "mean_g": float(selected[:, 1].mean()),
        "mean_b": float(selected[:, 2].mean()),
        # Pale interior flesh: bright and only lightly pigmented.
        "pale_fraction": float((((v > 110) & (s < 95)) | ((v > 175) & (s < 130))).mean()),
        # Deeply pigmented / sunken skin.
        "dark_fraction": float((v < 80).mean()),
        "saturated_fraction": float((s > 130).mean()),
    }


def quality_checks(rgb: np.ndarray) -> dict:
    """Basic capture-quality measurements used to warn about unusable uploads.

    ``blur_variance`` is the variance of the Laplacian: low values mean a soft, blurry
    image. The warning threshold itself is calibrated on the test split and stored in
    ``Models/ood_config.json``.
    """
    import cv2

    height, width = rgb.shape[:2]
    gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.float32)
    return {
        "width": int(width),
        "height": int(height),
        "brightness": float(gray.mean()),
        "blur_variance": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
        "mask_fraction": float(fruit_mask(rgb).mean()),
    }


def preprocess_for_model(rgb: np.ndarray, size: int = 224) -> np.ndarray:
    """Deterministic inference preprocessing — must match training exactly.

    Training uses ``tf.keras.utils.image_dataset_from_directory`` without a
    rescaling layer; MobileNetV2's ``preprocess_input`` therefore lives inside the
    saved model. Here we only resize and convert to float32 in 0-255 range.
    """
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"preprocess_for_model expects an HxWx3 RGB array, got shape {array.shape}")
    img = Image.fromarray(array.astype(np.uint8)).resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)
