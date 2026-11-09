"""Shared pytest fixtures for the fruit detection project."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "Dataset" / "processed"
MANIFEST_PATH = PROJECT_ROOT / "Dataset" / "dataset_manifest.csv"
MODEL_PATH = PROJECT_ROOT / "Models" / "final_model" / "fruit_model.keras"


def _require(path: Path, hint: str) -> None:
    if not path.exists():
        pytest.skip(f"{path.name} missing — {hint}")


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def test_images() -> np.ndarray:
    _require(PROCESSED_DIR / "test_images.npy", "run `python -m SRC.data.prepare_dataset --stage cache`")
    return np.load(PROCESSED_DIR / "test_images.npy")


@pytest.fixture(scope="session")
def test_labels() -> np.ndarray:
    _require(PROCESSED_DIR / "test_labels.npy", "run `python -m SRC.data.prepare_dataset --stage cache`")
    return np.load(PROCESSED_DIR / "test_labels.npy")


@pytest.fixture(scope="session")
def sample_rgb(test_images: np.ndarray) -> np.ndarray:
    return np.asarray(test_images[0])


@pytest.fixture(scope="session")
def sample_png_bytes(sample_rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(sample_rgb).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="session")
def predictor():
    _require(MODEL_PATH, "run `python -m SRC.models.train`")
    from SRC.models.predictor import FruitPredictor

    return FruitPredictor()


@pytest.fixture(scope="session")
def flask_app():
    from Website.app import create_app

    app = create_app({"TESTING": True})
    return app


@pytest.fixture(scope="session")
def client(flask_app):
    return flask_app.test_client()
