"""Dataset loading, manifest integrity and split/eakage checks."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from Tests.conftest import MANIFEST_PATH, PROCESSED_DIR, PROJECT_ROOT

CLASS_NAMES_PATH = PROJECT_ROOT / "Models" / "class_names.json"

pytestmark = pytest.mark.skipif(
    not MANIFEST_PATH.exists(), reason="dataset manifest missing — run SRC.data.prepare_dataset"
)


@pytest.fixture(scope="module")
def manifest() -> pd.DataFrame:
    return pd.read_csv(MANIFEST_PATH)


def test_manifest_has_all_images(manifest: pd.DataFrame) -> None:
    assert len(manifest) == 3200
    assert manifest["relative_path"].is_unique
    assert set(manifest["split"]) == {"train", "val", "test"}


def test_manifest_columns_present(manifest: pd.DataFrame) -> None:
    required = {
        "relative_path", "class_name", "fruit", "condition", "split",
        "sha256", "phash", "dup_group", "width", "height",
    }
    assert required.issubset(manifest.columns)


def test_sixteen_balanced_classes(manifest: pd.DataFrame) -> None:
    counts = manifest["class_name"].value_counts()
    assert len(counts) == 16
    assert (counts == 200).all(), counts.to_dict()


def test_split_sizes(manifest: pd.DataFrame) -> None:
    assert manifest["split"].value_counts().to_dict() == {"train": 2240, "val": 480, "test": 480}


def test_each_class_split_is_stratified(manifest: pd.DataFrame) -> None:
    per_class = pd.crosstab(manifest["class_name"], manifest["split"])
    for column in ("train", "val", "test"):
        assert (per_class[column] == {"train": 140, "val": 30, "test": 30}[column]).all()


def test_no_duplicate_group_crosses_splits(manifest: pd.DataFrame) -> None:
    spanning = manifest.groupby("dup_group")["split"].nunique()
    assert spanning.max() == 1, "a duplicate group appears in more than one split"


def test_no_identical_file_in_two_splits(manifest: pd.DataFrame) -> None:
    per_hash = manifest.groupby("sha256")["split"].nunique()
    assert per_hash.max() == 1


def test_files_exist_and_are_readable(manifest: pd.DataFrame) -> None:
    sample = manifest.sample(25, random_state=0)
    for relative in sample["relative_path"]:
        path = PROJECT_ROOT / relative
        assert path.is_file(), relative
        assert path.stat().st_size > 1024


def test_class_names_mapping_matches_manifest(manifest: pd.DataFrame) -> None:
    meta = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    assert meta["class_names"] == [
        "FreshApple", "RottenApple", "FreshBanana", "RottenBanana",
        "FreshOrange", "RottenOrange", "FreshGrape", "RottenGrape",
        "FreshGuava", "RottenGuava", "FreshJujube", "RottenJujube",
        "FreshPomegranate", "RottenPomegranate", "FreshStrawberry", "RottenStrawberry",
    ]
    assert set(manifest["class_name"]) == set(meta["class_names"])
    assert meta["class_to_index"]["FreshApple"] == 0
    assert meta["class_to_index"]["RottenStrawberry"] == 15
    # Fruit id must be idx // 2 for the aggregation used by the website.
    for name, index in meta["class_to_index"].items():
        fruit = name.replace("Fresh", "").replace("Rotten", "")
        assert meta["fruits"][index // 2] == fruit
        assert index % 2 == (0 if name.startswith("Fresh") else 1)


def test_cached_arrays_match_manifest() -> None:
    if not (PROCESSED_DIR / "train_images.npy").exists():
        pytest.skip("cached arrays missing")
    import numpy as np

    for split, expected in (("train", 2240), ("val", 480), ("test", 480)):
        images = np.load(PROCESSED_DIR / f"{split}_images.npy", mmap_mode="r")
        labels = np.load(PROCESSED_DIR / f"{split}_labels.npy")
        assert images.shape == (expected, 224, 224, 3), images.shape
        assert labels.shape == (expected,)
        assert labels.min() == 0 and labels.max() == 15
        # Labels must be ordered exactly like the manifest rows for that split.
        frame = (
            pd.read_csv(MANIFEST_PATH)
            .query("split == @split")
            .sort_values(["class_name", "file_name"])
            .reset_index(drop=True)
        )
        meta = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
        expected_labels = [meta["class_to_index"][name] for name in frame["class_name"]]
        assert list(labels) == expected_labels
