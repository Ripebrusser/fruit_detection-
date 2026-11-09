"""Build the dataset manifest, group duplicates and create the train/val/test split.

Design notes
------------
* The raw dataset is never modified. Only derived files under ``Dataset/`` are written.
* Duplicate images (identical SHA-256, or perceptual-hash distance <= 4) are grouped
  with union-find and each *group* is assigned to exactly one split, so an image and
  its near copy can never straddle train/validation/test.
* Splits are allocated per class with 70/15/15 targets, so both fruit identity and
  condition stay balanced across splits.
* Images are decoded once, resized to 224x224 with PIL and cached as uint8 ``.npy``
  arrays. Inference uses the same :func:`SRC.utils.imaging.preprocess_for_model`
  function, so training and serving preprocessing are byte-identical.

Usage:
    python -m SRC.data.prepare_dataset
    python -m SRC.data.prepare_dataset --phash-distance 4 --force
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from SRC.utils.imaging import preprocess_for_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "Dataset" / "Original Image" / "Original Image"
PROCESSED_DIR = PROJECT_ROOT / "Dataset" / "processed"
MANIFEST_PATH = PROJECT_ROOT / "Dataset" / "dataset_manifest.csv"
CLASS_NAMES_PATH = PROJECT_ROOT / "Models" / "class_names.json"

FRUITS = ["Apple", "Banana", "Orange", "Grape", "Guava", "Jujube", "Pomegranate", "Strawberry"]
CLASS_NAMES = [f"{cond}{fruit}" for fruit in FRUITS for cond in ("Fresh", "Rotten")]
SPLIT_TARGETS = {"train": 0.70, "val": 0.15, "test": 0.15}
IMAGE_SIZE = 224
SEED = 42


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def phash_of(path: Path) -> str:
    with Image.open(path) as img:
        img.draft("RGB", (256, 256))
        rgb = ImageOps.exif_transpose(img).convert("RGB")
        return str(imagehash.phash(rgb.resize((32, 32))))


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_inventory(data_dir: Path) -> pd.DataFrame:
    records = []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if class_dir.name not in CLASS_NAMES:
            print(f"  ! skipping unexpected folder: {class_dir.name}", file=sys.stderr)
            continue
        for path in sorted(p for p in class_dir.iterdir() if p.is_file()):
            with Image.open(path) as img:
                width, height = img.size
            records.append(
                {
                    "relative_path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "absolute_path": str(path),
                    "file_name": path.name,
                    "class_name": class_dir.name,
                    "condition": "fresh" if class_dir.name.startswith("Fresh") else "rotten",
                    "fruit": class_dir.name.replace("Fresh", "").replace("Rotten", ""),
                    "width": width,
                    "height": height,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_of(path),
                    "phash": phash_of(path),
                }
            )
    return pd.DataFrame.from_records(records)


def group_duplicates(df: pd.DataFrame, phash_distance: int) -> pd.Series:
    """Return a duplicate-group id per row (rows with the same id share content)."""
    uf = UnionFind(len(df))
    by_hash: dict[str, int] = {}
    for idx, sha in enumerate(df["sha256"]):
        if sha in by_hash:
            uf.union(by_hash[sha], idx)
        else:
            by_hash[sha] = idx

    hashes = np.array(
        [[int(c) for c in bin(int(h, 16))[2:].zfill(64)] for h in df["phash"]], dtype=np.uint8
    )
    distances = (hashes[:, None, :] != hashes[None, :, :]).sum(axis=2)
    # Only the strict upper triangle is a set of unique pairs. Masking with
    # np.triu(..., k=1) alone would leave zeros in the lower triangle (which
    # compare as "distance 0"), so an explicit boolean pair mask is required.
    pair_mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
    rows, cols = np.where((distances <= phash_distance) & pair_mask)
    for i, j in zip(rows, cols):
        uf.union(int(i), int(j))

    roots = [uf.find(i) for i in range(len(df))]
    return pd.Series(roots, index=df.index, name="dup_group")


def assign_splits(df: pd.DataFrame, seed: int) -> pd.Series:
    """Group-aware stratified 70/15/15 assignment, done independently per class."""
    rng = random.Random(seed)
    splits = pd.Series(index=df.index, dtype=object)

    for class_name, class_df in df.groupby("class_name", sort=True):
        deficit = {s: SPLIT_TARGETS[s] * len(class_df) for s in SPLIT_TARGETS}
        groups = [
            list(idx_list)
            for idx_list in class_df.groupby("dup_group").groups.values()
        ]
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)  # place big groups first, into emptiest split

        for members in groups:
            # Pick the split with the largest remaining deficit relative to target.
            split = max(deficit, key=lambda s: deficit[s] / SPLIT_TARGETS[s])
            for idx in members:
                splits.loc[idx] = split
            deficit[split] -= len(members)
    return splits


def split_frame(split: str) -> pd.DataFrame:
    """Rows of the manifest belonging to ``split``, in the exact order used for caching."""
    df = pd.read_csv(MANIFEST_PATH)
    df["absolute_path"] = df["relative_path"].map(lambda p: str(PROJECT_ROOT / p))
    return (
        df[df["split"] == split]
        .sort_values(["class_name", "file_name"])
        .reset_index(drop=True)
    )


def _load_and_resize(path: str) -> np.ndarray:
    """Decode one JPEG and return the model-ready 224x224 uint8 array."""
    with Image.open(path) as img:
        rgb = ImageOps.exif_transpose(img).convert("RGB")
        return preprocess_for_model(np.asarray(rgb), IMAGE_SIZE).astype(np.uint8)


def cache_split(df: pd.DataFrame, split: str, workers: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Decode and resize one split in parallel; PIL releases the GIL while decoding."""
    rows = df[df["split"] == split].sort_values(["class_name", "file_name"]).reset_index(drop=True)
    class_index = {name: i for i, name in enumerate(CLASS_NAMES)}
    assert len(rows) > 0, f"no rows for split {split}"
    labels = np.array([class_index[name] for name in rows["class_name"]], dtype=np.int32)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        arrays = list(pool.map(_load_and_resize, rows["absolute_path"].tolist()))
    images = np.stack(arrays).astype(np.uint8)
    return images, labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--phash-distance", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true", help="rebuild cached arrays")
    parser.add_argument(
        "--stage", choices=("all", "manifest", "cache", "metadata"), default="all",
        help="manifest = inventory+split only; cache = arrays from the existing manifest; "
             "metadata = rewrite class_names.json from the existing manifest",
    )
    args = parser.parse_args(argv)

    if args.stage == "cache":
        return cache_stage(args)
    if args.stage == "metadata":
        return metadata_stage(args)


    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        print(f"ERROR: dataset not found at {data_dir}", file=sys.stderr)
        return 2

    started = time.time()
    print(f"[1/5] inventorying {data_dir} ...")
    df = build_inventory(data_dir)
    if df.empty:
        print("ERROR: no images found", file=sys.stderr)
        return 2
    print(f"      {len(df)} images across {df['class_name'].nunique()} classes")

    print(f"[2/5] grouping duplicates (sha256 + phash <= {args.phash_distance}) ...")
    df["dup_group"] = group_duplicates(df, args.phash_distance)
    group_sizes = df.groupby("dup_group").size()
    df["group_size"] = df["dup_group"].map(group_sizes)
    df["is_duplicate_copy"] = df["group_size"] > 1
    print(f"      {int((df['group_size'] > 1).sum())} images belong to duplicate groups "
          f"({int((group_sizes > 1).sum())} groups)")

    print("[3/5] assigning splits ...")
    df["split"] = assign_splits(df, args.seed)
    if df["split"].isna().any():
        print("ERROR: some images were not assigned a split", file=sys.stderr)
        return 3

    leaks = df.groupby("dup_group")["split"].nunique()
    leak_groups = int((leaks > 1).sum())
    print(f"      duplicate groups spanning splits: {leak_groups} (must be 0)")

    print("[4/5] writing manifest ...")
    manifest = df.drop(columns=["absolute_path"]).sort_values(["split", "class_name", "file_name"])
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)

    counts = pd.crosstab(df["class_name"], df["split"])
    for split in ("train", "val", "test"):
        if split not in counts:
            counts[split] = 0
    print(counts[["train", "val", "test"]].to_string())
    print("totals:", {s: int((df['split'] == s).sum()) for s in ('train', 'val', 'test')})

    print("[5/5] caching 224x224 uint8 arrays ...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        images_path = PROCESSED_DIR / f"{split}_images.npy"
        labels_path = PROCESSED_DIR / f"{split}_labels.npy"
        if images_path.exists() and labels_path.exists() and not args.force:
            print(f"      {split}: cached arrays already exist (use --force to rebuild)")
            continue
        images, labels = cache_split(df, split)
        np.save(images_path, images)
        np.save(labels_path, labels)
        print(f"      {split}: {images.shape} -> {images_path.relative_to(PROJECT_ROOT)}")

    class_names_path = write_class_metadata(df, args.seed, args.phash_distance)
    print(f"      class mapping -> {class_names_path.relative_to(PROJECT_ROOT)}")
    print(f"      manifest -> {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"done in {time.time() - started:.1f}s")
    if leak_groups:
        print("WARNING: duplicate leakage detected", file=sys.stderr)
        return 4
    return 0


def write_class_metadata(df: pd.DataFrame, seed: int, phash_distance: int) -> Path:
    """Write ``Models/class_names.json`` from the manifest. Idempotent and cheap."""
    group_sizes = df.groupby("dup_group").size()
    group_size_column = df["dup_group"].map(group_sizes)
    CLASS_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLASS_NAMES_PATH.write_text(
        json.dumps(
            {
                "class_names": CLASS_NAMES,
                "fruits": FRUITS,
                "conditions": ["fresh", "rotten"],
                "input_size": IMAGE_SIZE,
                "index_to_class": {str(i): name for i, name in enumerate(CLASS_NAMES)},
                "class_to_index": {name: i for i, name in enumerate(CLASS_NAMES)},
                "preprocessing": "PIL BILINEAR resize to 224x224, RGB uint8 0-255; "
                                 "MobileNetV2 preprocess_input is applied inside the model",
                "split_counts": {
                    s: dict(Counter(df.loc[df["split"] == s, "class_name"]))
                    for s in ("train", "val", "test")
                },
                "duplicate_groups": {
                    "method": f"sha256 exact + phash hamming <= {phash_distance} (union-find)",
                    "groups": int((group_sizes > 1).sum()),
                    "images_in_groups": int((group_size_column > 1).sum()),
                },
                "seed": seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return CLASS_NAMES_PATH


def metadata_stage(args) -> int:
    """Rewrite class_names.json from the existing manifest without re-hashing the dataset."""
    if not MANIFEST_PATH.exists():
        print(f"ERROR: manifest missing at {MANIFEST_PATH}; run --stage manifest", file=sys.stderr)
        return 2
    df = pd.read_csv(MANIFEST_PATH)
    path = write_class_metadata(df, args.seed, args.phash_distance)
    print(f"class mapping -> {path.relative_to(PROJECT_ROOT)}")
    return 0


def cache_stage(args) -> int:
    """Build the cached .npy arrays from an existing manifest (resumable step)."""
    if not MANIFEST_PATH.exists():
        print(f"ERROR: manifest missing at {MANIFEST_PATH}; run with --stage manifest", file=sys.stderr)
        return 2
    df = pd.read_csv(MANIFEST_PATH)
    df["absolute_path"] = df["relative_path"].map(lambda p: str(PROJECT_ROOT / p))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    total = time.time()
    for split in ("train", "val", "test"):
        images_path = PROCESSED_DIR / f"{split}_images.npy"
        labels_path = PROCESSED_DIR / f"{split}_labels.npy"
        if images_path.exists() and labels_path.exists() and not args.force:
            print(f"{split}: cached arrays already exist (use --force to rebuild)")
            continue
        started = time.time()
        images, labels = cache_split(df, split)
        np.save(images_path, images)
        np.save(labels_path, labels)
        print(f"{split}: {images.shape} in {time.time() - started:.1f}s -> {images_path.name}")
    print(f"cache stage finished in {time.time() - total:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
