"""Dataset inspection utility.

Scans the raw fruit dataset, validates every image, computes file hashes and
perceptual hashes, looks for exact/near duplicates and suspicious files, and
writes a machine-readable summary plus contact sheets for visual review.

Usage:
    python -m SRC.data.inspect_dataset
    python -m SRC.data.inspect_dataset --data-dir "Dataset/Original Image/Original Image"
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "Dataset" / "Original Image" / "Original Image"
DEFAULT_JSON = PROJECT_ROOT / "Dataset" / "interim" / "inspection_summary.json"
DEFAULT_CSV = PROJECT_ROOT / "Dataset" / "interim" / "image_inventory.csv"
DEFAULT_SHEETS = PROJECT_ROOT / "Dataset" / "interim" / "contact_sheets"

EXIF_TAGS = {271: "make", 272: "model", 306: "datetime", 274: "orientation"}


def parse_class_name(folder: str) -> tuple[str, str]:
    """Split a folder name such as ``FreshApple`` into (condition, fruit)."""
    for condition in ("Fresh", "Rotten"):
        if folder.startswith(condition):
            return condition.lower(), folder[len(condition):]
    raise ValueError(f"Unrecognised class folder name: {folder}")


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def inspect_image(path: Path) -> dict:
    """Return per-image metadata. Never raises: failures are reported in-band."""
    record = {
        "relative_path": None,
        "file_name": path.name,
        "class_folder": path.parent.name,
        "size_bytes": path.stat().st_size,
        "readable": False,
        "error": None,
        "width": None,
        "height": None,
        "mode": None,
        "format": None,
        "sha256": None,
        "phash": None,
        "mean_brightness": None,
        "std_brightness": None,
        "mean_saturation": None,
        "exif_make": None,
        "exif_model": None,
        "exif_datetime": None,
    }
    try:
        record["sha256"] = sha256_of(path)
    except OSError as exc:  # pragma: no cover - filesystem edge case
        record["error"] = f"hash failed: {exc}"
        return record

    try:
        with Image.open(path) as img:
            record["format"] = img.format
            record["mode"] = img.mode
            record["width"], record["height"] = img.size  # original pixels, before any draft
            img.draft("RGB", (256, 256))  # fast JPEG DCT downscale for statistics
            rgb = ImageOps.exif_transpose(img).convert("RGB")
            rgb.thumbnail((256, 256))
            arr = np.asarray(rgb, dtype=np.float32)
            hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32)
            record["mean_brightness"] = float(arr.mean())
            record["std_brightness"] = float(arr.std())
            record["mean_saturation"] = float(hsv[..., 1].mean())
            record["phash"] = str(imagehash.phash(rgb.resize((32, 32))))
            exif = img.getexif()
            for tag, key in EXIF_TAGS.items():
                value = exif.get(tag)
                if value not in (None, ""):
                    record[f"exif_{key}"] = str(value)[:120]
            record["readable"] = True
    except Exception as exc:  # corrupt / truncated / unsupported file
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def find_duplicates(records: list[dict], phash_distance: int = 8) -> dict:
    """Group exact duplicates (same sha256) and near duplicates (phash distance)."""
    exact: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        if rec["sha256"]:
            exact[rec["sha256"]].append(rec["relative_path"])
    exact_groups = [sorted(paths) for paths in exact.values() if len(paths) > 1]

    hashed = [r for r in records if r["phash"]]
    near_groups: list[dict] = []
    if hashed:
        bits = np.array(
            [[int(c) for c in bin(int(r["phash"], 16))[2:].zfill(64)] for r in hashed],
            dtype=np.uint8,
        )
        distance = (bits[:, None, :] != bits[None, :, :]).sum(axis=2)
        upper = np.triu_indices(len(hashed), k=1)
        pairs = [(i, j) for i, j in zip(*upper) if 0 < distance[i, j] <= phash_distance]
        # Union-find to build near-duplicate clusters.
        parent = list(range(len(hashed)))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i, j in pairs:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri
        clusters: dict[int, list[int]] = defaultdict(list)
        for idx in range(len(hashed)):
            clusters[find(idx)].append(idx)
        for members in clusters.values():
            if len(members) < 2:
                continue
            groups = sorted(
                {
                    tuple(
                        sorted({
                            hashed[m]["sha256"],
                            hashed[m]["sha256"],
                        })
                    )[0]
                    for m in members
                }
            )
            classes = sorted({hashed[m]["class_folder"] for m in members})
            near_groups.append(
                {
                    "n_images": len(members),
                    "distance_threshold": phash_distance,
                    "class_folders": classes,
                    "cross_class": len(classes) > 1,
                    "paths": [hashed[m]["relative_path"] for m in members],
                    "sha256": list(groups),
                }
            )
    return {"exact_groups": exact_groups, "near_groups": near_groups}


def build_contact_sheets(records: list[dict], out_dir: Path, per_class: int = 8, cols: int = 4) -> list[str]:
    """Create one thumbnail grid per class so a human can verify image content."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        if rec["readable"]:
            by_class[rec["class_folder"]].append(rec)
    written = []
    thumb = 220
    for class_name, recs in sorted(by_class.items()):
        picks = recs[:per_class]
        rows = (len(picks) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb, rows * thumb), "white")
        for idx, rec in enumerate(picks):
            path = PROJECT_ROOT / rec["relative_path"]
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                img.thumbnail((thumb, thumb))
                sheet.paste(img, ((idx % cols) * thumb, (idx // cols) * thumb))
        target = out_dir / f"{class_name}.jpg"
        sheet.save(target, quality=88)
        written.append(str(target.relative_to(PROJECT_ROOT)))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--sheets-dir", type=Path, default=DEFAULT_SHEETS)
    parser.add_argument("--no-sheets", action="store_true")
    parser.add_argument("--phash-distance", type=int, default=8)
    args = parser.parse_args(argv)

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        print(f"ERROR: data directory not found: {data_dir}", file=sys.stderr)
        return 2

    class_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if not class_dirs:
        print(f"ERROR: no class folders inside {data_dir}", file=sys.stderr)
        return 2

    extensions = Counter()
    records: list[dict] = []
    total = 0
    for class_dir in class_dirs:
        files = sorted(p for p in class_dir.iterdir() if p.is_file())
        total += len(files)
        for path in files:
            extensions[path.suffix.lower()] += 1
            rec = inspect_image(path)
            rec["relative_path"] = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            try:
                condition, fruit = parse_class_name(class_dir.name)
                rec["condition"], rec["fruit"] = condition, fruit
            except ValueError as exc:
                rec["condition"], rec["fruit"], rec["error"] = None, None, str(exc)
            records.append(rec)

    summary = {
        "data_dir": str(data_dir),
        "n_class_folders": len(class_dirs),
        "n_files_total": total,
        "extensions": dict(extensions),
        "n_readable": sum(r["readable"] for r in records),
        "n_unreadable": sum(not r["readable"] for r in records),
        "unreadable": [
            {"path": r["relative_path"], "error": r["error"]} for r in records if not r["readable"]
        ],
        "per_class_counts": dict(sorted(Counter(r["class_folder"] for r in records).items())),
        "fruit_counts": dict(sorted(Counter(r.get("fruit") for r in records).items())),
        "condition_counts": dict(sorted(Counter(r.get("condition") for r in records).items())),
        "image_size_stats": {},
        "exif": {
            "with_make": sum(1 for r in records if r["exif_make"]),
            "with_datetime": sum(1 for r in records if r["exif_datetime"]),
            "sample_makes": Counter(r["exif_make"] for r in records if r["exif_make"]).most_common(5),
        },
    }

    sizes = [(r["width"], r["height"]) for r in records if r["readable"] and r["width"]]
    if sizes:
        widths = np.array([w for w, _ in sizes])
        heights = np.array([h for _, h in sizes])
        summary["image_size_stats"] = {
            "min": [int(widths.min()), int(heights.min())],
            "max": [int(widths.max()), int(heights.max())],
            "median": [int(np.median(widths)), int(np.median(heights))],
            "below_224_any_side": int(((widths < 224) | (heights < 224)).sum()),
        }

    summary["duplicates"] = find_duplicates(records, args.phash_distance)
    summary["duplicates"]["n_exact_groups"] = len(summary["duplicates"]["exact_groups"])
    summary["duplicates"]["n_exact_duplicate_files"] = sum(
        len(g) - 1 for g in summary["duplicates"]["exact_groups"]
    )
    summary["duplicates"]["n_near_groups"] = len(summary["duplicates"]["near_groups"])
    summary["duplicates"]["n_near_groups_cross_class"] = sum(
        1 for g in summary["duplicates"]["near_groups"] if g["cross_class"]
    )

    quality = [r for r in records if r["readable"]]
    if quality:
        bright = np.array([r["mean_brightness"] for r in quality])
        summary["brightness"] = {
            "mean": float(bright.mean()),
            "very_dark_images": int((bright < 40).sum()),
            "very_bright_images": int((bright > 220).sum()),
        }

    if not args.no_sheets:
        summary["contact_sheets"] = build_contact_sheets(records, args.sheets_dir.resolve())

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "relative_path", "class_folder", "fruit", "condition", "size_bytes", "readable",
        "width", "height", "mode", "format", "sha256", "phash", "mean_brightness",
        "std_brightness", "mean_saturation", "exif_make", "exif_model", "exif_datetime", "error",
    ]
    with args.csv_out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    print(json.dumps({k: summary[k] for k in (
        "n_files_total", "n_readable", "n_unreadable", "n_class_folders",
        "extensions", "per_class_counts", "image_size_stats",
    )}, indent=2))
    print(f"unreadable: {summary['n_unreadable']}")
    print(f"exact duplicate groups: {summary['duplicates']['n_exact_groups']}")
    print(f"near duplicate groups (phash<={args.phash_distance}): {summary['duplicates']['n_near_groups']}")
    print(f"near duplicate groups spanning classes: {summary['duplicates']['n_near_groups_cross_class']}")
    print(f"manifest -> {args.csv_out}")
    print(f"summary  -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
