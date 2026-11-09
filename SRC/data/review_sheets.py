"""Build visual review grids (montages) of the raw dataset.

The inspection script produces per-class contact sheets from the first files in
each folder; this utility samples images spread across the whole folder so a
human can verify content (whole fruit vs cross-section) across the dataset.

Usage:
    python -m SRC.data.review_sheets
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "Dataset" / "Original Image" / "Original Image"
DEFAULT_OUT = PROJECT_ROOT / "Dataset" / "interim" / "review_grids"


def montage(class_dirs: list[Path], out_path: Path, per_class: int, cell: int, label_w: int) -> None:
    rows, cols = len(class_dirs), per_class
    canvas = Image.new("RGB", (label_w + cols * cell, rows * cell), "white")
    draw = ImageDraw.Draw(canvas)
    for r, class_dir in enumerate(class_dirs):
        files = sorted(p for p in class_dir.iterdir() if p.is_file())
        step = max(1, len(files) // per_class)
        picks = [files[min(i * step, len(files) - 1)] for i in range(per_class)]
        draw.text((6, r * cell + cell // 2 - 6), class_dir.name, fill="black")
        for c, path in enumerate(picks):
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                img.thumbnail((cell - 6, cell - 6))
                x = label_w + c * cell + (cell - img.width) // 2
                y = r * cell + (cell - img.height) // 2
                canvas.paste(img, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=85)


def natural_key(path: Path) -> tuple:
    """Sort ``Name (12).jpg`` numerically so grid positions map to indices."""
    import re

    numbers = re.findall(r"\d+", path.stem)
    return (int(numbers[0]) if numbers else 0, path.stem)


def class_grid(class_dir: Path, out_path: Path, cols: int = 10, cell: int = 96) -> int:
    """Render every image of one class in reading order (numeric file order)."""
    files = sorted((p for p in class_dir.iterdir() if p.is_file()), key=natural_key)
    rows = (len(files) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell, rows * cell), "white")
    for idx, path in enumerate(files):
        try:
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                img.thumbnail((cell - 2, cell - 2))
                x = (idx % cols) * cell + (cell - img.width) // 2
                y = (idx // cols) * cell + (cell - img.height) // 2
                canvas.paste(img, (x, y))
        except Exception:  # unreadable images are reported by inspect_dataset
            continue
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=80)
    return len(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-class", type=int, default=6)
    parser.add_argument("--cell", type=int, default=150)
    parser.add_argument(
        "--mode", choices=("spread", "full", "both"), default="both",
        help="spread = 6 samples per class, full = every image of every class",
    )
    args = parser.parse_args(argv)

    class_dirs = sorted(p for p in args.data_dir.iterdir() if p.is_dir())
    if args.mode in ("spread", "both"):
        fresh = [p for p in class_dirs if p.name.startswith("Fresh")]
        rotten = [p for p in class_dirs if p.name.startswith("Rotten")]
        for name, group in (("grid_fresh.jpg", fresh), ("grid_rotten.jpg", rotten)):
            target = args.out_dir / name
            montage(group, target, args.per_class, args.cell, label_w=110)
            print(f"wrote {target.relative_to(PROJECT_ROOT)}")
    if args.mode in ("full", "both"):
        for class_dir in class_dirs:
            target = args.out_dir / "classes" / f"{class_dir.name}.jpg"
            count = class_grid(class_dir, target)
            print(f"wrote {target.relative_to(PROJECT_ROOT)} ({count} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
