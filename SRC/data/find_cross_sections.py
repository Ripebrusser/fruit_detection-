"""Flag candidate cross-section (cut fruit) images for manual review.

The dataset ships no metadata saying whether a photograph shows a whole fruit or a
cut fruit, so this module scores every image and renders labelled candidate
montages. A human then confirms which candidates really are cross-sections; the
confirmed paths are stored in ``Dataset/interim/cross_section_labels.csv`` and are
the only cross-section information the rest of the pipeline trusts.

Scoring uses the background-masked fruit region (see ``SRC.utils.imaging``) so a
white tabletop is never mistaken for pale fruit flesh.

Usage:
    python -m SRC.data.find_cross_sections                # score + montages
    python -m SRC.data.find_cross_sections --top 16
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from SRC.utils.imaging import fruit_colour_stats, load_rgb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "Dataset" / "Original Image" / "Original Image"
DEFAULT_OUT = PROJECT_ROOT / "Dataset" / "interim"
SCORES_CSV = "fruit_colour_scores.csv"

SCORE_FIELDS = [
    "relative_path", "class_folder", "mask_fraction", "pale_fraction", "dark_fraction",
    "saturated_fraction", "mean_hue", "mean_saturation", "mean_value",
    "mean_r", "mean_g", "mean_b",
]


def montage(entries: list[tuple[str, Path]], out_path: Path, cols: int = 8, cell: int = 132) -> None:
    rows = (len(entries) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell, rows * cell), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, path) in enumerate(entries):
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((cell - 4, cell - 4))
            canvas.paste(img, ((idx % cols) * cell + 2, (idx // cols) * cell + 2))
        draw.text(((idx % cols) * cell + 4, (idx // cols) * cell + cell - 12), label, fill="red")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=82)


def short_label(relative_path: str) -> str:
    """``.../FreshApple/FreshApple (34).jpg`` -> ``FreshApple#34``."""
    stem = Path(relative_path).stem
    if "(" in stem and stem.endswith(")"):
        name, number = stem[:-1].split("(")
        return f"{name.strip()}#{number}"
    return stem


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top", type=int, default=16, help="candidates per class in the montage")
    args = parser.parse_args(argv)

    rows: list[dict] = []
    for class_dir in sorted(p for p in args.data_dir.iterdir() if p.is_dir()):
        for path in sorted(p for p in class_dir.iterdir() if p.is_file()):
            rgb = load_rgb(path, size=(224, 224))
            rows.append(
                {
                    "relative_path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "class_folder": class_dir.name,
                    **fruit_colour_stats(rgb),
                }
            )

    csv_path = args.out_dir / SCORES_CSV
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    by_class: dict[str, list[dict]] = {}
    for row in rows:
        by_class.setdefault(row["class_folder"], []).append(row)

    for group_name, prefix in (("candidates_fresh.jpg", "Fresh"), ("candidates_rotten.jpg", "Rotten")):
        entries: list[tuple[str, Path]] = []
        for class_name in sorted(k for k in by_class if k.startswith(prefix)):
            ranked = sorted(by_class[class_name], key=lambda r: r["pale_fraction"], reverse=True)[: args.top]
            for r in ranked:
                entries.append((short_label(r["relative_path"]), PROJECT_ROOT / r["relative_path"]))
        montage(entries, args.out_dir / "review_grids" / group_name)
        print(f"wrote review_grids/{group_name} ({len(entries)} candidates)")

    ranked_all = sorted(rows, key=lambda r: r["pale_fraction"], reverse=True)
    print("pale_fraction percentiles:",
          {p: round(ranked_all[int(len(ranked_all) * p / 100)]["pale_fraction"], 3) for p in (50, 90, 95, 99)})
    print(f"scores -> {csv_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
