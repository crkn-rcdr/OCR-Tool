#!/usr/bin/env python3
"""Save grid and content-box tiles for a source image.

Usage:
  python tools/save_tiles.py /path/to/image.jpg output/tiles_demo

This script uses the app's tiling logic in `services/tile_ocr.py` and
writes tile images and a JSON metadata file to the output folder.
"""

from pathlib import Path
import argparse
import json
import sys

# Ensure the repository root is on sys.path when this script is run
# directly (so `from services import tile_ocr` works when executing
# `python tools/save_tiles.py`).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from services import tile_ocr


def save_tiles_for_image(image_path: Path, outdir: Path, strategies=("grid", "content_boxes"), grid_rows=None, grid_cols=None, overlap_px=None, padding_px=None):
    image_path = image_path.resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    outdir.mkdir(parents=True, exist_ok=True)

    results = {}

    if "grid" in strategies:
        grid_dir = outdir / "grid"
        grid_dir.mkdir(parents=True, exist_ok=True)
        # Use provided values or defaults from tile_ocr
        rows = tile_ocr.DEFAULT_TILE_ROWS if grid_rows is None else int(grid_rows)
        cols = tile_ocr.DEFAULT_TILE_COLS if grid_cols is None else int(grid_cols)
        overlap = tile_ocr.DEFAULT_TILE_OVERLAP_PX if overlap_px is None else int(overlap_px)

        tiles = tile_ocr.build_image_tiles(
            image_path,
            grid_dir,
            grid_rows=rows,
            grid_cols=cols,
            overlap_px=overlap,
        )

        results["grid"] = [
            {
                "filename": str(tile.path.name),
                "path": str(tile.path.resolve()),
                "row": tile.row,
                "col": tile.col,
                "left": tile.left,
                "top": tile.top,
                "right": tile.right,
                "bottom": tile.bottom,
                "nominal_left": tile.nominal_left,
                "nominal_top": tile.nominal_top,
                "nominal_right": tile.nominal_right,
                "nominal_bottom": tile.nominal_bottom,
            }
            for tile in tiles
        ]

    if "content_boxes" in strategies:
        cb_dir = outdir / "content_boxes"
        cb_dir.mkdir(parents=True, exist_ok=True)
        padding = tile_ocr.DEFAULT_CONTENT_BOX_PADDING_PX if padding_px is None else int(padding_px)

        tiles = tile_ocr.build_content_box_tiles(
            image_path,
            cb_dir,
            padding_px=padding,
        )

        results["content_boxes"] = [
            {
                "filename": str(tile.path.name),
                "path": str(tile.path.resolve()),
                "index": tile.row,
                "left": tile.left,
                "top": tile.top,
                "right": tile.right,
                "bottom": tile.bottom,
                "nominal_left": tile.nominal_left,
                "nominal_top": tile.nominal_top,
                "nominal_right": tile.nominal_right,
                "nominal_bottom": tile.nominal_bottom,
            }
            for tile in tiles
        ]

    # Write metadata
    meta_path = outdir / f"{image_path.stem}_tiles.json"
    meta_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return outdir, meta_path


def parse_args():
    parser = argparse.ArgumentParser(description="Save tiles for an image using the repo's tiling logic.")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("outdir", nargs="?", default="output/tiles_demo", help="Output directory to save tiles")
    parser.add_argument("--strategies", nargs="+", choices=["grid", "content_boxes"], default=["grid", "content_boxes"], help="Which tiling strategies to run")
    parser.add_argument("--grid-rows", type=int, default=None)
    parser.add_argument("--grid-cols", type=int, default=None)
    parser.add_argument("--overlap-px", type=int, default=None)
    parser.add_argument("--padding-px", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        outdir, meta = save_tiles_for_image(
            Path(args.image),
            Path(args.outdir),
            strategies=args.strategies,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            overlap_px=args.overlap_px,
            padding_px=args.padding_px,
        )
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        return 2

    print(f"Tiles saved to: {outdir}")
    print(f"Metadata: {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
