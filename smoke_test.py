import argparse
from pathlib import Path

from services.ocr_defaults import (
    DEFAULT_DOCLAYOUT_PADDING_PX,
    DEFAULT_IGNORE_BLACK_BORDER,
    DEFAULT_PREPROCESSING_PRESET,
    DEFAULT_DOCLAYOUT_THRESHOLD,
    DEFAULT_TEXT_DET_BOX_THRESH,
    DEFAULT_TEXT_DET_LIMIT_SIDE_LEN,
    DEFAULT_TEXT_DET_LIMIT_TYPE,
    DEFAULT_TEXT_DET_MAX_SIDE_LIMIT,
    DEFAULT_TEXT_DET_THRESH,
    DEFAULT_TEXT_DET_UNCLIP_RATIO,
    DEFAULT_TEXT_REC_SCORE_THRESH,
    DEFAULT_TEXT_RECOGNITION_BATCH_SIZE,
    DEFAULT_TILE_OVERLAP_PX,
    PREPROCESSING_PRESET_TO_STEPS,
    PREPROCESSING_PRESET_CHOICES,
    PREPROCESSING_STEP_CHOICES,
)
from services.tile_ocr import tile_grid_for_strategy

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    REPO_ROOT
    / "testdata"
    / "source"
    / "small"
    / "image.jpg"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a headless OCR smoke test without launching Gradio."
    )
    parser.add_argument(
        "--method",
        choices=["PP-OCRv5", "PP-StructureV3", "PaddleVL", "Auto Routing"],
        default="PP-OCRv5",
        help="OCR method to test.",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=(
            "Input image file or folder. Defaults to a bundled repo-local "
            "sample under testdata/source."
        ),
    )
    parser.add_argument(
        "--output",
        default="output-smoke",
        help="Output folder for OCR results.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Recognition language for PP-OCRv5 and PP-StructureV3.",
    )
    parser.add_argument(
        "--text-det-thresh",
        type=float,
        default=DEFAULT_TEXT_DET_THRESH,
        help="Text detection threshold.",
    )
    parser.add_argument(
        "--text-det-limit-type",
        choices=["max", "min"],
        default=DEFAULT_TEXT_DET_LIMIT_TYPE,
        help="Text detection resize policy.",
    )
    parser.add_argument(
        "--text-det-limit-side-len",
        type=int,
        default=DEFAULT_TEXT_DET_LIMIT_SIDE_LEN,
        help="Text detection side length limit.",
    )
    parser.add_argument(
        "--text-det-max-side-limit",
        type=int,
        default=DEFAULT_TEXT_DET_MAX_SIDE_LIMIT,
        help="Text detection hard maximum side length before PaddleX clamps/resizes.",
    )
    parser.add_argument(
        "--text-det-box-thresh",
        type=float,
        default=DEFAULT_TEXT_DET_BOX_THRESH,
        help="Text detection box threshold.",
    )
    parser.add_argument(
        "--text-det-unclip-ratio",
        type=float,
        default=DEFAULT_TEXT_DET_UNCLIP_RATIO,
        help="Text detection unclip ratio.",
    )
    parser.add_argument(
        "--text-rec-score-thresh",
        type=float,
        default=DEFAULT_TEXT_REC_SCORE_THRESH,
        help="Text recognition score threshold.",
    )
    parser.add_argument(
        "--text-recognition-batch-size",
        type=int,
        default=DEFAULT_TEXT_RECOGNITION_BATCH_SIZE,
        help="Text recognition batch size.",
    )
    parser.add_argument(
        "--use-doc-orientation-classify",
        action="store_true",
        help="Enable document orientation classification.",
    )
    parser.add_argument(
        "--use-doc-unwarping",
        action="store_true",
        help="Enable document unwarping.",
    )
    parser.add_argument(
        "--use-textline-orientation",
        action="store_true",
        help="Enable textline orientation classification.",
    )
    parser.add_argument(
        "--use-layout-detection",
        action="store_true",
        help="Enable layout detection for PaddleVL or Auto Routing.",
    )
    parser.add_argument(
        "--generate-text-file",
        action="store_true",
        help="Generate TXT output instead of PDF/ALTO output.",
    )
    parser.add_argument(
        "--preprocessing-preset",
        choices=[value for _, value in PREPROCESSING_PRESET_CHOICES],
        default=DEFAULT_PREPROCESSING_PRESET,
        help=(
            "Legacy single image preprocessing preset. Ignored when "
            "--preprocessing-step is provided."
        ),
    )
    parser.add_argument(
        "--preprocessing-step",
        action="append",
        choices=[value for _, value in PREPROCESSING_STEP_CHOICES],
        dest="preprocessing_steps",
        default=None,
        help=(
            "Image preprocessing step applied before local Paddle OCR. "
            "Repeat this flag to select multiple steps."
        ),
    )
    parser.add_argument(
        "--crop-black-border",
        action="store_true",
        default=DEFAULT_IGNORE_BLACK_BORDER,
        help=(
            "Run local Paddle OCR on a detected page crop, then offset OCR "
            "geometry back into the original image coordinates."
        ),
    )
    parser.add_argument(
        "--tile-source-images",
        action="store_true",
        help="Split each source image into tiles before running local OCR.",
    )
    parser.add_argument(
        "--tile-strategy",
        choices=[
            "grid",
            "grid_3x2",
            "doclayout_v3",
            "grid_doclayout_v3",
            "grid_3x2_doclayout_v3",
        ],
        default="grid_doclayout_v3",
        help=(
            "Tile strategy: fixed 3x1/3x2 grid or PP-DocLayoutV3 layout regions. "
            "Use grid_doclayout_v3 or grid_3x2_doclayout_v3 to run "
            "PP-DocLayoutV3 inside each fixed grid tile."
        ),
    )
    parser.add_argument(
        "--tile-overlap-px",
        type=int,
        default=DEFAULT_TILE_OVERLAP_PX,
        help="Fixed grid tile overlap, in pixels.",
    )
    parser.add_argument(
        "--tile-doclayout-padding-px",
        type=int,
        default=DEFAULT_DOCLAYOUT_PADDING_PX,
        help="Optional PP-DocLayoutV3 tile padding, in pixels.",
    )
    parser.add_argument(
        "--tile-doclayout-threshold",
        type=float,
        default=DEFAULT_DOCLAYOUT_THRESHOLD,
        help="Optional PP-DocLayoutV3 detection threshold. Lower values recover more blocks.",
    )
    parser.add_argument(
        "--disable-doclayout-recovery",
        action="store_true",
        help="Disable the fallback OCR tile added to recover DocLayout missed regions.",
    )
    return parser.parse_args()


def build_options(args: argparse.Namespace) -> dict:
    tile_rows, tile_cols = tile_grid_for_strategy(args.tile_strategy)
    preprocessing_steps = args.preprocessing_steps
    if preprocessing_steps is None:
        preprocessing_steps = PREPROCESSING_PRESET_TO_STEPS.get(
            args.preprocessing_preset,
            [],
        )

    opt = {
        "method": args.method,
        "use_doc_orientation_classify": args.use_doc_orientation_classify,
        "use_doc_unwarping": args.use_doc_unwarping,
        "generate_text_file": args.generate_text_file,
        "preprocessing_steps": preprocessing_steps,
        "ignore_black_border": args.crop_black_border,
        "tile_source_images": args.tile_source_images,
        "tile_strategy": args.tile_strategy,
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
        "tile_overlap_px": max(0, args.tile_overlap_px),
        "tile_doclayout_padding_px": max(0, args.tile_doclayout_padding_px),
        "tile_doclayout_threshold": max(
            0.01,
            min(args.tile_doclayout_threshold, 1.0),
        ),
        "tile_doclayout_recovery": not args.disable_doclayout_recovery,
    }

    if args.method in {"PP-OCRv5", "PP-StructureV3", "Auto Routing"}:
        opt.update(
            {
                "lang": args.language,
                "use_textline_orientation": args.use_textline_orientation,
                "text_det_thresh": args.text_det_thresh,
                "text_det_limit_type": args.text_det_limit_type,
                "text_det_limit_side_len": max(1, args.text_det_limit_side_len),
                "text_det_max_side_limit": max(1, args.text_det_max_side_limit),
                "text_det_box_thresh": max(0.0, min(args.text_det_box_thresh, 1.0)),
                "text_det_unclip_ratio": max(0.1, args.text_det_unclip_ratio),
                "text_rec_score_thresh": max(0.0, min(args.text_rec_score_thresh, 1.0)),
                "text_recognition_batch_size": max(1, args.text_recognition_batch_size),
            }
        )

    if args.method in {"PaddleVL", "Auto Routing"}:
        opt["use_layout_detection"] = args.use_layout_detection

    return opt


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.is_absolute():
        input_path = (REPO_ROOT / input_path).resolve()

    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    opt = build_options(args)

    if args.method in {"PP-OCRv5", "PP-StructureV3"}:
        from services.ocr_pp import run_ocr_job

        iterator = run_ocr_job(input_path, output_path, opt)
    elif args.method == "PaddleVL":
        from services.ocr_vl import run_paddlevl_job

        iterator = run_paddlevl_job(input_path, output_path, opt)
    else:
        from services.auto_ocr import run_auto_job

        iterator = run_auto_job(input_path, output_path, opt)

    for message in iterator:
        print(message, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
