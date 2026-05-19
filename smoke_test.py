import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    REPO_ROOT
    / "testdata"
    / "source"
    / "aeu.00037_19470820"
    / "aeu.00037_19470820.1.jpg"
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
        default=0.3,
        help="Text detection threshold.",
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
    return parser.parse_args()


def build_options(args: argparse.Namespace) -> dict:
    opt = {
        "method": args.method,
        "use_doc_orientation_classify": args.use_doc_orientation_classify,
        "use_doc_unwarping": args.use_doc_unwarping,
        "generate_text_file": args.generate_text_file,
    }

    if args.method in {"PP-OCRv5", "PP-StructureV3", "Auto Routing"}:
        opt.update(
            {
                "lang": args.language,
                "use_textline_orientation": args.use_textline_orientation,
                "text_det_thresh": args.text_det_thresh,
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
