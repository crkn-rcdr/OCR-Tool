import logging 
from datetime import datetime
from pathlib import Path

from services.ocr_defaults import (
    DEFAULT_DOCLAYOUT_PADDING_PX,
    DEFAULT_DOCLAYOUT_THRESHOLD,
    DEFAULT_TEXT_DET_BOX_THRESH,
    DEFAULT_TEXT_DET_LIMIT_SIDE_LEN,
    DEFAULT_TEXT_DET_LIMIT_TYPE,
    DEFAULT_TEXT_DET_MAX_SIDE_LIMIT,
    DEFAULT_TEXT_DET_THRESH,
    DEFAULT_TEXT_DET_UNCLIP_RATIO,
    DEFAULT_TEXT_REC_SCORE_THRESH,
    DEFAULT_TEXT_RECOGNITION_BATCH_SIZE,
)

OPTION_LABELS = {
    "method": "Method",
    "lang": "Language",
    "use_doc_orientation_classify": "Document orientation classification",
    "use_doc_unwarping": "Document unwarping",
    "use_textline_orientation": "Textline orientation",
    "use_layout_detection": "PaddleVL layout detection",
    "generate_text_file": "Generate TXT instead of PDF/ALTO",
    "tile_source_images": "Tile OCR enabled",
    "tile_strategy": "Tile strategy",
    "tile_rows": "Tile rows",
    "tile_cols": "Tile columns",
    "tile_overlap_px": "Tile overlap px",
    "tile_doclayout_padding_px": "DocLayout padding px",
    "tile_doclayout_threshold": "DocLayout threshold",
    "tile_doclayout_recovery": "DocLayout recovery tile",
    "text_det_thresh": "Text detection threshold",
    "text_det_limit_type": "Text detection limit type",
    "text_det_limit_side_len": "Text detection limit side length",
    "text_det_max_side_limit": "Text detection max side limit",
    "text_det_box_thresh": "Text detection box threshold",
    "text_det_unclip_ratio": "Text detection unclip ratio",
    "text_rec_score_thresh": "Text recognition score threshold",
    "text_recognition_batch_size": "Text recognition batch size",
}

OPTION_ORDER = [
    "method",
    "lang",
    "use_doc_orientation_classify",
    "use_doc_unwarping",
    "use_textline_orientation",
    "use_layout_detection",
    "generate_text_file",
    "tile_source_images",
    "tile_strategy",
    "tile_rows",
    "tile_cols",
    "tile_overlap_px",
    "tile_doclayout_padding_px",
    "tile_doclayout_threshold",
    "tile_doclayout_recovery",
    "text_det_thresh",
    "text_det_limit_type",
    "text_det_limit_side_len",
    "text_det_max_side_limit",
    "text_det_box_thresh",
    "text_det_unclip_ratio",
    "text_rec_score_thresh",
    "text_recognition_batch_size",
]

TILE_STRATEGY_LABELS = {
    "grid": "Fixed 3x1 grid",
    "doclayout_v3": "PP-DocLayoutV3 Layout",
    "grid_doclayout_v3": "3x1 + PP-DocLayoutV3",
}

GRID_TILE_KEYS = {"tile_rows", "tile_cols", "tile_overlap_px"}
DOCLAYOUT_TILE_KEYS = {
    "tile_doclayout_padding_px",
    "tile_doclayout_threshold",
    "tile_doclayout_recovery",
}
TILE_DETAIL_KEYS = {
    "tile_strategy",
    *GRID_TILE_KEYS,
    *DOCLAYOUT_TILE_KEYS,
}

DOCLAYOUT_DEFAULT_OPTIONS = {
    "tile_doclayout_padding_px": f"{DEFAULT_DOCLAYOUT_PADDING_PX} (default)",
    "tile_doclayout_threshold": f"{DEFAULT_DOCLAYOUT_THRESHOLD:g} (default)",
    "tile_doclayout_recovery": "on (default)",
}

PADDLE_DEFAULT_OPTIONS = {
    "text_det_thresh": f"{DEFAULT_TEXT_DET_THRESH:g} (default)",
    "text_det_limit_type": f"{DEFAULT_TEXT_DET_LIMIT_TYPE} (default)",
    "text_det_limit_side_len": f"{DEFAULT_TEXT_DET_LIMIT_SIDE_LEN} (default)",
    "text_det_max_side_limit": f"{DEFAULT_TEXT_DET_MAX_SIDE_LIMIT} (default)",
    "text_det_box_thresh": f"{DEFAULT_TEXT_DET_BOX_THRESH:g} (default)",
    "text_det_unclip_ratio": f"{DEFAULT_TEXT_DET_UNCLIP_RATIO:g} (default)",
    "text_rec_score_thresh": f"{DEFAULT_TEXT_REC_SCORE_THRESH:g} (default)",
    "text_recognition_batch_size": f"{DEFAULT_TEXT_RECOGNITION_BATCH_SIZE} (default)",
}


def tile_strategy_from_options(opt):
    return str(opt.get("tile_strategy") or "").strip()


def option_applies_to_strategy(key, opt):
    tile_enabled = bool(opt.get("tile_source_images", False))
    strategy = tile_strategy_from_options(opt)

    if key == "tile_source_images":
        return True

    if key in TILE_DETAIL_KEYS and not tile_enabled:
        return False

    if key in GRID_TILE_KEYS:
        return strategy in {"grid", "grid_doclayout_v3"}

    if key in DOCLAYOUT_TILE_KEYS:
        return strategy in {"doclayout_v3", "grid_doclayout_v3"}

    return True


def default_strategy_options(opt):
    strategy = tile_strategy_from_options(opt)

    if strategy in {"doclayout_v3", "grid_doclayout_v3"}:
        for key, value in DOCLAYOUT_DEFAULT_OPTIONS.items():
            if key not in opt:
                yield key, value


def default_paddle_options(opt):
    method = opt.get("method")
    if method in {"PP-OCRv5", "PP-StructureV3", "Auto Routing"}:
        for key, value in PADDLE_DEFAULT_OPTIONS.items():
            if key not in opt:
                yield key, value


def format_option_value(key, value):
    if key == "tile_strategy":
        return TILE_STRATEGY_LABELS.get(str(value), str(value))

    if isinstance(value, bool):
        return "on" if value else "off"

    return str(value)


def iter_selected_options(opt):
    seen = set()
    defaults = {
        **dict(default_strategy_options(opt)),
        **dict(default_paddle_options(opt)),
    }

    for key in OPTION_ORDER:
        seen.add(key)

        if key in opt and option_applies_to_strategy(key, opt):
            yield OPTION_LABELS.get(key, key), format_option_value(key, opt[key])
        elif key in defaults and option_applies_to_strategy(key, opt):
            yield OPTION_LABELS.get(key, key), defaults[key]

    for key in sorted(k for k in opt if k not in seen):
        if option_applies_to_strategy(key, opt):
            yield OPTION_LABELS.get(key, key), format_option_value(key, opt[key])


def models_for_run(opt, select=None):
    method = opt.get("method")
    effective_method = select or method
    tile_enabled = bool(opt.get("tile_source_images", False))
    tile_strategy = opt.get("tile_strategy")
    models = []

    if select and select != method:
        models.append(f"Auto Routing selected engine: {select}")

    if effective_method == "PP-OCRv5":
        models.append("OCR pipeline: PaddleOCR PP-OCRv5")
        models.append("Text detection: PaddleOCR PP-OCRv5 detection model")
        models.append("Text recognition: PP-OCRv5 server recognition model")
        models.append("Word boxes: enabled")
    elif effective_method == "PP-StructureV3":
        models.append("OCR/layout pipeline: PP-StructureV3")
    elif effective_method == "PaddleVL":
        models.append("Layout parsing pipeline: PaddleOCR-VL API service")
    elif method:
        models.append(f"Primary pipeline: {method}")

    if tile_enabled and tile_strategy in {"doclayout_v3", "grid_doclayout_v3"}:
        models.append("Tile layout detector: PP-DocLayoutV3")

    if tile_enabled:
        strategy_label = TILE_STRATEGY_LABELS.get(str(tile_strategy), str(tile_strategy))
        models.append(f"Tile OCR strategy: {strategy_label}")

    models.append("PDF text layer: PyMuPDF")
    models.append("ALTO writer: local ALTO generator")

    return models


def log_run_configuration(
    logger,
    opt,
    select=None,
    input_path=None,
    output_root=None,
    stage="OCR",
):
    logger.info(f"========== {stage} RUN CONFIGURATION ==========")

    if input_path is not None:
        logger.info(f"Input path: {input_path}")
    if output_root is not None:
        logger.info(f"Output root: {output_root}")

    logger.info("Selected options:")
    for label, value in iter_selected_options(opt or {}):
        logger.info(f"  {label}: {value}")

    logger.info("Models / components used:")
    for model in models_for_run(opt or {}, select=select):
        logger.info(f"  - {model}")


class OCRLoggerManger:
    """
    Singleton-style logger manager shared across ocr_pp.py, ocr_vl.py, and auto_ocr.py.
    """
    _initialized = False
    _summary_logger = None
    _error_logger = None
    _summary_log_path = None
    _error_log_path = None
    _run_timestamp = None

    @classmethod
    def _reset(cls):
        for logger in (cls._summary_logger, cls._error_logger):
            if logger is None:
                continue

            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

        cls._initialized = False
        cls._summary_logger = None
        cls._error_logger = None
        cls._summary_log_path = None
        cls._error_log_path = None
        cls._run_timestamp = None

    @classmethod
    def setup(cls,method:str,log_dir:str="logs"):
        if cls._initialized:
            cls._reset()

        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        cls._run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        method_sanitized = method.replace("-", "_")

        summary_log_path = log_dir_path / f"{method_sanitized}_summary_{cls._run_timestamp}.log"
        error_log_path = log_dir_path / f"{method_sanitized}_error_{cls._run_timestamp}.log"

        formatter = logging.Formatter('%(asctime)s | %(levelname)s |%(message)s')

        summary_logger = logging.getLogger("ocr_summary")
        summary_logger.setLevel(logging.INFO)
        summary_logger.handlers.clear()
        summary_logger.propagate = False

        summary_handler = logging.FileHandler(summary_log_path,encoding='utf-8')
        summary_handler.setFormatter(formatter)
        summary_logger.addHandler(summary_handler)

        error_logger = logging.getLogger("ocr_error")
        error_logger.setLevel(logging.ERROR)
        error_logger.handlers.clear()
        error_logger.propagate = False

        error_handler = logging.FileHandler(error_log_path,encoding='utf-8')
        error_handler.setFormatter(formatter)
        error_logger.addHandler(error_handler)

        cls._summary_log_path = summary_log_path
        cls._error_log_path = error_log_path
        cls._summary_logger = summary_logger
        cls._error_logger = error_logger
        cls._initialized = True

        return (
            cls._summary_logger,
            cls._error_logger,
            cls._summary_log_path,
            cls._error_log_path
        )
