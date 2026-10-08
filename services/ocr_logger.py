"""Format and persist OCR run configuration, summaries, and error logs."""

import logging 
from datetime import datetime
from pathlib import Path

from services.ocr_defaults import (
    DEFAULT_DOCLAYOUT_PADDING_PX,
    DEFAULT_IGNORE_BLACK_BORDER,
    DEFAULT_OCR_VERSION,
    DEFAULT_OLMOCR_API_URL,
    DEFAULT_OLMOCR_MODEL,
    DEFAULT_PREPROCESSING_PRESET,
    DEFAULT_PREPROCESSING_STEPS,
    DEFAULT_CLEAN_PUNCTUATION_ISSUES,
    DEFAULT_DOCLAYOUT_THRESHOLD,
    DEFAULT_TEXT_DET_BOX_THRESH,
    DEFAULT_TEXT_DET_LIMIT_SIDE_LEN,
    DEFAULT_TEXT_DET_LIMIT_TYPE,
    DEFAULT_TEXT_DET_MAX_SIDE_LIMIT,
    DEFAULT_TEXT_DET_THRESH,
    DEFAULT_TEXT_DET_UNCLIP_RATIO,
    DEFAULT_TEXT_REC_SCORE_THRESH,
    DEFAULT_TEXT_RECOGNITION_BATCH_SIZE,
    PREPROCESSING_PRESET_TO_STEPS,
    PREPROCESSING_PRESET_LABELS,
    PREPROCESSING_STEP_LABELS,
)

PADDLE_OCR_METHODS = {"PP-OCRv6"}

OPTION_LABELS = {
    "method": "Method",
    "ocr_version": "PaddleOCR model version",
    "lang": "Language",
    "use_doc_orientation_classify": "Document orientation classification",
    "use_doc_unwarping": "Document unwarping",
    "use_textline_orientation": "Textline orientation",
    "generate_text_file": "Generate TXT instead of PDF/ALTO",
    "clean_punctuation_issues": "Clean punctuation artifacts in output text",
    "preprocessing_steps": "Image preprocessing steps",
    "preprocessing_preset": "Image preprocessing preset",
    "ignore_black_border": "Crop black border before OCR",
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
    "post_ocr_vl_enabled": "Post-OCR VL word check",
    "post_ocr_vl_candidate_mode": "Post-OCR VL candidate mode",
    "post_ocr_vl_max_words_enabled": "Post-OCR VL max words enabled",
    "post_ocr_vl_max_words": "Post-OCR VL max words",
    "post_ocr_vl_padding_px": "Post-OCR VL padding px",
    "post_ocr_vl_low_confidence_threshold": "Post-OCR VL low confidence threshold",
    "post_ocr_vl_apply_corrections": "Apply post-OCR VL corrections",
    "post_ocr_vl_save_tiles": "Save post-OCR VL tiles",
    "post_ocr_vl_concurrency": "Post-OCR VL concurrency",
    "post_ocr_vl_batch_size": "Post-OCR VL batch size",
    "post_ocr_vl_paddlevl_url": "Post-OCR PaddleVL URL",
    "olmocr_url": "OLM OCR API URL",
    "olmocr_model": "OLM OCR model",
    "olmocr_max_tokens": "OLM OCR max tokens",
}

OPTION_ORDER = [
    "method",
    "lang",
    "use_doc_orientation_classify",
    "use_doc_unwarping",
    "use_textline_orientation",
    "generate_text_file",
    "clean_punctuation_issues",
    "preprocessing_steps",
    "ignore_black_border",
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
    "post_ocr_vl_enabled",
    "post_ocr_vl_candidate_mode",
    "post_ocr_vl_max_words_enabled",
    "post_ocr_vl_max_words",
    "post_ocr_vl_padding_px",
    "post_ocr_vl_low_confidence_threshold",
    "post_ocr_vl_apply_corrections",
    "post_ocr_vl_save_tiles",
    "post_ocr_vl_concurrency",
    "post_ocr_vl_batch_size",
    "post_ocr_vl_paddlevl_url",
    "olmocr_url",
    "olmocr_model",
    "olmocr_max_tokens",
]

TILE_STRATEGY_LABELS = {
    "grid": "Fixed 3x1 grid",
    "grid_3x2": "Fixed 3x2 grid",
    "doclayout_v3": "PP-DocLayoutV3 Layout",
    "grid_doclayout_v3": "3x1 + PP-DocLayoutV3",
    "grid_3x2_doclayout_v3": "3x2 + PP-DocLayoutV3",
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
POST_OCR_VL_DETAIL_KEYS = {
    "post_ocr_vl_candidate_mode",
    "post_ocr_vl_max_words_enabled",
    "post_ocr_vl_max_words",
    "post_ocr_vl_padding_px",
    "post_ocr_vl_low_confidence_threshold",
    "post_ocr_vl_apply_corrections",
    "post_ocr_vl_save_tiles",
    "post_ocr_vl_concurrency",
    "post_ocr_vl_batch_size",
    "post_ocr_vl_paddlevl_url",
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

PREPROCESSING_DEFAULT_OPTIONS = {
    "preprocessing_steps": "None (default)",
    "ignore_black_border": (
        "on (default)" if DEFAULT_IGNORE_BLACK_BORDER else "off (default)"
    ),
    "clean_punctuation_issues": (
        "on (default)" if DEFAULT_CLEAN_PUNCTUATION_ISSUES else "off (default)"
    ),
}

PREPROCESSING_STEP_ORDER = tuple(PREPROCESSING_STEP_LABELS)


def tile_strategy_from_options(opt):
    """Return the configured tile strategy string from a run options dict."""
    return str(opt.get("tile_strategy") or "").strip()


def option_applies_to_strategy(key, opt):
    """Return whether an option is relevant for the current method and tiling mode."""
    method = opt.get("method")
    tile_enabled = bool(opt.get("tile_source_images", False))
    strategy = tile_strategy_from_options(opt)
    if method not in PADDLE_OCR_METHODS and key == "text_det_max_side_limit":
        return False

    if key == "post_ocr_vl_enabled":
        return method in PADDLE_OCR_METHODS

    if key == "post_ocr_vl_max_words":
        return (
            method in PADDLE_OCR_METHODS
            and bool(opt.get("post_ocr_vl_enabled"))
            and bool(opt.get("post_ocr_vl_max_words_enabled"))
        )

    if key == "post_ocr_vl_paddlevl_url":
        return method in PADDLE_OCR_METHODS and bool(opt.get("post_ocr_vl_enabled"))

    if key in POST_OCR_VL_DETAIL_KEYS:
        return method in PADDLE_OCR_METHODS and bool(opt.get("post_ocr_vl_enabled"))

    if key in {"olmocr_url", "olmocr_model", "olmocr_max_tokens"}:
        return method == "OLMOCR"

    if key == "tile_source_images":
        return True

    if key in TILE_DETAIL_KEYS and not tile_enabled:
        return False

    if key in GRID_TILE_KEYS:
        return strategy in {"grid", "grid_3x2", "grid_doclayout_v3", "grid_3x2_doclayout_v3"}

    if key in DOCLAYOUT_TILE_KEYS:
        return strategy in {"doclayout_v3", "grid_doclayout_v3", "grid_3x2_doclayout_v3"}

    return True


def default_strategy_options(opt):
    """Yield documented default values for the active tiling strategy."""
    strategy = tile_strategy_from_options(opt)

    if strategy in {"doclayout_v3", "grid_doclayout_v3", "grid_3x2_doclayout_v3"}:
        for key, value in DOCLAYOUT_DEFAULT_OPTIONS.items():
            if key not in opt:
                yield key, value


def default_paddle_options(opt):
    """Yield documented default OCR thresholds for Paddle-based runs."""
    method = opt.get("method")
    if method in PADDLE_OCR_METHODS:
        for key, value in PADDLE_DEFAULT_OPTIONS.items():
            if key not in opt:
                yield key, value


def default_preprocessing_options(opt):
    """Yield documented default preprocessing-related options."""
    for key, value in PREPROCESSING_DEFAULT_OPTIONS.items():
        if key == "preprocessing_steps" and "preprocessing_preset" in opt:
            continue
        if key not in opt:
            yield key, value


def preprocessing_steps_from_options(opt):
    """Resolve the ordered preprocessing steps implied by the option set."""
    if "preprocessing_steps" in opt:
        selected = opt.get("preprocessing_steps") or []
        if isinstance(selected, str):
            selected = [
                step.strip()
                for step in selected.replace(";", ",").split(",")
                if step.strip()
            ]
    else:
        preset = str(opt.get("preprocessing_preset") or DEFAULT_PREPROCESSING_PRESET)
        selected = PREPROCESSING_PRESET_TO_STEPS.get(preset, DEFAULT_PREPROCESSING_STEPS)

    selected_set = {str(step) for step in selected}
    return [step for step in PREPROCESSING_STEP_ORDER if step in selected_set]


def format_preprocessing_steps(value):
    """Format preprocessing steps for human-readable logs."""
    selected = preprocessing_steps_from_options({"preprocessing_steps": value})
    if not selected:
        return "None"
    return ", ".join(PREPROCESSING_STEP_LABELS.get(step, step) for step in selected)


def format_option_value(key, value):
    """Format one option value using the same labels shown in logs and UI."""
    if key == "preprocessing_steps":
        return format_preprocessing_steps(value)

    if key == "preprocessing_preset":
        return PREPROCESSING_PRESET_LABELS.get(str(value), str(value))

    if key == "tile_strategy":
        return TILE_STRATEGY_LABELS.get(str(value), str(value))

    if isinstance(value, bool):
        return "on" if value else "off"

    return str(value)


def iter_selected_options(opt):
    """Yield the run options that should be written into the OCR log."""
    seen = set()
    defaults = {
        **dict(default_preprocessing_options(opt)),
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
    """Summarize the models and major processing stages active for a run."""
    method = opt.get("method")
    effective_method = select or method
    tile_enabled = bool(opt.get("tile_source_images", False))
    tile_strategy = opt.get("tile_strategy")
    preprocessing_steps = preprocessing_steps_from_options(opt)
    ignore_black_border = bool(opt.get("ignore_black_border", DEFAULT_IGNORE_BLACK_BORDER))
    models = []

    if effective_method in PADDLE_OCR_METHODS:
        ocr_version = DEFAULT_OCR_VERSION
        models.append(f"OCR pipeline: PaddleOCR {ocr_version}")
        models.append(f"Text detection: PaddleOCR {ocr_version} detection model")
        models.append(f"Text recognition: {ocr_version} recognition model")
        models.append("Word boxes: enabled")
    elif effective_method == "PaddleVL":
        models.append("OCR pipeline: PaddleOCR-VL 1.6 API service")
    elif effective_method == "OLMOCR":
        models.append("OCR pipeline: olmOCR vLLM API service")
        models.append(f"OLM OCR model: {opt.get('olmocr_model') or DEFAULT_OLMOCR_MODEL}")
        models.append(f"OLM OCR API URL: {opt.get('olmocr_url') or DEFAULT_OLMOCR_API_URL}")
    elif method:
        models.append(f"Primary pipeline: {method}")

    if tile_enabled and tile_strategy in {"doclayout_v3", "grid_doclayout_v3", "grid_3x2_doclayout_v3"}:
        models.append("Tile layout detector: PP-DocLayoutV3")

    if tile_enabled:
        strategy_label = TILE_STRATEGY_LABELS.get(str(tile_strategy), str(tile_strategy))
        models.append(f"Tile OCR strategy: {strategy_label}")

    if opt.get("post_ocr_vl_enabled") and effective_method in PADDLE_OCR_METHODS:
        models.append("Post-OCR word check: PaddleVL")

    if effective_method in PADDLE_OCR_METHODS and preprocessing_steps:
        models.append(
            f"Image preprocessing: {format_preprocessing_steps(preprocessing_steps)}"
        )

    if effective_method in PADDLE_OCR_METHODS and ignore_black_border:
        models.append("Image preprocessing: crop black border and offset OCR geometry")

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
    """Write selected run options and model choices into the active logger."""
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
    Singleton-style logger manager shared by the OCR pipeline.
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
