import logging 
from datetime import datetime
from pathlib import Path

from services.ocr_defaults import (
    DEFAULT_DOCLAYOUT_PADDING_PX,
    DEFAULT_IGNORE_BLACK_BORDER,
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

OPTION_LABELS = {
    "method": "Method",
    "lang": "Language",
    "use_doc_orientation_classify": "Document orientation classification",
    "use_doc_unwarping": "Document unwarping",
    "use_textline_orientation": "Textline orientation",
    "use_layout_detection": "PaddleVL layout detection",
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
    "pero_source_path": "PERO Python module path",
    "pero_config_path": "PERO config.ini path",
    "pero_device": "PERO device",
    "pero_gpu_id": "PERO GPU id",
    "pero_min_line_confidence": "PERO min line confidence",
    "pero_save_pagexml": "Save PERO PAGE XML",
    "post_ocr_vl_enabled": "Post-OCR VL word check",
    "post_ocr_vl_backend": "Post-OCR VL backend",
    "post_ocr_vl_candidate_mode": "Post-OCR VL candidate mode",
    "post_ocr_vl_max_words_enabled": "Post-OCR VL max words enabled",
    "post_ocr_vl_max_words": "Post-OCR VL max words",
    "post_ocr_vl_padding_px": "Post-OCR VL padding px",
    "post_ocr_vl_low_confidence_threshold": "Post-OCR VL low confidence threshold",
    "post_ocr_vl_apply_corrections": "Apply post-OCR VL corrections",
    "post_ocr_vl_save_tiles": "Save post-OCR VL tiles",
    "post_ocr_vl_group_lines": "Group post-OCR VL same-line candidates",
    "post_ocr_vl_paddlevl_url": "Post-OCR PaddleVL URL",
    "post_ocr_vl_ollama_openai_url": "Post-OCR Ollama OpenAI URL",
    "post_ocr_vl_ollama_openai_model": "Post-OCR Ollama OpenAI model",
}

OPTION_ORDER = [
    "method",
    "lang",
    "use_doc_orientation_classify",
    "use_doc_unwarping",
    "use_textline_orientation",
    "use_layout_detection",
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
    "pero_source_path",
    "pero_config_path",
    "pero_device",
    "pero_gpu_id",
    "pero_min_line_confidence",
    "pero_save_pagexml",
    "post_ocr_vl_enabled",
    "post_ocr_vl_backend",
    "post_ocr_vl_candidate_mode",
    "post_ocr_vl_max_words_enabled",
    "post_ocr_vl_max_words",
    "post_ocr_vl_padding_px",
    "post_ocr_vl_low_confidence_threshold",
    "post_ocr_vl_apply_corrections",
    "post_ocr_vl_save_tiles",
    "post_ocr_vl_group_lines",
    "post_ocr_vl_paddlevl_url",
    "post_ocr_vl_ollama_openai_url",
    "post_ocr_vl_ollama_openai_model",
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
    "post_ocr_vl_backend",
    "post_ocr_vl_candidate_mode",
    "post_ocr_vl_max_words_enabled",
    "post_ocr_vl_max_words",
    "post_ocr_vl_padding_px",
    "post_ocr_vl_low_confidence_threshold",
    "post_ocr_vl_apply_corrections",
    "post_ocr_vl_save_tiles",
    "post_ocr_vl_group_lines",
    "post_ocr_vl_paddlevl_url",
    "post_ocr_vl_ollama_openai_url",
    "post_ocr_vl_ollama_openai_model",
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
    return str(opt.get("tile_strategy") or "").strip()


def option_applies_to_strategy(key, opt):
    method = opt.get("method")
    tile_enabled = bool(opt.get("tile_source_images", False))
    strategy = tile_strategy_from_options(opt)

    if method != "PP-OCRv5" and key == "text_det_max_side_limit":
        return False

    if key == "post_ocr_vl_enabled":
        return method == "PP-OCRv5"

    if key == "post_ocr_vl_max_words":
        return (
            method == "PP-OCRv5"
            and bool(opt.get("post_ocr_vl_enabled"))
            and bool(opt.get("post_ocr_vl_max_words_enabled"))
        )

    if key in POST_OCR_VL_DETAIL_KEYS:
        return method == "PP-OCRv5" and bool(opt.get("post_ocr_vl_enabled"))

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
    strategy = tile_strategy_from_options(opt)

    if strategy in {"doclayout_v3", "grid_doclayout_v3", "grid_3x2_doclayout_v3"}:
        for key, value in DOCLAYOUT_DEFAULT_OPTIONS.items():
            if key not in opt:
                yield key, value


def default_paddle_options(opt):
    method = opt.get("method")
    if method in {"PP-OCRv5", "PP-StructureV3", "Auto Routing"}:
        for key, value in PADDLE_DEFAULT_OPTIONS.items():
            if method != "PP-OCRv5" and key == "text_det_max_side_limit":
                continue
            if key not in opt:
                yield key, value


def default_preprocessing_options(opt):
    for key, value in PREPROCESSING_DEFAULT_OPTIONS.items():
        if key == "preprocessing_steps" and "preprocessing_preset" in opt:
            continue
        if key not in opt:
            yield key, value


def preprocessing_steps_from_options(opt):
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
    selected = preprocessing_steps_from_options({"preprocessing_steps": value})
    if not selected:
        return "None"
    return ", ".join(PREPROCESSING_STEP_LABELS.get(step, step) for step in selected)


def format_option_value(key, value):
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
    method = opt.get("method")
    effective_method = select or method
    tile_enabled = bool(opt.get("tile_source_images", False))
    tile_strategy = opt.get("tile_strategy")
    preprocessing_steps = preprocessing_steps_from_options(opt)
    ignore_black_border = bool(opt.get("ignore_black_border", DEFAULT_IGNORE_BLACK_BORDER))
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
    elif effective_method == "PERO-OCR":
        models.append("OCR pipeline: PERO-OCR PageParser")
        models.append("Text layout/recognition: PERO engine config")
    elif method:
        models.append(f"Primary pipeline: {method}")

    if tile_enabled and tile_strategy in {"doclayout_v3", "grid_doclayout_v3", "grid_3x2_doclayout_v3"}:
        models.append("Tile layout detector: PP-DocLayoutV3")

    if tile_enabled:
        strategy_label = TILE_STRATEGY_LABELS.get(str(tile_strategy), str(tile_strategy))
        models.append(f"Tile OCR strategy: {strategy_label}")

    if opt.get("post_ocr_vl_enabled") and effective_method == "PP-OCRv5":
        backend = opt.get("post_ocr_vl_backend") or "paddlevl"
        models.append(f"Post-OCR word check: {backend}")

    if effective_method in {"PP-OCRv5", "PP-StructureV3", "PERO-OCR"} and preprocessing_steps:
        models.append(
            f"Image preprocessing: {format_preprocessing_steps(preprocessing_steps)}"
        )

    if effective_method in {"PP-OCRv5", "PP-StructureV3", "PERO-OCR"} and ignore_black_border:
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
