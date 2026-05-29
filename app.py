import gradio as gr
import os
try:
    import tkinter as tk
    from tkinter import filedialog
except ModuleNotFoundError:
    tk = None
    filedialog = None

from pathlib import Path

from services.ocr_pp import run_ocr_job,stop_ocr
from services.ocr_vl import run_paddlevl_job
from services.auto_ocr import run_auto_job
from services.ocr_defaults import (
    DEFAULT_CHANDRA_API_URL,
    DEFAULT_CHANDRA_MODEL,
    DEFAULT_CLEAN_PUNCTUATION_ISSUES,
    DEFAULT_DOCLAYOUT_PADDING_PX,
    DEFAULT_IGNORE_BLACK_BORDER,
    DEFAULT_OLLAMA_OPENAI_API_URL,
    DEFAULT_OLLAMA_OPENAI_MODEL,
    DEFAULT_PADDLE_VL_API_URL,
    DEFAULT_POST_OCR_VL_APPLY_CORRECTIONS,
    DEFAULT_POST_OCR_VL_BACKEND,
    DEFAULT_POST_OCR_VL_CANDIDATE_MODE,
    DEFAULT_POST_OCR_VL_ENABLED,
    DEFAULT_POST_OCR_VL_GROUP_LINES,
    DEFAULT_POST_OCR_VL_LOW_CONFIDENCE_THRESHOLD,
    DEFAULT_POST_OCR_VL_MAX_WORDS,
    DEFAULT_POST_OCR_VL_PADDING_PX,
    DEFAULT_POST_OCR_VL_SAVE_TILES,
    DEFAULT_PREPROCESSING_STEPS,
    DEFAULT_TEXT_DET_BOX_THRESH,
    DEFAULT_DOCLAYOUT_THRESHOLD,
    DEFAULT_TEXT_DET_LIMIT_SIDE_LEN,
    DEFAULT_TEXT_DET_LIMIT_TYPE,
    DEFAULT_TEXT_DET_MAX_SIDE_LIMIT,
    DEFAULT_TEXT_DET_THRESH,
    DEFAULT_TEXT_DET_UNCLIP_RATIO,
    DEFAULT_TEXT_REC_SCORE_THRESH,
    DEFAULT_TEXT_RECOGNITION_BATCH_SIZE,
    DEFAULT_TILE_OVERLAP_PX,
    PREPROCESSING_STEP_CHOICES,
)
from services.tile_ocr import tile_grid_for_strategy

LOCAL_OCR_METHODS = {"PP-OCRv5", "PP-StructureV3"}
POST_OCR_VL_BACKEND_CHOICES = [
    ("PaddleVL", "paddlevl"),
    ("Chandra", "chandra"),
    ("OpenAI-compatible", "olmocr"),
    ("PaddleVL + Chandra", "both_chandra"),
    ("PaddleVL + OpenAI-compatible", "both"),
]

APP_CSS = """
#fixed-header {
    position: fixed;
    top: 0;
    background: white;
    z-index: 1000;
    padding: 10px 0;
}
.gradio-container {
    padding-top: 100px !important;
}
"""

FULL_TILE_STRATEGY_CHOICES = [
    ("Fixed 3x1 grid", "grid"),
    ("Fixed 3x2 grid", "grid_3x2"),
    ("PP-DocLayoutV3 Layout", "doclayout_v3"),
    ("3x1 + PP-DocLayoutV3", "grid_doclayout_v3"),
    ("3x2 + PP-DocLayoutV3", "grid_3x2_doclayout_v3"),
]
PPSTRUCTURE_TILE_STRATEGY_CHOICES = [
    ("Fixed 3x1 grid", "grid"),
    ("Fixed 3x2 grid", "grid_3x2"),
]
PPSTRUCTURE_TILE_STRATEGIES = {value for _, value in PPSTRUCTURE_TILE_STRATEGY_CHOICES}
DEFAULT_TILE_STRATEGY = "grid_3x2_doclayout_v3"
DEFAULT_PPSTRUCTURE_TILE_STRATEGY = "grid_3x2"

# Define the list of supported languages for the dropdown menu
LANGUAGE_CHOICES = [
    ("Abkhazian", "abq"),
    ("Afrikaans", "af"),
    ("Old English", "ang"),
    ("Arabic", "ar"),
    ("Avar", "ava"),
    ("Azerbaijani", "az"),
    ("Belarusian", "be"),
    ("Bulgarian", "bg"),
    ("Haryanvi", "bgc"),
    ("Bihari", "bh"),
    ("Bhojpuri", "bho"),
    ("Bosnian", "bs"),
    ("Simplified Chinese", "ch"),
    ("Chechen", "che"),
    ("Traditional Chinese", "chinese_cht"),
    ("Czech", "cs"),
    ("Welsh", "cy"),
    ("Danish", "da"),
    ("Dargwa", "dar"),
    ("German", "de"),
    ("English", "en"),
    ("Spanish", "es"),
    ("Estonian", "et"),
    ("Persian", "fa"),
    ("French", "fr"),
    ("Irish", "ga"),
    ("Konkani", "gom"),
    ("Hindi", "hi"),
    ("Croatian", "hr"),
    ("Hungarian", "hu"),
    ("Indonesian", "id"),
    ("Ingush", "inh"),
    ("Icelandic", "is"),
    ("Italian", "it"),
    ("Japanese", "japan"),
    ("Georgian", "ka"),
    ("Kabardian", "kbd"),
    ("Korean", "korean"),
    ("Kurdish", "ku"),
    ("Latin", "la"),
    ("Lak", "lbe"),
    ("Lezgin", "lez"),
    ("Lithuanian", "lt"),
    ("Latvian", "lv"),
    ("Magahi", "mah"),
    ("Maithili", "mai"),
    ("Maori", "mi"),
    ("Mongolian", "mn"),
    ("Marathi", "mr"),
    ("Malay", "ms"),
    ("Maltese", "mt"),
    ("Nepali", "ne"),
    ("Newari", "new"),
    ("Dutch", "nl"),
    ("Norwegian", "no"),
    ("Occitan", "oc"),
    ("Pali", "pi"),
    ("Polish", "pl"),
    ("Portuguese", "pt"),
    ("Romanian", "ro"),
    ("Serbian (Cyrillic)", "rs_cyrillic"),
    ("Serbian (Latin)", "rs_latin"),
    ("Russian", "ru"),
    ("Sanskrit", "sa"),
    ("Sadri", "sck"),
    ("Slovak", "sk"),
    ("Slovenian", "sl"),
    ("Albanian", "sq"),
    ("Swedish", "sv"),
    ("Swahili", "sw"),
    ("Tabasaran", "tab"),
    ("Tamil", "ta"),
    ("Telugu", "te"),
    ("Tagalog", "tl"),
    ("Turkish", "tr"),
    ("Uyghur", "ug"),
    ("Ukrainian", "uk"),
    ("Urdu", "ur"),
    ("Uzbek", "uz"),
    ("Vietnamese", "vi"),
]


def set_ui_interactive(interactive: bool):
    """
    Enable or disable all UI components.
    """
    return [ 
        gr.update(interactive=interactive) for _ in ui_components 
    ]

def disable_run_button():
    """
    Disable the Run OCR button.
    """
    return gr.update(interactive=False)


def enable_run_button():
    """
    Enable the Run OCR button.
    """
    return gr.update(interactive=True)

def choose_folder(current_value=""):
    """
    Open a folder selection dialog and return the selected folder path.
    """
    if tk is None or filedialog is None:
        gr.Info(
            "Folder browsing is unavailable in this WSL environment. "
            "Paste the folder path manually."
        )
        return current_value

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory()
        root.destroy()
        return folder or current_value
    except Exception:
        gr.Info(
            "Folder browsing is unavailable in this environment. "
            "Paste the folder path manually."
        )
        return current_value


def coerce_tile_strategy_for_method(method, tile_strategy):
    strategy = tile_strategy or DEFAULT_TILE_STRATEGY
    if method == "PP-StructureV3" and strategy not in PPSTRUCTURE_TILE_STRATEGIES:
        return DEFAULT_PPSTRUCTURE_TILE_STRATEGY
    return strategy


def tile_strategy_choices_for_method(method):
    if method == "PP-StructureV3":
        return PPSTRUCTURE_TILE_STRATEGY_CHOICES
    return FULL_TILE_STRATEGY_CHOICES


def update_tile_controls(method, tile_strategy):
    supports_tiling = method in LOCAL_OCR_METHODS
    strategy = coerce_tile_strategy_for_method(method, tile_strategy)
    show_fixed_grid_controls = supports_tiling and strategy in {
        "grid",
        "grid_3x2",
        "grid_doclayout_v3",
        "grid_3x2_doclayout_v3",
    }
    show_doclayout_controls = supports_tiling and strategy in {
        "doclayout_v3",
        "grid_doclayout_v3",
        "grid_3x2_doclayout_v3",
    }

    return (
        gr.update(visible=show_fixed_grid_controls, interactive=supports_tiling),
        gr.update(visible=show_doclayout_controls, interactive=supports_tiling),
        gr.update(visible=show_doclayout_controls, interactive=supports_tiling),
        gr.update(visible=show_doclayout_controls, interactive=supports_tiling),
    )


def update_options(method, tile_strategy):
    """
    Update visible UI options based on the selected OCR method.
    """
    supports_tiling = method in LOCAL_OCR_METHODS
    show_tiling_options = method in ["PP-OCRv5", "Auto Routing"]
    supports_preprocessing = method in [
        "PP-OCRv5",
        "PP-StructureV3",
        "Auto Routing",
    ]
    effective_tile_strategy = coerce_tile_strategy_for_method(method, tile_strategy)
    tile_checkbox_update = (
        gr.update(interactive=True)
        if supports_tiling
        else gr.update(interactive=False, value=False)
    )
    if not show_tiling_options:
        tile_checkbox_update = gr.update(interactive=False, value=False)
    tile_strategy_update = gr.update(
        choices=tile_strategy_choices_for_method(method),
        value=effective_tile_strategy,
        interactive=supports_tiling,
    )
    preprocessing_update = (
        gr.update(interactive=True)
        if supports_preprocessing
        else gr.update(interactive=False, value=list(DEFAULT_PREPROCESSING_STEPS))
    )
    black_border_update = (
        gr.update(interactive=True)
        if supports_preprocessing
        else gr.update(interactive=False, value=DEFAULT_IGNORE_BLACK_BORDER)
    )
    tile_control_updates = update_tile_controls(method, effective_tile_strategy)

    return (
        gr.update(visible=(method in ["PP-OCRv5","PP-StructureV3","Auto Routing"])),
        gr.update(visible=(method in ["PaddleVL","Auto Routing"])),
        gr.update(visible=(method == "PP-OCRv5")),
        preprocessing_update,
        black_border_update,
        gr.update(**(tile_checkbox_update | {"visible": show_tiling_options})),
        gr.update(**(tile_strategy_update | {"visible": show_tiling_options})),
        *(gr.update(**(update | {"visible": show_tiling_options and update.get("visible", False)})) for update in tile_control_updates),
        gr.update(visible=(method == "PP-OCRv5")),
    )


def normalize_post_ocr_vl_backend_choice(value):
    """Normalize a post-OCR backend value to the supported UI keys."""
    backend = str(value or DEFAULT_POST_OCR_VL_BACKEND).strip().lower()
    if backend in {"paddle", "paddlevl", "paddle_vl", "paddle-vl"}:
        return "paddlevl"
    if backend in {"chandra", "chandra_ocr", "chandra-ocr"}:
        return "chandra"
    if backend in {
        "both_chandra",
        "chandra_both",
        "paddlevl_chandra",
        "paddlevl+chandra",
        "paddlevl-chandra",
    }:
        return "both_chandra"
    if backend in {"olm", "olmocr", "olm-ocr", "olm_ocr"}:
        return "olmocr"
    if backend == "both":
        return "both"
    return DEFAULT_POST_OCR_VL_BACKEND


def update_post_ocr_vl_backend_controls(backend):
    """Show only the configuration inputs relevant to the selected VL backend."""
    backend = normalize_post_ocr_vl_backend_choice(backend)
    uses_paddlevl = backend in {"paddlevl", "both", "both_chandra"}
    uses_openai_compat = backend in {"olmocr", "both"}
    uses_chandra = backend in {"chandra", "both_chandra"}
    standalone_openai_compat = backend == "olmocr"

    return (
        gr.update(visible=uses_paddlevl),
        gr.update(visible=uses_openai_compat),
        gr.update(visible=uses_chandra),
        gr.update(
            interactive=standalone_openai_compat,
            info=(
                "Only used for the standalone OpenAI-compatible backend."
                if standalone_openai_compat
                else "Ignored unless backend is standalone OpenAI-compatible."
            ),
        ),
    )


def build_paddle_options(
    language,
    use_textline_orientation,
    text_det_thresh,
    text_det_limit_type,
    text_det_limit_side_len,
    text_det_max_side_limit,
    text_det_box_thresh,
    text_det_unclip_ratio,
    text_rec_score_thresh,
    text_recognition_batch_size,
):
    return {
        "lang": language,
        "use_textline_orientation": use_textline_orientation,
        "text_det_thresh": float(
            DEFAULT_TEXT_DET_THRESH if text_det_thresh is None else text_det_thresh
        ),
        "text_det_limit_type": text_det_limit_type or DEFAULT_TEXT_DET_LIMIT_TYPE,
        "text_det_limit_side_len": int(
            DEFAULT_TEXT_DET_LIMIT_SIDE_LEN
            if text_det_limit_side_len is None
            else text_det_limit_side_len
        ),
        "text_det_max_side_limit": int(
            DEFAULT_TEXT_DET_MAX_SIDE_LIMIT
            if text_det_max_side_limit is None
            else text_det_max_side_limit
        ),
        "text_det_box_thresh": float(
            DEFAULT_TEXT_DET_BOX_THRESH
            if text_det_box_thresh is None
            else text_det_box_thresh
        ),
        "text_det_unclip_ratio": float(
            DEFAULT_TEXT_DET_UNCLIP_RATIO
            if text_det_unclip_ratio is None
            else text_det_unclip_ratio
        ),
        "text_rec_score_thresh": float(
            DEFAULT_TEXT_REC_SCORE_THRESH
            if text_rec_score_thresh is None
            else text_rec_score_thresh
        ),
        "text_recognition_batch_size": int(
            DEFAULT_TEXT_RECOGNITION_BATCH_SIZE
            if text_recognition_batch_size is None
            else text_recognition_batch_size
        ),
    }


def build_post_ocr_vl_options(
    post_ocr_vl_enabled,
    post_ocr_vl_backend,
    post_ocr_vl_candidate_mode,
    post_ocr_vl_max_words_enabled,
    post_ocr_vl_max_words,
    post_ocr_vl_padding_px,
    post_ocr_vl_low_confidence_threshold,
    post_ocr_vl_apply_corrections,
    post_ocr_vl_save_tiles,
    post_ocr_vl_group_lines,
    post_ocr_vl_paddlevl_url,
    post_ocr_vl_ollama_openai_url,
    post_ocr_vl_ollama_openai_model,
    post_ocr_vl_chandra_url,
    post_ocr_vl_chandra_model,
):
    return {
        "post_ocr_vl_enabled": bool(post_ocr_vl_enabled),
        "post_ocr_vl_backend": normalize_post_ocr_vl_backend_choice(
            post_ocr_vl_backend
        ),
        "post_ocr_vl_candidate_mode": (
            post_ocr_vl_candidate_mode or DEFAULT_POST_OCR_VL_CANDIDATE_MODE
        ),
        "post_ocr_vl_max_words_enabled": bool(post_ocr_vl_max_words_enabled),
        "post_ocr_vl_max_words": int(
            0
            if not post_ocr_vl_max_words_enabled
            else (
                DEFAULT_POST_OCR_VL_MAX_WORDS
                if post_ocr_vl_max_words is None
                else post_ocr_vl_max_words
            )
        ),
        "post_ocr_vl_padding_px": int(
            DEFAULT_POST_OCR_VL_PADDING_PX
            if post_ocr_vl_padding_px is None
            else post_ocr_vl_padding_px
        ),
        "post_ocr_vl_low_confidence_threshold": float(
            DEFAULT_POST_OCR_VL_LOW_CONFIDENCE_THRESHOLD
            if post_ocr_vl_low_confidence_threshold is None
            else post_ocr_vl_low_confidence_threshold
        ),
        "post_ocr_vl_apply_corrections": bool(post_ocr_vl_apply_corrections),
        "post_ocr_vl_save_tiles": bool(post_ocr_vl_save_tiles),
        "post_ocr_vl_group_lines": bool(post_ocr_vl_group_lines),
        "post_ocr_vl_paddlevl_url": (
            post_ocr_vl_paddlevl_url or DEFAULT_PADDLE_VL_API_URL
        ),
        "post_ocr_vl_ollama_openai_url": (
            post_ocr_vl_ollama_openai_url or DEFAULT_OLLAMA_OPENAI_API_URL
        ),
        "post_ocr_vl_ollama_openai_model": (
            post_ocr_vl_ollama_openai_model or DEFAULT_OLLAMA_OPENAI_MODEL
        ),
        "post_ocr_vl_chandra_url": (
            post_ocr_vl_chandra_url or DEFAULT_CHANDRA_API_URL
        ),
        "post_ocr_vl_chandra_model": (
            post_ocr_vl_chandra_model or DEFAULT_CHANDRA_MODEL
        ),
    }


def submit_config(
    method,
    image_input_folder,
    output_folder,
    language,
    use_doc_orientation_classify,
    use_doc_unwarping,
    use_textline_orientation,
    use_layout_detection,
    generate_text_file,
    clean_punctuation_issues,
    preprocessing_steps,
    ignore_black_border,
    tile_source_images,
    tile_strategy,
    tile_overlap_px,
    tile_doclayout_padding_px,
    tile_doclayout_threshold,
    tile_doclayout_recovery,
    text_det_thresh,
    text_det_limit_type,
    text_det_limit_side_len,
    text_det_max_side_limit,
    text_det_box_thresh,
    text_det_unclip_ratio,
    text_rec_score_thresh,
    text_recognition_batch_size,
    post_ocr_vl_enabled,
    post_ocr_vl_backend,
    post_ocr_vl_candidate_mode,
    post_ocr_vl_max_words_enabled,
    post_ocr_vl_max_words,
    post_ocr_vl_padding_px,
    post_ocr_vl_low_confidence_threshold,
    post_ocr_vl_apply_corrections,
    post_ocr_vl_save_tiles,
    post_ocr_vl_group_lines,
    post_ocr_vl_paddlevl_url,
    post_ocr_vl_ollama_openai_url,
    post_ocr_vl_ollama_openai_model,
    post_ocr_vl_chandra_url,
    post_ocr_vl_chandra_model,
):
    """
    Validate user input and start the OCR processing pipeline.
    """
    if not image_input_folder:
        yield "Please select an image input folder."
        return

    if not output_folder:
        yield "Please select an output folder."
        return

    input_path = Path(image_input_folder)
    output_root = Path(output_folder)
    tile_strategy = coerce_tile_strategy_for_method(method, tile_strategy)
    tile_rows, tile_cols = tile_grid_for_strategy(tile_strategy)
    tile_source_images = bool(tile_source_images) and method != "PP-StructureV3"
    
    opt = {
        "method": method,
        "use_doc_orientation_classify": use_doc_orientation_classify,
        "use_doc_unwarping": use_doc_unwarping,
        "generate_text_file": generate_text_file,
        "clean_punctuation_issues": bool(clean_punctuation_issues),
        "preprocessing_steps": list(preprocessing_steps or []),
        "ignore_black_border": bool(ignore_black_border),
        "tile_source_images": tile_source_images,
        "tile_strategy": tile_strategy,
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
        "tile_overlap_px": (
            int(tile_overlap_px)
            if tile_overlap_px is not None
            else DEFAULT_TILE_OVERLAP_PX
        ),
        "tile_doclayout_padding_px": (
            int(tile_doclayout_padding_px)
            if tile_doclayout_padding_px is not None
            else DEFAULT_DOCLAYOUT_PADDING_PX
        ),
        "tile_doclayout_threshold": (
            float(tile_doclayout_threshold)
            if tile_doclayout_threshold is not None
            else DEFAULT_DOCLAYOUT_THRESHOLD
        ),
        "tile_doclayout_recovery": bool(tile_doclayout_recovery),
    }

    if tile_source_images and method not in LOCAL_OCR_METHODS:
        yield "Source-image tiling is currently available for PP-OCRv5 and PP-StructureV3 only."
        return

    yield "Preparing OCR models... OCR will start after initialization is complete."
    paddle_options = build_paddle_options(
        language,
        use_textline_orientation,
        text_det_thresh,
        text_det_limit_type,
        text_det_limit_side_len,
        text_det_max_side_limit,
        text_det_box_thresh,
        text_det_unclip_ratio,
        text_rec_score_thresh,
        text_recognition_batch_size,
    )
    if method != "PP-OCRv5":
        paddle_options.pop("text_det_max_side_limit", None)

    # Start to run the OCR job based on the selected method and options
    if method in ["PP-StructureV3","PP-OCRv5"]:
        opt.update(paddle_options)
        if method == "PP-OCRv5":
            opt.update(
                build_post_ocr_vl_options(
                    post_ocr_vl_enabled,
                    post_ocr_vl_backend,
                    post_ocr_vl_candidate_mode,
                    post_ocr_vl_max_words_enabled,
                    post_ocr_vl_max_words,
                    post_ocr_vl_padding_px,
                    post_ocr_vl_low_confidence_threshold,
                    post_ocr_vl_apply_corrections,
                    post_ocr_vl_save_tiles,
                post_ocr_vl_group_lines,
                post_ocr_vl_paddlevl_url,
                post_ocr_vl_ollama_openai_url,
                post_ocr_vl_ollama_openai_model,
                post_ocr_vl_chandra_url,
                post_ocr_vl_chandra_model,
            )
        )
        else:
            opt["post_ocr_vl_enabled"] = False
      
        for message in run_ocr_job(input_path, output_root, opt):
            yield message

    elif method == "PaddleVL":
        opt.update({
             "use_layout_detection": use_layout_detection
        })

        for message in run_paddlevl_job(input_path, output_root,opt):
            yield message

    elif method == "Auto Routing":
        opt.update({
            "use_layout_detection": use_layout_detection,
            **paddle_options,
        })
        for message in run_auto_job(input_path, output_root, opt):
            yield message


    yield " OCR Completed, please check pdf and alto files at your output folder"

# Create the main Gradio application interface.
with gr.Blocks(css=APP_CSS) as demo:

    # Create a fixed header section for the application title
    # and OCR method selection.
    with gr.Column(elem_id="fixed-header"):
        gr.Markdown("## CRKN OCR Engine")

        method_radio = gr.Radio(
            choices=["PP-OCRv5","PP-StructureV3", "PaddleVL", "Auto Routing"],
            label="Select Processing Method",
            value="PP-OCRv5"
        )

    # Display the folder selection section title.
    gr.Markdown("### Folder Selection")

    # Input folder selection components.
    with gr.Row():
        image_input_folder = gr.Textbox(label="Image Input Folder")
        image_input_btn = gr.Button("Browse Input Folder")

    # Output folder selection components.
    with gr.Row():
        output_folder = gr.Textbox(label="Output Folder")
        output_folder_btn = gr.Button("Browse Output Folder")

    # Common OCR options shared across methods.
    with gr.Column(visible=True) as common_options:
        use_doc_orientation_classify = gr.Checkbox(
        label='Use document orientation classification during inference.',
        value=False,
        )

        use_doc_unwarping = gr.Checkbox(
            label='Use document image unwarping during inference.',
            value=False
        )

        generate_text_file = gr.Checkbox(
            label='Generate a text file with OCR results for each image.',
            value=False
        )

        clean_punctuation_issues = gr.Checkbox(
            label="Clean punctuation artifacts in TXT, PDF, and ALTO output.",
            value=DEFAULT_CLEAN_PUNCTUATION_ISSUES,
            info=(
                "Applies a conservative punctuation cleanup based on the OCR "
                "comparison script before output files are generated."
            ),
        )

        preprocessing_steps = gr.CheckboxGroup(
            choices=PREPROCESSING_STEP_CHOICES,
            value=list(DEFAULT_PREPROCESSING_STEPS),
            label="Image Preprocessing Steps",
            info=(
                "Applied before local OCR and tiling. Active steps save "
                "a preprocessed PNG beside the OCR outputs."
            ),
        )

        ignore_black_border = gr.Checkbox(
            label="Crop black border before OCR.",
            value=DEFAULT_IGNORE_BLACK_BORDER,
            info=(
                "Runs OCR on the detected page crop, then offsets OCR geometry "
                "back into the original image coordinates."
            ),
        )

        tile_source_images = gr.Checkbox(
            value=True,
            label="Enable Tiling",
        )

        tile_strategy = gr.Dropdown(
            choices=FULL_TILE_STRATEGY_CHOICES,
            value=DEFAULT_TILE_STRATEGY,
            label="Tile Strategy",
            info=(
                "Use fixed tiles, PaddleX PP-DocLayoutV3 regions, "
                "or PP-DocLayoutV3 inside each fixed grid tile."
            ),
        )

        tile_overlap_px = gr.Slider(
            minimum=0,
            maximum=256,
            value=DEFAULT_TILE_OVERLAP_PX,
            step=4,
            label="Tile Overlap (px)",
            info="Overlap for fixed grid tiles before deduping.",
        )

        tile_doclayout_padding_px = gr.Slider(
            minimum=0,
            maximum=128,
            value=DEFAULT_DOCLAYOUT_PADDING_PX,
            step=4,
            label="DocLayout Padding (px)",
        )

        tile_doclayout_threshold = gr.Slider(
            minimum=0.01,
            maximum=1.0,
            value=DEFAULT_DOCLAYOUT_THRESHOLD,
            step=0.01,
            label="DocLayout Threshold",
        )

        tile_doclayout_recovery = gr.Checkbox(
            label="Use DocLayout recovery tile.",
            value=True,
        )

    # Optional PP-OCRv5 post-processing with word crops sent to a VL backend.
    with gr.Column(visible=True) as post_ocr_vl_options:
        post_ocr_vl_enabled = gr.Checkbox(
            value=True,
            label="Enable Post-OCR VL",
        )

        post_ocr_vl_backend = gr.Dropdown(
            choices=POST_OCR_VL_BACKEND_CHOICES,
            value=normalize_post_ocr_vl_backend_choice(DEFAULT_POST_OCR_VL_BACKEND),
            label="Post-OCR VL Backend",
        )

        post_ocr_vl_candidate_mode = gr.Dropdown(
            choices=[
                ("Suspicious words", "suspicious"),
                ("Low-confidence lines", "low_confidence"),
                ("All words in reading order", "all"),
            ],
            value=DEFAULT_POST_OCR_VL_CANDIDATE_MODE,
            label="Words to Check",
        )

        post_ocr_vl_max_words_enabled = gr.Checkbox(
            value=False,
            label="Enable Max Words Check",
        )

        post_ocr_vl_max_words = gr.Number(
            value=DEFAULT_POST_OCR_VL_MAX_WORDS,
            label="Max Words to Check",
            interactive=False,
        )

        post_ocr_vl_padding_px = gr.Slider(
            minimum=0,
            maximum=64,
            value=DEFAULT_POST_OCR_VL_PADDING_PX,
            step=1,
            label="Word Tile Padding (px)",
        )

        post_ocr_vl_low_confidence_threshold = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=DEFAULT_POST_OCR_VL_LOW_CONFIDENCE_THRESHOLD,
            step=0.01,
            label="Low Confidence Threshold",
        )

        post_ocr_vl_apply_corrections = gr.Checkbox(
            label="Apply accepted VL corrections to OCR text.",
            value=DEFAULT_POST_OCR_VL_APPLY_CORRECTIONS,
        )

        post_ocr_vl_save_tiles = gr.Checkbox(
            value=False,
            label="Save post-OCR VL word tiles.",
        )

        post_ocr_vl_group_lines = gr.Checkbox(
            value=DEFAULT_POST_OCR_VL_GROUP_LINES,
            label="Group same-line suspicious words for OpenAI-compatible backend.",
            interactive=normalize_post_ocr_vl_backend_choice(DEFAULT_POST_OCR_VL_BACKEND) == "olmocr",
            info="Only used for the standalone OpenAI-compatible backend.",
        )

        with gr.Column(
            visible=normalize_post_ocr_vl_backend_choice(DEFAULT_POST_OCR_VL_BACKEND)
            in {"paddlevl", "both", "both_chandra"}
        ) as post_ocr_vl_paddlevl_options:
            post_ocr_vl_paddlevl_url = gr.Textbox(
                label="PaddleVL API URL",
                value=DEFAULT_PADDLE_VL_API_URL,
                info="Used when backend includes PaddleVL.",
            )

        with gr.Column(
            visible=normalize_post_ocr_vl_backend_choice(DEFAULT_POST_OCR_VL_BACKEND)
            in {"olmocr", "both"}
        ) as post_ocr_vl_openai_options:
            post_ocr_vl_ollama_openai_url = gr.Textbox(
                label="OpenAI-Compatible API URL",
                value=DEFAULT_OLLAMA_OPENAI_API_URL,
                info="Used when backend includes the generic OpenAI-compatible path; requests go to /chat/completions.",
            )

            post_ocr_vl_ollama_openai_model = gr.Dropdown(
                label="OpenAI-Compatible Model",
                choices=[
                    ("olmOCR 2", "richardyoung/olmocr2"),
                    ("DeepSeek OCR 3B", "deepseek-ocr:3b"),
                ],
                value=DEFAULT_OLLAMA_OPENAI_MODEL,
                allow_custom_value=True,
                info="Any non-Chandra model served by the OpenAI-compatible endpoint, for example richardyoung/olmocr2 or deepseek-ocr:3b.",
            )

        with gr.Column(
            visible=normalize_post_ocr_vl_backend_choice(DEFAULT_POST_OCR_VL_BACKEND)
            in {"chandra", "both_chandra"}
        ) as post_ocr_vl_chandra_options:
            post_ocr_vl_chandra_url = gr.Textbox(
                label="Chandra API URL",
                value=DEFAULT_CHANDRA_API_URL,
                info="Used when backend includes Chandra; requests go to /chat/completions.",
            )

            post_ocr_vl_chandra_model = gr.Dropdown(
                label="Chandra Model",
                choices=[
                    ("Chandra OCR 2 (local GGUF)", "mradermacher/chandra-ocr-2-GGUF:Q4_K_M"),
                    ("Chandra OCR 2 (Ollama)", "fredrezones55/chandra-ocr-2"),
                ],
                value=DEFAULT_CHANDRA_MODEL,
                allow_custom_value=True,
                info="Any Chandra model served by the selected Chandra endpoint.",
            )

    # PaddleVL-specific OCR options.
    with gr.Column(visible=False) as paddlevl_options:
        use_layout_detection = gr.Checkbox(
            label='Use layout detection during inference.',
            value=False
        )

    # PaddleOCR-specific OCR options.
    with gr.Column(visible=True) as paddleocr_options:

        # Language selection dropdown.
        language_dropdown = gr.Dropdown(
            choices=LANGUAGE_CHOICES,
            label="Select Language",
            value=None
        )
        
        # Text detection confidence threshold slider.
        text_det_thresh = gr.Slider(
        minimum=0.0,
        maximum=1.0,
        value=DEFAULT_TEXT_DET_THRESH,
        step=0.01,
        label="Text Detection Threshold"
        )

        text_det_limit_type = gr.Dropdown(
            choices=[
                ("Max side", "max"),
                ("Min side", "min"),
            ],
            value=DEFAULT_TEXT_DET_LIMIT_TYPE,
            label="Text Detection Limit Type",
        )

        text_det_limit_side_len = gr.Slider(
            minimum=512,
            maximum=10000,
            value=DEFAULT_TEXT_DET_LIMIT_SIDE_LEN,
            step=32,
            label="Text Detection Limit Side Length",
        )

        text_det_max_side_limit = gr.Slider(
            minimum=1000,
            maximum=20000,
            value=DEFAULT_TEXT_DET_MAX_SIDE_LIMIT,
            step=1000,
            label="Max Side Limit",
        )

        text_det_box_thresh = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=DEFAULT_TEXT_DET_BOX_THRESH,
            step=0.01,
            label="Text Detection Box Threshold",
        )

        text_det_unclip_ratio = gr.Slider(
            minimum=1.0,
            maximum=4.0,
            value=DEFAULT_TEXT_DET_UNCLIP_RATIO,
            step=0.1,
            label="Text Detection Unclip Ratio",
        )

        text_rec_score_thresh = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=DEFAULT_TEXT_REC_SCORE_THRESH,
            step=0.01,
            label="Text Recognition Score Threshold",
        )

        text_recognition_batch_size = gr.Slider(
            minimum=1,
            maximum=128,
            value=DEFAULT_TEXT_RECOGNITION_BATCH_SIZE,
            step=1,
            label="Text Recognition Batch Size",
        )

        # Enable text line orientation classification.
        use_textline_orientation = gr.Checkbox(
            label='Use textline orientation classification during inference.',
            value=False
        )
        
    # OCR action buttons.      
    with gr.Row():
        submit_btn = gr.Button("Run OCR")
        stop_btn = gr.Button("Stop OCR",interactive=False)

    # Display OCR processing messages and results.
    output = gr.Textbox(label="OCR Result", lines=20)

    # Store UI components for dynamic state updates.
    ui_components = [
        submit_btn,
        method_radio,
        image_input_folder,
        image_input_btn,
        output_folder,
        output_folder_btn,
        language_dropdown,
        use_doc_orientation_classify,
        use_doc_unwarping,
        use_textline_orientation,
        use_layout_detection,
        generate_text_file,
        clean_punctuation_issues,
        preprocessing_steps,
        ignore_black_border,
        tile_source_images,
        tile_strategy,
        tile_overlap_px,
        tile_doclayout_padding_px,
        tile_doclayout_threshold,
        tile_doclayout_recovery,
        text_det_thresh,
        text_det_limit_type,
        text_det_limit_side_len,
        text_det_max_side_limit,
        text_det_box_thresh,
        text_det_unclip_ratio,
        text_rec_score_thresh,
        text_recognition_batch_size,
        post_ocr_vl_enabled,
        post_ocr_vl_backend,
        post_ocr_vl_candidate_mode,
        post_ocr_vl_max_words_enabled,
        post_ocr_vl_max_words,
        post_ocr_vl_padding_px,
        post_ocr_vl_low_confidence_threshold,
        post_ocr_vl_apply_corrections,
        post_ocr_vl_save_tiles,
        post_ocr_vl_group_lines,
        post_ocr_vl_paddlevl_url,
        post_ocr_vl_ollama_openai_url,
        post_ocr_vl_ollama_openai_model,
        post_ocr_vl_chandra_url,
        post_ocr_vl_chandra_model,
    ]

    # Update visible OCR option sections
    # when the processing method changes.
    method_radio.change(
        fn=update_options,
        inputs=[method_radio, tile_strategy],
        outputs=[
            paddleocr_options,
            paddlevl_options,
            post_ocr_vl_options,
            preprocessing_steps,
            ignore_black_border,
            tile_source_images,
            tile_strategy,
            tile_overlap_px,
            tile_doclayout_padding_px,
            tile_doclayout_threshold,
            tile_doclayout_recovery,
            text_det_max_side_limit,
        ]
    )

    tile_strategy.change(
        fn=update_tile_controls,
        inputs=[method_radio, tile_strategy],
        outputs=[
            tile_overlap_px,
            tile_doclayout_padding_px,
            tile_doclayout_threshold,
            tile_doclayout_recovery,
        ],
    )

    # Handle post-OCR VL max words enable/disable
    post_ocr_vl_max_words_enabled.change(
        fn=lambda enabled: gr.update(interactive=enabled),
        inputs=post_ocr_vl_max_words_enabled,
        outputs=post_ocr_vl_max_words,
    )

    post_ocr_vl_backend.change(
        fn=update_post_ocr_vl_backend_controls,
        inputs=post_ocr_vl_backend,
        outputs=[
            post_ocr_vl_paddlevl_options,
            post_ocr_vl_openai_options,
            post_ocr_vl_chandra_options,
            post_ocr_vl_group_lines,
        ],
    )



    # Open a folder selection dialog for the input folder.
    image_input_btn.click(
        fn=choose_folder,
        inputs=image_input_folder,
        outputs=image_input_folder
    )

    # Open a folder selection dialog for the output folder.
    output_folder_btn.click(
        fn=choose_folder,
        inputs=output_folder,
        outputs=output_folder
    )


    def before_run():
        """
        Disable UI components before OCR processing starts.
        """
        return [
            *[gr.update(interactive=False) for _ in ui_components],
            gr.update(interactive=True)
        ]

    def after_run(post_ocr_vl_max_words_enabled_value, post_ocr_vl_backend_value):
        """
        Re-enable UI components after OCR processing completes.
        """
        updates = []
        normalized_backend = normalize_post_ocr_vl_backend_choice(
            post_ocr_vl_backend_value
        )
        for component in ui_components:
            if component is post_ocr_vl_max_words:
                updates.append(
                    gr.update(interactive=bool(post_ocr_vl_max_words_enabled_value))
                )
            elif component is post_ocr_vl_group_lines:
                updates.append(
                    gr.update(interactive=(normalized_backend == "olmocr"))
                )
            else:
                updates.append(gr.update(interactive=True))

        return [
            *updates,
            gr.update(interactive=False)
        ]

    def stop_clicked():
        """
        Stop OCR processing after the current image is completed.
        """
        stop_ocr()
        return "Stopping OCR after current image finishes..."


    # Disable UI components before starting OCR processing,
    # then run the OCR pipeline and restore the UI state after completion.
    submit_btn.click(
    fn=before_run,
    inputs=None,
    outputs=[*ui_components, stop_btn]
).then(
    fn=submit_config,
    inputs=[
        method_radio,
        image_input_folder,
        output_folder,
        language_dropdown,
        use_doc_orientation_classify,
        use_doc_unwarping,
        use_textline_orientation,
        use_layout_detection,
        generate_text_file,
        clean_punctuation_issues,
        preprocessing_steps,
        ignore_black_border,
        tile_source_images,
        tile_strategy,
        tile_overlap_px,
        tile_doclayout_padding_px,
        tile_doclayout_threshold,
        tile_doclayout_recovery,
        text_det_thresh,
        text_det_limit_type,
        text_det_limit_side_len,
        text_det_max_side_limit,
        text_det_box_thresh,
        text_det_unclip_ratio,
        text_rec_score_thresh,
        text_recognition_batch_size,
        post_ocr_vl_enabled,
        post_ocr_vl_backend,
        post_ocr_vl_candidate_mode,
        post_ocr_vl_max_words_enabled,
        post_ocr_vl_max_words,
        post_ocr_vl_padding_px,
        post_ocr_vl_low_confidence_threshold,
        post_ocr_vl_apply_corrections,
        post_ocr_vl_save_tiles,
        post_ocr_vl_group_lines,
        post_ocr_vl_paddlevl_url,
        post_ocr_vl_ollama_openai_url,
        post_ocr_vl_ollama_openai_model,
        post_ocr_vl_chandra_url,
        post_ocr_vl_chandra_model,
    ],
    outputs=output
    ).then(
    fn=after_run,
    inputs=[post_ocr_vl_max_words_enabled, post_ocr_vl_backend],
    outputs=[*ui_components, stop_btn]
    )

    stop_btn.click(
        fn=stop_clicked,
        inputs=None,
        outputs=output
    ).then(
        fn=lambda: gr.update(interactive=False),
        inputs=None,
        outputs=stop_btn
    )

def launch_app():
    """
    Launch the Gradio application.
    """
    demo.launch(
    server_name=os.getenv("OCR_TOOL_HOST", "0.0.0.0"),
    server_port=int(os.getenv("OCR_TOOL_PORT", "7860")),
    )


if __name__ == "__main__":
    launch_app()
