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
from services.ocr_olm import run_olmocr_job
from services.ocr_dots import run_dotsocr_job
from services.ocr_indra import run_indra_job
from services.ocr_defaults import (
    DEFAULT_CLEAN_PUNCTUATION_ISSUES,
    DEFAULT_DOCLAYOUT_PADDING_PX,
    DEFAULT_DOTSOCR_API_URL,
    DEFAULT_DOTSOCR_MODEL,
    DEFAULT_IGNORE_BLACK_BORDER,
    DEFAULT_OCR_VERSION,
    DEFAULT_OLMOCR_API_URL,
    DEFAULT_OLMOCR_MODEL,
    DEFAULT_PADDLE_VL_API_URL,
    DEFAULT_POST_OCR_VL_APPLY_CORRECTIONS,
    DEFAULT_POST_OCR_VL_BACKEND,
    DEFAULT_POST_OCR_VL_BATCH_SIZE,
    DEFAULT_POST_OCR_VL_CANDIDATE_MODE,
    DEFAULT_POST_OCR_VL_CONCURRENCY,
    DEFAULT_POST_OCR_VL_DOTSOCR_MODEL,
    DEFAULT_POST_OCR_VL_DOTSOCR_URL,
    DEFAULT_POST_OCR_VL_ENABLED,
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
    DEFAULT_TILE_SOURCE_IMAGES,
    DEFAULT_TILE_STRATEGY,
)
from services.tile_ocr import tile_grid_for_strategy, normalize_tile_strategy

PADDLE_OCR_METHODS = {"PP-OCRv6"}
OCR_METHOD_CHOICES = [
    ("PaddleOCR (PP-OCRv6)", "PP-OCRv6"),
    ("Indra OCR (Fast CPU)", "IndraOCR"),
    ("PaddleVL", "PaddleVL"),
    ("OLM OCR", "OLMOCR"),
    ("Dots OCR (BHL #1)", "DotsOCR"),
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

BEST_PREPROCESSING_STEPS = list(DEFAULT_PREPROCESSING_STEPS)
BEST_TILE_STRATEGY = "auto"

TILE_STRATEGY_CHOICES = [
    ("Auto-Detect (3x2 on broadsheets, full-image on books)", "auto"),
    ("3x2 Grid (Always tile 6 quadrants)", "grid_3x2"),
    ("No Tiling (Full-image single pass)", "none"),
    ("3x1 Grid (3 horizontal tiles)", "grid"),
    ("PP-DocLayout-V3 (Adaptive block crops)", "doclayout_v3"),
]
DEFAULT_TILE_STRATEGY_UI = "auto"

PRE_PROCESS_INFO = (
    "Crops detected scan borders, auto-deskews, and flattens uneven lighting. "
    "Automatically detects degraded microfilm to apply adaptive CLAHE and stroke reconnection, "
    "suppresses table grid borders to preserve tabular cell text, and applies adaptive upscaling "
    "for micro-print when needed."
)
POST_PROCESS_INFO = (
    "Cleans conservative punctuation artifacts, consolidates legal abbreviations (&c.), "
    "fixes archaic long-s print, and verifies suspicious text or faded lines against "
    "Dots OCR VLM and historical Canadian dictionaries."
)

# Define the list of supported languages for the dropdown menu,
# prioritized by the official Canadiana 54-language census and Indigenous Canadian archives.
LANGUAGE_CHOICES = [
    # Automatic Detection
    ("Auto-detect Script & Collection (Recommended)", "auto"),
    # Top Canadiana Collections (Census Volume)
    ("English (287,983 docs)", "en"),
    ("French / Français (87,877 docs)", "fr"),
    ("Traditional Chinese / 繁體中文 - e.g. 大漢公報 (23,006 docs)", "chinese_cht"),
    ("German / Deutsch - e.g. Berliner Journal (5,540 docs)", "de"),
    ("Yiddish / ייִדיש (5,516 docs)", "yi"),
    ("Ukrainian / Українська - Prairie Press (3,245 docs)", "uk"),
    ("Italian / Italiano - e.g. Il Cittadino Canadese (1,747 docs)", "it"),
    ("Finnish / Suomi (606 docs)", "fi"),
    ("Latin / Latina - Church & Parish Records (546 docs)", "la"),
    ("Icelandic / Íslenska - Gimli Colony (459 docs)", "is"),
    ("Hungarian / Magyar (383 docs)", "hu"),
    # Indigenous Languages of Canada (Spellcheck safely bypassed)
    ("Cree / ᓀᐦᐃᔭᐍᐏᐣ - Syllabics (257 docs)", "cr"),
    ("Chinook Jargon (210 docs)", "chn"),
    ("Mi'kmaq (86 docs)", "mic"),
    ("Ojibwa / Anishinaabemowin (80 docs)", "oj"),
    ("Mohawk / Kanienʼkéha (50 docs)", "moh"),
    ("Inuktitut / ᐃᓄᒃᑎᑐᑦ (20 docs)", "iu"),
    ("Tsimshian / Sm'algya̱x (11 docs)", "tsi"),
    ("Siksika / Blackfoot (3 docs)", "bla"),
    ("Chipewyan / Dënesųłı̨në (3 docs)", "chp"),
    ("Delaware / Lenape (3 docs)", "del"),
    ("Haida / X̱aad Kíl (2 docs)", "hai"),
    ("Slave / Athapascan (1 doc)", "den"),
    # Other Canadiana Archive & Diaspora Languages
    ("Dutch / Nederlands (108 docs)", "nl"),
    ("Spanish / Español (76 docs)", "es"),
    ("Swedish / Svenska (32 docs)", "sv"),
    ("Greek, Modern / Ελληνικά (31 docs)", "el"),
    ("Russian / Русский - Doukhobor Records (28 docs)", "ru"),
    ("Gaelic / Scottish Gaelic / Gàidhlig (23 docs)", "gd"),
    ("Danish / Dansk (17 docs)", "da"),
    ("Norwegian / Norsk (13 docs)", "no"),
    ("Hebrew / עברית (12 docs)", "he"),
    ("Polish / Polski (8 docs)", "pl"),
    ("Irish / Gaeilge (4 docs)", "ga"),
    ("Hindi / हिन्दी (4 docs)", "hi"),
    ("Welsh / Cymraeg (3 docs)", "cy"),
    ("Portuguese / Português (3 docs)", "pt"),
    ("Bengali / বাংলা (2 docs)", "bn"),
    ("Romanian / Română (1 doc)", "ro"),
    # Additional Asian & International Scripts Supported by PP-OCRv6
    ("Simplified Chinese / 简体中文", "ch"),
    ("Japanese / 日本語", "japan"),
    ("Korean / 한국어", "korean"),
    ("Arabic / العربية", "ar"),
    ("Persian / Farsi / فارسی", "fa"),
    ("Vietnamese / Tiếng Việt", "vi"),
    ("Turkish / Türkçe", "tr"),
    ("Czech / Čeština", "cs"),
    ("Slovak / Slovenčina", "sk"),
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


def build_post_ocr_vl_options(post_process_enabled, arbiter_backend=None):
    backend = arbiter_backend or DEFAULT_POST_OCR_VL_BACKEND
    return {
        "post_ocr_vl_enabled": bool(post_process_enabled)
        and bool(DEFAULT_POST_OCR_VL_ENABLED),
        "post_ocr_vl_backend": backend,
        "post_ocr_vl_candidate_mode": DEFAULT_POST_OCR_VL_CANDIDATE_MODE,
        "post_ocr_vl_max_words_enabled": False,
        "post_ocr_vl_max_words": int(DEFAULT_POST_OCR_VL_MAX_WORDS),
        "post_ocr_vl_padding_px": int(DEFAULT_POST_OCR_VL_PADDING_PX),
        "post_ocr_vl_low_confidence_threshold": float(
            DEFAULT_POST_OCR_VL_LOW_CONFIDENCE_THRESHOLD
        ),
        "post_ocr_vl_apply_corrections": bool(
            DEFAULT_POST_OCR_VL_APPLY_CORRECTIONS
        ),
        "post_ocr_vl_save_tiles": bool(DEFAULT_POST_OCR_VL_SAVE_TILES),
        "post_ocr_vl_paddlevl_url": DEFAULT_PADDLE_VL_API_URL,
        "post_ocr_vl_dotsocr_url": DEFAULT_POST_OCR_VL_DOTSOCR_URL,
        "post_ocr_vl_dotsocr_model": DEFAULT_POST_OCR_VL_DOTSOCR_MODEL,
        "post_ocr_vl_concurrency": int(DEFAULT_POST_OCR_VL_CONCURRENCY),
        "post_ocr_vl_batch_size": int(DEFAULT_POST_OCR_VL_BATCH_SIZE),
    }


def processing_supported(method):
    """Return whether the selected visible method supports pre/post processing."""
    return method in PADDLE_OCR_METHODS


def update_options_for_method(method, post_val=True):
    """Update visibility and interactivity of options based on selected OCR method.

    Only shows options that are relevant to the selected model:
    - PP-OCRv6: language, tiling, pre-process, post-process, (post-ocr arbiter if post-process checked)
    - IndraOCR: language (for ALTO metadata)
    - PaddleVL: language
    - OLMOCR & DotsOCR: vision-prompt based, no language flag
    - All methods: TXT output, HTML results dashboard
    """
    method = method or "PP-OCRv6"
    is_paddle = method in PADDLE_OCR_METHODS
    is_vl_only = method in {"OLMOCR", "DotsOCR"}

    lang_visible = not is_vl_only
    tile_visible = is_paddle
    pre_visible = is_paddle
    post_visible = is_paddle
    backend_visible = is_paddle and bool(post_val)

    return [
        gr.update(visible=lang_visible),
        gr.update(visible=tile_visible),
        gr.update(visible=pre_visible, interactive=is_paddle),
        gr.update(visible=post_visible, interactive=is_paddle),
        gr.update(visible=backend_visible, interactive=backend_visible),
    ]


def submit_config(
    method,
    image_input_folder,
    output_folder,
    language,
    generate_text_file,
    pre_process_enabled,
    post_process_enabled,
    post_ocr_backend=None,
    tile_strategy="auto",
    generate_results_dashboard=True,
):
    """
    Validate user input and start the OCR processing pipeline.
    """
    method = method or "PP-OCRv6"
    process_enabled = processing_supported(method)
    pre_process_enabled = bool(pre_process_enabled) and process_enabled
    post_process_enabled = bool(post_process_enabled) and process_enabled

    if not image_input_folder:
        yield "Please select an image input folder."
        return

    if not output_folder:
        yield "Please select an output folder."
        return

    input_path = Path(image_input_folder)
    output_root = Path(output_folder)
    effective_tile_strat = normalize_tile_strategy(tile_strategy or BEST_TILE_STRATEGY)
    tile_rows, tile_cols = tile_grid_for_strategy(effective_tile_strat)
    tile_source_images = (
        (effective_tile_strat != "none") if method in PADDLE_OCR_METHODS else False
    )
    
    opt = {
        "method": method,
        "ocr_version": DEFAULT_OCR_VERSION,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "generate_text_file": bool(generate_text_file),
        "generate_results_dashboard": bool(generate_results_dashboard),
        "clean_punctuation_issues": bool(post_process_enabled)
        and bool(DEFAULT_CLEAN_PUNCTUATION_ISSUES),
        "preprocessing_steps": (
            list(BEST_PREPROCESSING_STEPS) if pre_process_enabled else []
        ),
        "ignore_black_border": bool(pre_process_enabled)
        and bool(DEFAULT_IGNORE_BLACK_BORDER),
        "tile_source_images": tile_source_images,
        "tile_strategy": effective_tile_strat,
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
        "tile_overlap_px": int(DEFAULT_TILE_OVERLAP_PX),
        "tile_doclayout_padding_px": int(DEFAULT_DOCLAYOUT_PADDING_PX),
        "tile_doclayout_threshold": float(DEFAULT_DOCLAYOUT_THRESHOLD),
        "tile_doclayout_recovery": True,
        "use_layout_detection": method == "PaddleVL",
        "olmocr_url": DEFAULT_OLMOCR_API_URL,
        "olmocr_model": DEFAULT_OLMOCR_MODEL,
        "olmocr_max_tokens": 8192,
        "dotsocr_url": DEFAULT_DOTSOCR_API_URL,
        "dotsocr_model": DEFAULT_DOTSOCR_MODEL,
        "dotsocr_max_tokens": 8192,
    }

    yield "Preparing OCR models... OCR will start after initialization is complete."
    paddle_options = build_paddle_options(
        language,
        False,
        DEFAULT_TEXT_DET_THRESH,
        DEFAULT_TEXT_DET_LIMIT_TYPE,
        DEFAULT_TEXT_DET_LIMIT_SIDE_LEN,
        DEFAULT_TEXT_DET_MAX_SIDE_LIMIT,
        DEFAULT_TEXT_DET_BOX_THRESH,
        DEFAULT_TEXT_DET_UNCLIP_RATIO,
        DEFAULT_TEXT_REC_SCORE_THRESH,
        DEFAULT_TEXT_RECOGNITION_BATCH_SIZE,
    )
    if method not in PADDLE_OCR_METHODS:
        paddle_options.pop("text_det_max_side_limit", None)

    opt.update(paddle_options)
    opt.update(build_post_ocr_vl_options(post_process_enabled, post_ocr_backend))
    if method in PADDLE_OCR_METHODS:
        iterator = run_ocr_job(input_path, output_root, opt)
    elif method == "PaddleVL":
        iterator = run_paddlevl_job(input_path, output_root, opt)
    elif method == "OLMOCR":
        iterator = run_olmocr_job(input_path, output_root, opt)
    elif method == "DotsOCR":
        iterator = run_dotsocr_job(input_path, output_root, opt)
    elif method == "IndraOCR":
        iterator = run_indra_job(input_path, output_root, opt)
    else:
        yield f"Unsupported OCR method: {method}"
        return

    last_message = ""
    for message in iterator:
        last_message = message
        yield message

    if not last_message:
        yield "OCR Completed, please check pdf and alto files at your output folder."

# Create the main Gradio application interface.
with gr.Blocks() as demo:

    # Create a fixed header section for the application title and method choice.
    with gr.Column(elem_id="fixed-header"):
        gr.Markdown("## CRKN OCR Engine")
        method_radio = gr.Radio(
            choices=OCR_METHOD_CHOICES,
            label="OCR method",
            value="PP-OCRv6",
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

    # Simplified OCR options. The detailed tuning values are fixed from the
    # evaluation-backed recipe in submit_config.
    with gr.Column(visible=True):
        language_dropdown = gr.Dropdown(
            choices=LANGUAGE_CHOICES,
            label="Select Language",
            value="auto",
        )

        tile_strategy_dropdown = gr.Dropdown(
            choices=TILE_STRATEGY_CHOICES,
            value=DEFAULT_TILE_STRATEGY_UI,
            label="Image Tiling Strategy",
            info=(
                "Auto-Detect (Recommended) automatically applies 3x2 quadrant tiling for large broadsheets (30+ MP) "
                "to maximize fine-print word recall, and fast full-image pass for standard books/pamphlets."
            ),
        )

        generate_text_file = gr.Checkbox(
            label="Generate TXT instead of PDF/ALTO.",
            value=False,
            info=(
                "When enabled, writes one TXT file per image and skips "
                "searchable PDF and ALTO generation."
            ),
        )

        generate_results_dashboard = gr.Checkbox(
            label="Generate HTML results dashboard",
            value=True,
            info=(
                "When enabled, builds an interactive standalone HTML results dashboard "
                "with bounding boxes color-coded by confidence, word counts, and navigation, "
                "and opens it in a new browser tab upon completion."
            ),
        )

        pre_process = gr.Checkbox(
            label="pre process",
            value=True,
            info=PRE_PROCESS_INFO,
        )

        post_process = gr.Checkbox(
            label="post process",
            value=True,
            info=POST_PROCESS_INFO,
        )

        post_ocr_backend = gr.Dropdown(
            choices=[
                ("Dots OCR (BHL #1 - Recommended)", "dots_ocr"),
                ("PaddleVL", "paddlevl"),
            ],
            value=DEFAULT_POST_OCR_VL_BACKEND,
            label="Post-OCR VLM Arbiter",
            info="Vision-language model used for word-level arbitration and spellcheck verification.",
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
        image_input_folder,
        image_input_btn,
        output_folder,
        output_folder_btn,
        language_dropdown,
        tile_strategy_dropdown,
        method_radio,
        generate_text_file,
        generate_results_dashboard,
        pre_process,
        post_process,
        post_ocr_backend,
    ]

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

    method_radio.change(
        fn=update_options_for_method,
        inputs=[method_radio, post_process],
        outputs=[
            language_dropdown,
            tile_strategy_dropdown,
            pre_process,
            post_process,
            post_ocr_backend,
        ],
    )

    post_process.change(
        fn=lambda post_val, method_val: gr.update(
            visible=bool(post_val) and processing_supported(method_val),
            interactive=bool(post_val) and processing_supported(method_val),
        ),
        inputs=[post_process, method_radio],
        outputs=[post_ocr_backend],
    )


    def before_run():
        """
        Disable UI components before OCR processing starts.
        """
        return [
            *[gr.update(interactive=False) for _ in ui_components],
            gr.update(interactive=True)
        ]

    def after_run(method, pre_value, post_value):
        """
        Re-enable UI components after OCR processing completes while preserving model-specific visibility.
        """
        lang_up, tile_up, pre_up, post_up, backend_up = update_options_for_method(
            method, post_value
        )
        return [
            gr.update(interactive=True),                              # submit_btn
            gr.update(interactive=True),                              # image_input_folder
            gr.update(interactive=True),                              # image_input_btn
            gr.update(interactive=True),                              # output_folder
            gr.update(interactive=True),                              # output_folder_btn
            gr.update(interactive=True, visible=lang_up["visible"]),  # language_dropdown
            gr.update(interactive=True, visible=tile_up["visible"]),  # tile_strategy_dropdown
            gr.update(interactive=True),                              # method_radio
            gr.update(interactive=True),                              # generate_text_file
            gr.update(interactive=True),                              # generate_results_dashboard
            pre_up,                                                   # pre_process
            post_up,                                                  # post_process
            backend_up,                                               # post_ocr_backend
            gr.update(interactive=False),                             # stop_btn
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
        generate_text_file,
        pre_process,
        post_process,
        post_ocr_backend,
        tile_strategy_dropdown,
        generate_results_dashboard,
    ],
    outputs=output
    ).then(
    fn=after_run,
    inputs=[method_radio, pre_process, post_process],
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
    css=APP_CSS,
    )


if __name__ == "__main__":
    launch_app()
