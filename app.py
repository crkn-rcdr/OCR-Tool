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
from services.ocr_defaults import (
    DEFAULT_CLEAN_PUNCTUATION_ISSUES,
    DEFAULT_DOCLAYOUT_PADDING_PX,
    DEFAULT_IGNORE_BLACK_BORDER,
    DEFAULT_OCR_VERSION,
    DEFAULT_OLMOCR_API_URL,
    DEFAULT_OLMOCR_MODEL,
    DEFAULT_PADDLE_VL_API_URL,
    DEFAULT_POST_OCR_VL_APPLY_CORRECTIONS,
    DEFAULT_POST_OCR_VL_CANDIDATE_MODE,
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
)
from services.tile_ocr import tile_grid_for_strategy

PADDLE_OCR_METHODS = {"PP-OCRv6"}
OCR_METHOD_CHOICES = [
    ("PaddleOCR (PP-OCRv6)", "PP-OCRv6"),
    ("PaddleVL", "PaddleVL"),
    ("OLM OCR", "OLMOCR"),
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
BEST_TILE_STRATEGY = "grid_3x2_doclayout_v3"

PRE_PROCESS_INFO = (
    "Crops the detected black scan border, runs OCR on that page crop, "
    "offsets OCR geometry back to the original image coordinates, and applies "
    "background flattening to even newspaper page shading. The OCR-facing PNG "
    "is saved under _preprocessed/."
)
POST_PROCESS_INFO = (
    "Cleans conservative punctuation artifacts in the saved OCR JSON. For "
    "PP-OCRv6, it also sends suspicious word crops to PaddleVL at "
    f"{DEFAULT_PADDLE_VL_API_URL}, applies only accepted corrections, and "
    "writes the post-OCR review artifacts beside the OCR output. This is "
    "disabled for standalone PaddleVL and OLM OCR."
)

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


def build_post_ocr_vl_options(post_process_enabled):
    return {
        "post_ocr_vl_enabled": bool(post_process_enabled)
        and bool(DEFAULT_POST_OCR_VL_ENABLED),
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
    }


def processing_supported(method):
    """Return whether the selected visible method supports pre/post processing."""
    return method in PADDLE_OCR_METHODS


def processing_control_updates(method, pre_value=True, post_value=True):
    """Keep pre/post unavailable for standalone VL-style OCR methods."""
    enabled = processing_supported(method)
    return [
        gr.update(value=bool(pre_value) if enabled else False, interactive=enabled),
        gr.update(value=bool(post_value) if enabled else False, interactive=enabled),
    ]


def submit_config(
    method,
    image_input_folder,
    output_folder,
    language,
    generate_text_file,
    pre_process_enabled,
    post_process_enabled,
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
    tile_strategy = BEST_TILE_STRATEGY
    tile_rows, tile_cols = tile_grid_for_strategy(tile_strategy)
    tile_source_images = method in PADDLE_OCR_METHODS
    
    opt = {
        "method": method,
        "ocr_version": DEFAULT_OCR_VERSION,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "generate_text_file": bool(generate_text_file),
        "clean_punctuation_issues": bool(post_process_enabled)
        and bool(DEFAULT_CLEAN_PUNCTUATION_ISSUES),
        "preprocessing_steps": (
            list(BEST_PREPROCESSING_STEPS) if pre_process_enabled else []
        ),
        "ignore_black_border": bool(pre_process_enabled)
        and bool(DEFAULT_IGNORE_BLACK_BORDER),
        "tile_source_images": tile_source_images,
        "tile_strategy": tile_strategy,
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
    opt.update(build_post_ocr_vl_options(post_process_enabled))
    if method in PADDLE_OCR_METHODS:
        iterator = run_ocr_job(input_path, output_root, opt)
    elif method == "PaddleVL":
        iterator = run_paddlevl_job(input_path, output_root, opt)
    elif method == "OLMOCR":
        iterator = run_olmocr_job(input_path, output_root, opt)
    else:
        yield f"Unsupported OCR method: {method}"
        return

    for message in iterator:
        yield message

    yield " OCR Completed, please check pdf and alto files at your output folder"

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
            value=None,
        )

        generate_text_file = gr.Checkbox(
            label="Generate TXT instead of PDF/ALTO.",
            value=False,
            info=(
                "When enabled, writes one TXT file per image and skips "
                "searchable PDF and ALTO generation."
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
        method_radio,
        generate_text_file,
        pre_process,
        post_process,
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
        fn=processing_control_updates,
        inputs=[method_radio, pre_process, post_process],
        outputs=[pre_process, post_process],
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
        Re-enable UI components after OCR processing completes.
        """
        pre_update, post_update = processing_control_updates(
            method,
            pre_value,
            post_value,
        )
        return [
            *[
                gr.update(interactive=True)
                for _ in ui_components[:-2]
            ],
            pre_update,
            post_update,
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
        generate_text_file,
        pre_process,
        post_process,
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
