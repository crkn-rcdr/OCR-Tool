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
    DEFAULT_DOCLAYOUT_PADDING_PX,
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


def update_tile_controls(method, tile_strategy):
    supports_tiling = method in ["PP-OCRv5", "PP-StructureV3"]
    strategy = tile_strategy or "grid_doclayout_v3"
    show_fixed_grid_controls = supports_tiling and strategy in {
        "grid",
        "grid_doclayout_v3",
    }
    show_doclayout_controls = supports_tiling and strategy in {
        "doclayout_v3",
        "grid_doclayout_v3",
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
    supports_tiling = method in ["PP-OCRv5", "PP-StructureV3"]
    tile_checkbox_update = (
        gr.update(interactive=True)
        if supports_tiling
        else gr.update(interactive=False, value=False)
    )
    tile_strategy_update = gr.update(interactive=supports_tiling)
    tile_control_updates = update_tile_controls(method, tile_strategy)

    return (
        gr.update(visible=(method in ["PP-OCRv5","PP-StructureV3","Auto Routing"])),
        gr.update(visible=(method in ["PaddleVL","Auto Routing"])),
        tile_checkbox_update,
        tile_strategy_update,
        *tile_control_updates,
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
    
    opt = {
        "method": method,
        "use_doc_orientation_classify": use_doc_orientation_classify,
        "use_doc_unwarping": use_doc_unwarping,
        "generate_text_file": generate_text_file,
        "tile_source_images": tile_source_images,
        "tile_strategy": tile_strategy or "grid_doclayout_v3",
        "tile_rows": 3,
        "tile_cols": 1,
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

    if tile_source_images and method not in ["PP-OCRv5", "PP-StructureV3"]:
        yield "Source-image tiling is currently available for PP-OCRv5 and PP-StructureV3 only."
        return

    yield "Downloading models... OCR will start after the download is complete."
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

    # Start to run the OCR job based on the selected method and options
    if method in ["PP-StructureV3","PP-OCRv5"]:
        opt.update(paddle_options)
      
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
        gr.Markdown("## CRKN PaddleOCR Engine")

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

        tile_source_images = gr.Checkbox(
            label='Tile source images before OCR.',
            value=False,
            info="Available for PP-OCRv5 and PP-StructureV3. Fixed grid uses 3 rows x 1 column.",
        )

        tile_strategy = gr.Dropdown(
            choices=[
                ("Fixed grid", "grid"),
                ("PP-DocLayoutV3 Layout", "doclayout_v3"),
                ("3x1 + PP-DocLayoutV3", "grid_doclayout_v3"),
            ],
            value="grid_doclayout_v3",
            label="Tile Strategy",
            info=(
                "Use fixed 3x1 tiles, PaddleX PP-DocLayoutV3 regions, "
                "or PP-DocLayoutV3 inside each 3x1 band."
            ),
        )

        tile_overlap_px = gr.Slider(
            minimum=0,
            maximum=256,
            value=DEFAULT_TILE_OVERLAP_PX,
            step=4,
            label="Tile Overlap (px)",
            info="Overlap for fixed 3x1 grid tiles before deduping.",
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
            minimum=4000,
            maximum=12000,
            value=DEFAULT_TEXT_DET_MAX_SIDE_LIMIT,
            step=32,
            label="Text Detection Max Side Limit",
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
    output = gr.Textbox(label="PaddleOCR Result", lines=20)

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
    ]

    # Update visible OCR option sections
    # when the processing method changes.
    method_radio.change(
        fn=update_options,
        inputs=[method_radio, tile_strategy],
        outputs=[
            paddleocr_options,
            paddlevl_options,
            tile_source_images,
            tile_strategy,
            tile_overlap_px,
            tile_doclayout_padding_px,
            tile_doclayout_threshold,
            tile_doclayout_recovery,
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

    def after_run():
        """
        Re-enable UI components after OCR processing completes.
        """
        return [
            *[gr.update(interactive=True) for _ in ui_components],
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
    ],
    outputs=output
    ).then(
    fn=after_run,
    inputs=None,
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
