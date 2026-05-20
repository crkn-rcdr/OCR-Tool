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


def update_grid_overlap_control(method, tile_strategy):
    supports_tiling = method in ["PP-OCRv5", "PP-StructureV3"]
    show_fixed_grid_controls = supports_tiling and (tile_strategy or "grid") in {
        "grid",
        "grid_doclayout_v3",
    }

    return gr.update(visible=show_fixed_grid_controls, interactive=supports_tiling)


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
    tile_overlap_update = update_grid_overlap_control(method, tile_strategy)

    return (
        gr.update(visible=(method in ["PP-OCRv5","PP-StructureV3","Auto Routing"])),
        gr.update(visible=(method in ["PaddleVL","Auto Routing"])),
        tile_checkbox_update,
        tile_strategy_update,
        tile_overlap_update,
    )


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
    text_det_thresh,
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
        "tile_strategy": tile_strategy or "grid",
        "tile_rows": 3,
        "tile_cols": 1,
        "tile_overlap_px": int(tile_overlap_px) if tile_overlap_px is not None else 48,
    }

    if tile_source_images and method not in ["PP-OCRv5", "PP-StructureV3"]:
        yield "Source-image tiling is currently available for PP-OCRv5 and PP-StructureV3 only."
        return

    yield "Downloading models... OCR will start after the download is complete."

    # Start to run the OCR job based on the selected method and options
    if method in ["PP-StructureV3","PP-OCRv5"]:
        opt.update({
            "lang": language,
            "use_textline_orientation": use_textline_orientation,
            "text_det_thresh": text_det_thresh
        })
      
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
            "lang": language,
            "use_textline_orientation": use_textline_orientation,
            "text_det_thresh": text_det_thresh
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
                ("Content boxes (CV)", "content_boxes"),
                ("PP-DocLayoutV3 Layout", "doclayout_v3"),
                ("3x1 + PP-DocLayoutV3", "grid_doclayout_v3"),
            ],
            value="grid",
            label="Tile Strategy",
            info=(
                "Use fixed 3x1 tiles, OpenCV content boxes, PaddleX "
                "PP-DocLayoutV3 regions, or PP-DocLayoutV3 inside each "
                "3x1 band."
            ),
        )

        tile_overlap_px = gr.Slider(
            minimum=0,
            maximum=256,
            value=48,
            step=4,
            label="Tile Overlap (px)",
            info="Overlap for fixed 3x1 grid tiles before deduping.",
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
        value=0.3,
        step=0.01,
        label="Text Detection Threshold"
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
        text_det_thresh,
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
        ]
    )

    tile_strategy.change(
        fn=update_grid_overlap_control,
        inputs=[method_radio, tile_strategy],
        outputs=tile_overlap_px,
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
        text_det_thresh,
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
