import gradio as gr
import tkinter as tk
from tkinter import filedialog

from pathlib import Path

from services.ocr_pp import run_ocr_job  
from services.ocr_vl import run_paddlevl_job
from services.auto_ocr import run_auto_job

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
    return [ 
        gr.update(interactive=interactive) for _ in ui_components 
    ]

def disable_run_button():
    return gr.update(interactive=False)


def enable_run_button():
    return gr.update(interactive=True)

def choose_folder():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory()
    root.destroy()
    return folder


def update_options(method):
    return (
        gr.update(visible=(method in ["PP-OCRv5","PP-StructureV3","Auto Routing"])),
        gr.update(visible=(method in ["PaddleVL","Auto Routing"])),
    )


def submit_config(
    method,
    image_input_folder,
    output_folder,
    language,
    use_doc_orientation_classify,
    use_doc_unwarping,
    use_textline_orientation,
    use_layout_detection
):
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
    }

    yield "Downloading models... OCR will start after the download is complete."

    if method in ["PP-StructureV3","PP-OCRv5"]:
        opt.update({
            "lang": language,
            "use_textline_orientation": use_textline_orientation,
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
        })
        for message in run_auto_job(input_path, output_root, opt):
            yield message


    yield " OCR Completed, please check pdf and alto files at your output folder"

with gr.Blocks() as demo:
    with gr.Column(elem_id="fixed-header"):
        gr.Markdown(
            "## CRKN PaddleOCR Engine",
        
            )

        method_radio = gr.Radio(
            choices=["PP-OCRv5","PP-StructureV3", "PaddleVL", "Auto Routing"],
            label="Select Processing Method",
            value="PP-OCRv5"
        )

    gr.Markdown("### Folder Selection")

    with gr.Row():
        image_input_folder = gr.Textbox(label="Image Input Folder")
        image_input_btn = gr.Button("Browse Input Folder")

    with gr.Row():
        output_folder = gr.Textbox(label="Output Folder")
        output_folder_btn = gr.Button("Browse Output Folder")

    with gr.Column(visible=True) as common_options:
        use_doc_orientation_classify = gr.Checkbox(
        label='Use document orientation classification during inference.',
        value=False,
        )

        use_doc_unwarping = gr.Checkbox(
            label='Use document image unwarping during inference.',
            value=False
        )

    with gr.Column(visible=False) as paddlevl_options:
        use_layout_detection = gr.Checkbox(
            label='Use layout detection during inference.',
            value=False
        )

    with gr.Column(visible=True) as paddleocr_options:
        language_dropdown = gr.Dropdown(
            choices=LANGUAGE_CHOICES,
            label="Select Language",
            value=None
        )
        
        use_textline_orientation = gr.Checkbox(
            label='Use textline orientation classification during inference.',
            value=False
        )

        
           

    submit_btn = gr.Button("Run OCR")

    output = gr.Textbox(label="OCR Result", lines=20)

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
    ]

    method_radio.change(
        fn=update_options,
        inputs=method_radio,
        outputs=[paddleocr_options, paddlevl_options]
    )

    image_input_btn.click(
        fn=choose_folder,
        inputs=None,
        outputs=image_input_folder
    )

    output_folder_btn.click(
        fn=choose_folder,
        inputs=None,
        outputs=output_folder
    )

    submit_btn.click(
    fn=lambda: set_ui_interactive(False),
    inputs=None,
    outputs=ui_components
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
        use_layout_detection
    ],
    outputs=output
    ).then(
    fn=lambda: set_ui_interactive(True),
    inputs=None,
    outputs=ui_components
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    css="""
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
  
    """)