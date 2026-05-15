# PaddleOCR Pipeline
This project is an OCR tool for internal CRKN staff, built on the PaddleOCR product suite. It provides PP-OCRv5, PP-StructureV3, and PaddleOCR-VL as internal OCR solutions.

## PaddleOCR Product Suite Installation

1. PP-OCRv5 was installed on the PaddleOCR server using pip. For installation details, please refer to:
https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL-MetaX-GPU.html#12-method-2-manually-install-paddlepaddle-and-paddleocr
  After installation, both PP-OCRv5 and PP-StructureV3 are available for use.
  PaddleOCR-VL was deployed using the official Docker Compose setup. For deployment details, please refer to:
  https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html#4

2. Dependency Installation
  ```
    pip install -r requirements.txt

  ```

## Accessing the Development Environment
1. Navigate to the Development Directory
```
  cd "C:\CRKN Projects\paddle_pipeline"

```
2. Activate Virtual Environment
```
  venv\Scripts\activate

```
3. Start the Application
```
  gradio app.py  

```
4. Stop the Application
  Press Ctrl + C directly to terminate the application.

## Client Usage
1. Click the Windows Start icon.
2. In the pop-up window, select Launcher.
3. In the launched window, if you see the following message:
```
  Running on local URL:  http://0.0.0.0:7860
  * To create a public link, set `share=True` in `launch()`.
```
4. Open any web browser and enter:
```
  http://127.0.0.1:7860

```
5. Stop the Application
 Close the launcher window opened in Step 3.
  
## Project Structure

```text
paddle_pipeline/
├── app.py                               # Main application entry point
├── .env                                 # Environment variables
├── .gitignore                           # Git ignore configuration
│
├── engine/
│   ├── __init__.py
│   └── paddle_engine.py                 # PaddleOCR engine initialization and wrapper
│
├── services/
│   ├── __init__.py
│   ├── auto_ocr.py                      # Automatically routes images to PP-StructureV3 or PaddleOCR-VL based on image quality
│   ├── ocr_pp.py                        # PP-OCRv5 and PP-StructureV3 OCR processing
│   ├── ocr_vl.py                        # PaddleOCR-VL OCR processing
│   ├── alto_generation.py               # Generates ALTO XML files from OCR results
│   ├── pdf_generation.py                # Generates searchable PDF files from OCR results
│   ├── txt_generation.py                # Generates TXT output files from OCR results
│   ├── email_notify.py                  # Sends summary and error notification emails
│   ├── ocr_logger.py                    # Logging configuration and OCR log management
│   ├── constants.py                     # Shared constants and configuration values
│   │
│   └── fonts/                           # Font files used for PDF generation
│
├── utils/                               # Shared utility functions
│
└── venv/                                # Python virtual environment
```

# Troubleshooting
- When switching between PP-OCRv5 and PP-StructureV3, parameter loading errors may occasionally occur. Restarting the application will usually resolve the issue.
