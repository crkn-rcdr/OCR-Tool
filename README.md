# OCR Pipeline

This repository is an internal OCR application for CRKN staff. It combines:

- `PP-OCRv6` for the main local OCR workflow
- `olmOCR` for standalone visual OCR through a local vLLM Docker service
- `PaddleOCR-VL 1.6` for standalone visual OCR and optional post-OCR visual
  language checks

The app can run as:

- a Gradio desktop/web app from `app.py`
- a simple launcher from `launcher.py`
- a CLI smoke-test runner from `smoke_test.py`
- a managed shared-PC Windows bundle under `windows_bundle/`
- analysis and reporting helpers under `tools/`

## Overview

At a high level, the app:

1. loads one image or a folder of images
2. optionally preprocesses the page image
3. runs local OCR with `PP-OCRv6`
4. optionally runs a post-OCR word check on suspicious `PP-OCRv6` words
5. writes normalized JSON output
6. writes either:
   - TXT files, or
   - searchable PDF plus ALTO XML

The app is designed for historical newspaper pages, large scans, black borders,
uneven background, and visually noisy OCR cases.

For each processed image, the app can write:

- normalized OCR JSON
- TXT
- searchable PDF
- ALTO XML

Depending on options, it may also write:

- `_preprocessed/` images
- `_post_ocr_vl_tiles/` cropped post-OCR audit tiles
- post-OCR CSV summaries
- progress JSON files
- log files in `logs/`

## Quick Start

If you already know which environment you want:

- Windows: go to [Windows Setup](#windows-setup)
- WSL: go to [WSL Setup](#wsl-setup)

After setup:

1. optionally start the Docker OCR services with `docker compose up -d`
2. run `python app.py` or `python launcher.py`
3. open `http://127.0.0.1:7860`

If you want a scriptable non-Gradio path instead, use [Smoke Test CLI](#smoke-test-cli).

## OCR Method

The app and smoke-test CLI expose three OCR method choices:

- `PaddleOCR (PP-OCRv6)`: the best general default when you want the most
  tunable local OCR path and access to post-OCR VL correction.
- `PaddleVL`: sends each image directly to the PaddleOCR-VL 1.6 API service.
- `OLM OCR`: sends each image directly to the local olmOCR vLLM service.

`PaddleOCR-VL 1.6` is also used by the `post process` checkbox to review
suspicious PP-OCRv6 word crops.

The `pre process` and `post process` checkboxes apply to PP-OCRv6 work only.
They are disabled for standalone PaddleVL and standalone OLM OCR.

The Docker configs and image build files are pinned to `PaddleOCR-VL-1.6` /
`PaddleOCR-VL-1.6-0.9B` and `alleninstituteforai/olmocr:v0.4.27-with-model`.

## Setup

### Common Requirements

- Python 3
- a virtual environment
- enough disk space for Paddle models and OCR output

Optional components:

- Docker Desktop or Docker Engine for `olmOCR` and `PaddleOCR-VL 1.6`
- NVIDIA GPU support for the Docker OCR model services


## Windows Setup

These steps are for running the app directly on Windows.

### 1. Create a virtual environment

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

You can also use `venv` instead of `.venv` if that is already your local
convention.

### 2. Install Python dependencies

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Notes:

- `requirements.txt` is the full Windows-oriented dependency list
- it includes Paddle OCR dependencies, Gradio, PDF generation, and optional
  libraries used by the app

### 3. Optional: start Docker OCR services

If you want standalone OLM OCR, standalone PaddleVL, or post-OCR VL checks,
start the Docker services:

```powershell
docker compose up -d
```

This uses [compose.yaml](compose.yaml).

The default localhost service URLs are:

```text
http://127.0.0.1:8000/v1
http://127.0.0.1:8080
```

### 4. Run the app

From the repo root:

```powershell
python app.py
```

Or:

```powershell
python launcher.py
```

The launcher tries these interpreters in order:

- `venv\Scripts\python.exe`
- `.venv\Scripts\python.exe`
- the current interpreter
- `py -3`
- `python3` / `python`

### 6. Open the UI

Once Gradio starts, open:

```text
http://127.0.0.1:7860
```

### 7. Stop services

- stop the Gradio app with `Ctrl+C`
- stop Docker OCR services with:

```powershell
docker compose down
```

## WSL Setup

These steps are for running under WSL. This is often the cleaner path for
headless or scripted OCR work.

### 1. Create or use a WSL distro

Ubuntu is the easiest target.

Make sure basic Python tooling exists:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

### 2. Run the WSL setup helper

CPU:

```bash
bash setup_wsl.sh cpu
```

GPU:

```bash
bash setup_wsl.sh gpu
```

What [setup_wsl.sh](setup_wsl.sh) does:

- creates `.venv` if missing
- installs or upgrades `pip`, `setuptools`, and `wheel`
- installs `paddlepaddle==3.2.1` for CPU, or `paddlepaddle-gpu==3.2.1` for GPU
- installs `requirements-wsl.txt`
- forces `opencv-contrib-python-headless` so WSL does not depend on GUI OpenCV
- prints versions for `cv2`, `fitz`, `gradio`, and `paddleocr`

### 3. Activate the environment

```bash
source .venv/bin/activate
```

### 4. Optional: start Docker OCR services

If Docker is available inside WSL:

```bash
docker compose up -d
```

This is usually easiest when Docker Desktop is installed on Windows with WSL
integration enabled.

### 5. Optional: connect to Docker OCR endpoints

From WSL, the default Docker services are:

- OLM OCR: `http://127.0.0.1:8000/v1`
- PaddleOCR-VL: `http://127.0.0.1:8080`

### 6. Run the app

```bash
python3 app.py
```

Or use the launcher:

```bash
python3 launcher.py
```

### 7. WSL path notes

- if `tkinter` is unavailable in WSL, the browse buttons will not work
- paste paths manually instead
- Windows paths are typically:

```text
/mnt/c/CRKN Projects/paddle_pipeline/...
```

The app already handles this better than older versions, but manual path entry
is still the safest WSL workflow.

## Runtime Configuration

The repo ships with [.env.example](.env.example):

```text
API_IMAGE_TAG_SUFFIX=latest-nvidia-gpu-offline
VLM_IMAGE_TAG_SUFFIX=latest-nvidia-gpu-offline
VLM_BACKEND=vllm
```

Useful runtime environment variables include:

### PaddleVL

- `PADDLE_VL_API_URL`
  - default: `http://127.0.0.1:8080`

### OLM OCR

- `OLMOCR_API_URL`
  - default: `http://127.0.0.1:8000/v1`
- `OLMOCR_MODEL`
  - default: `allenai/olmOCR-2-7B-1025-FP8`
- `OLMOCR_IMAGE_TAG`
  - default: `v0.4.27-with-model`

### Gradio server

- `OCR_TOOL_HOST`
  - default: `0.0.0.0`
- `OCR_TOOL_PORT`
  - default: `7860`

### Post-OCR VL behavior

The post-OCR VL backend is fixed to PaddleOCR-VL 1.6. Configure the service URL
with `PADDLE_VL_API_URL`.

### ALTO output

- `OCR_ALTO_PROFILE`
  - default: `abbyy_like`
  - accepted values: `abbyy_like`, `legacy`

## Running The App

### Gradio UI

Start:

```bash
python app.py
```

or:

```bash
python launcher.py
```

## Gradio Option Guide

The Gradio app exposes only the production controls:

- `Select Processing Method`
- `Image Input Folder`
- `Output Folder`
- `OCR method`
- `Select Language`
- `Generate TXT instead of PDF/ALTO.`
- `pre process`
- `post process`

The detailed OCR tuning values are fixed in `app.py` from the evaluation-backed
recipe. For `PP-OCRv6`, the app enables source-image tiling with
`3x2 + PP-DocLayoutV3`, `96` px overlap, `32` px DocLayout padding, `0.20`
DocLayout threshold, and the recovery tile. This favors text recall on dense
historical pages; cleanup happens later in the pipeline.

### Generate TXT vs PDF/ALTO

`Generate TXT instead of PDF/ALTO.` writes one TXT file per image and skips
searchable PDF and ALTO generation. Leave it off for normal archival output.

### pre process

`pre process` means:

- crop the detected black scan border before OCR
- run OCR on the page crop
- offset OCR geometry back into the original image coordinates
- apply `background_flatten` to even newspaper page shading
- save the OCR-facing PNG under `_preprocessed/`

This matches the strongest preprocessing result from the evaluation work:
isolate the page and flatten uneven background without adding more aggressive
steps that can damage useful detail.

### post process

`post process` means:

- clean conservative punctuation artifacts in the saved OCR JSON
- for `PP-OCRv6`, send suspicious word crops to PaddleVL
- apply only accepted PaddleVL corrections
- write post-OCR review artifacts beside the OCR output

The fixed post-OCR settings are:

- backend: `paddlevl`
- candidate mode: `suspicious`
- max words: disabled
- word crop padding: `5` px
- low-confidence threshold: `0.90`
- apply accepted corrections: enabled
- save word tiles: disabled

## Outputs

### Searchable PDF generation

Searchable PDFs are created with PyMuPDF.

Current behavior:

- the source page image is inserted as the visible page
- invisible OCR text is written on top as the searchable layer
- if punctuation cleanup is enabled, the cleaned OCR text is what gets written
  into the searchable layer
- Latin OCR text now uses built-in `Helvetica`
- CJK text still uses the embedded Noto CJK font

This change was made to improve copy/select behavior in PDF viewers.

Rationale:

- searchable PDF is the practical delivery format when people want the original
  page image preserved but still need copy, search, and highlight support
- ALTO is the better fit when downstream systems care about OCR structure and
  coordinates

### Output files

For a typical image run, expect files like:

```text
output/
  large/
    oocihm.N_00155_18750610.4.json
    oocihm.N_00155_18750610.4.txt
    oocihm.N_00155_18750610.4.pdf
    oocihm.N_00155_18750610.4.xml
    oocihm.N_00155_18750610.4.post_ocr_vl.csv
    oocihm.N_00155_18750610.4.post_ocr_vl_progress.json
    _preprocessed/
    _post_ocr_vl_tiles/
```

Meaning:

- `.json`: normalized OCR payload
- `.txt`: plain text, if TXT generation was selected
- `.pdf`: searchable PDF, if TXT mode was not selected
- `.xml`: ALTO XML, if TXT mode was not selected
- `.post_ocr_vl.csv`: applied and rejected post-OCR VL records
- `.post_ocr_vl_progress.json`: summary/progress metadata for the post-OCR run

If punctuation cleanup is enabled, the `.json` text fields are cleaned before
TXT/PDF/ALTO generation so the output formats stay in sync.

## Smoke Test CLI

The repo includes a bundled test corpus under `testdata/source`.

Default smoke test:

```bash
python smoke_test.py
```

Target a specific image:

```bash
python smoke_test.py \
  --method PP-OCRv6 \
  --input testdata/source/large/oocihm.N_00155_18750610.4.jpg \
  --output output-smoke
```

Run PP-OCRv6 with PaddleVL post-OCR auditing:

```bash
python smoke_test.py \
  --method PP-OCRv6 \
  --input testdata/source/large/oocihm.N_00155_18750610.4.jpg \
  --output output-smoke-post-vl \
  --post-ocr-vl-check \
  --post-ocr-vl-candidate-mode suspicious \
  --post-ocr-vl-max-words 25 \
  --post-ocr-vl-no-apply
```

`smoke_test.py` covers almost all of the same option families as the Gradio
app, plus a couple of script-only details:

- `--disable-doclayout-recovery` turns off the fallback tile used to recover
  regions missed by PP-DocLayoutV3
- `--post-ocr-vl-no-apply`, `--post-ocr-vl-no-save-tiles`, and
  `--post-ocr-vl-no-group-lines` are negative flags for the corresponding
  positive Gradio checkboxes

Current smoke-test defaults are slightly different from the Gradio UI:

- preprocessing defaults to the same best pre-process bundle when
  `--crop-black-border` is active
- post-OCR VL defaults to enabled for `PP-OCRv6`
- `--tile-strategy` defaults to `grid_doclayout_v3`

## Helper Scripts

#### Email Notification Helper

[services/email_notify.py](services/email_notify.py) is the only `services/`
module that is currently intended to be run directly as a small utility script.
It sends summary and error logs from the configured log directory using SMTP
settings from the environment.

Required environment variables:

- `SMTP_HOST`
- `SMTP_PORT`
- `FROM_EMAIL`
- `TO_EMAIL`

## Repository Layout

```text
app.py                     Gradio application
launcher.py                Lightweight launcher for app.py
smoke_test.py              CLI smoke test and scripted runs
compose.yaml               PaddleOCR-VL Docker services
setup_wsl.sh               WSL setup helper
requirements.txt           Main Windows / full environment dependencies
requirements-wsl.txt       WSL-friendly dependency set
windows_bundle/            Shared-PC Windows start/stop/install scripts
engine/                    Paddle engine setup and compatibility code
services/
  ocr_pp.py                PP-OCRv6 execution
  post_ocr_vl.py           Post-OCR VL checks for PP-OCRv6
  image_preprocessing.py   Preprocessing steps
  tile_ocr.py              Tiling logic
  pdf_generation.py        Searchable PDF generation
  alto_generation.py       ALTO XML generation
  txt_generation.py        TXT generation
  punctuation_cleanup.py   Conservative OCR text cleanup
  word_box_normalization.py Normalize word-box metadata
  ocr_defaults.py          Shared defaults
  ocr_logger.py            Logging and run summaries
  constants.py             Shared file-extension/constants
  email_notify.py          Optional email log sender
  fonts/                   Fonts used for PDF generation
tools/
  batch_runner.py          Headless CLI batch runner with resume support
  evaluate_ocr.py          CER/WER benchmark tool against Ground Truth
utils/                     Shared utilities
logs/                      Run logs
output/                    Typical output folder
```

## Headless Batch Runner

For large unattended runs (e.g. overnight processing of entire microfilm reels,
directories of monographs, or multi-year newspaper collections), use
`tools/batch_runner.py` without launching Gradio:

```bash
python tools/batch_runner.py \
  --input "input/Source Files/oocihm.lac_reel_h1198" \
  --output "output/lac_reel_h1198" \
  --method PP-OCRv6 \
  --resume
```

Key features:

- `--resume` / `-r`: automatically skips pages that already have complete output
  in the target destination folder.
- `--format`: choice between `pdf_alto` (searchable PDF + ALTO XML, default) or
  `txt`.
- `--pre-process` / `--no-pre-process`: toggle black-border crop and background
  flattening.
- `--post-process` / `--no-post-process`: toggle post-OCR VLM verification.
- `--concurrency` & `--batch-size`: control concurrent worker threads and candidate
  chunking sent to the VLM service.
- Live progress bar with throughput tracking (pages/min and ETA) via `tqdm`.
- Writes a complete run manifest and error telemetry file to
  `batch_run_summary.json` at the output root.

## Accuracy Evaluation (CER / WER)

To benchmark OCR and VLM post-processing quality against Ground Truth transcripts
or ALTO XML references across sample Canadian collections, use
`tools/evaluate_ocr.py`:

```bash
python tools/evaluate_ocr.py \
  --ref "testdata/ground_truth" \
  --hyp "output/test_run" \
  --norm standard \
  --output-csv "output/evaluation_summary.csv"
```

Features:

- Computes **Character Error Rate (CER)**, **Word Error Rate (WER)**, Accuracy,
  and **Bag-of-Words F1**.
- Normalization modes (`--norm`):
  - `standard`: historical Canadian typography normalization (reconnects line-break
    hyphens, normalizes archaic long-s `ſ` -> `s`, ligatures `æ`, `œ`, `ﬀ`, typographic
    quotes and dashes) plus casefolding.
  - `raw`: exact verbatim comparison.
  - `alpha`: alphanumeric token comparison.
- Input flexibility: reference and hypothesis paths can be files or directories,
  supporting `.txt`, `.xml` (ALTO), `.json`, and `.pdf` files.
- **VLM Impact Analysis**: automatically reads `.post_ocr_vl.csv` sidecars if present,
  reconstructs pre-VLM text, and calculates $\Delta\text{CER}$ and $\Delta\text{WER}$
  to quantify whether VLM verification yielded positive accuracy improvements.
- Exports results to Markdown summary tables, `--output-csv`, and `--output-json`.

## Troubleshooting

### The app starts but folder browse does not work

In WSL or headless environments, `tkinter` may be unavailable. Paste the input
and output paths manually.

### PaddleVL methods fail immediately

Make sure:

- Docker is running
- `docker compose up -d` is healthy
- `http://127.0.0.1:8080/health` responds

### A PDF opens in VS Code with a binary / encoding warning

That is normal. A PDF is a binary file. Open it in a PDF viewer rather than the
text editor.

### The selected PDF text shows replacement characters

Recent versions of this repo changed the Latin OCR text layer to use built-in
`Helvetica` instead of an embedded custom Latin font. Regenerate the PDF if you
created it before that change.

### My comparison workbook formatting keeps changing

The comparison tool now preserves formatting from an existing workbook when you
overwrite the same `.xlsx` file. If you want to keep your manual sheet names,
freeze panes, widths, and row heights, rerun the tool against the same workbook
path.

## Logs

Run logs are written under `logs/`.

You will typically see:

- summary logs
- error logs
- per-run PDF or OCR errors

If something behaves differently than the README says, prefer the code path in:

- [app.py](app.py)
- [services/ocr_pp.py](services/ocr_pp.py)
- [services/post_ocr_vl.py](services/post_ocr_vl.py)
- [services/pdf_generation.py](services/pdf_generation.py)
