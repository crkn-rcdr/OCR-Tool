# OCR Pipeline

This repository is an internal OCR application for CRKN staff. It combines:

- `PP-OCRv5` for the main local OCR workflow
- `PP-StructureV3` for Paddle's structure-oriented OCR workflow
- `PaddleVL` for the PaddleOCR-VL Docker service
- `PERO-OCR` as an optional external OCR backend
- optional post-OCR visual language checks for `PP-OCRv5`

The app can run as:

- a Gradio desktop/web app from `app.py`
- a simple launcher from `launcher.py`
- a CLI smoke-test runner from `smoke_test.py`

## What The App Does

At a high level, the app:

1. loads one image or a folder of images
2. optionally preprocesses the page image
3. runs OCR with one of the supported backends
4. optionally runs a post-OCR word check on suspicious `PP-OCRv5` words
5. writes normalized JSON output
6. writes either:
   - TXT files, or
   - searchable PDF plus ALTO XML

The app is designed for historical newspaper pages, large scans, black borders,
uneven background, and visually noisy OCR cases.

## Current Backend Behavior

### `PP-OCRv5`

This is the main local OCR path and the best default for most runs.

Available with:

- preprocessing
- black-border cropping
- fixed-grid tiling
- PP-DocLayoutV3 tiling
- post-OCR VL word checking
- TXT output
- searchable PDF
- ALTO XML

### `PP-StructureV3`

This is available as a direct Paddle structure pipeline, but in the current app
it is intentionally simpler than `PP-OCRv5`.

Current behavior:

- preprocessing is available
- tiling is disabled
- post-OCR VL is disabled
- TXT output or PDF/ALTO output still works

The UI now hides tiling and post-OCR VL controls for `PP-StructureV3`, and the
submit path also forces those options off even if they were previously enabled.

### `PaddleVL`

This uses the PaddleOCR-VL service from Docker Compose.

Use it when you want the service-based VLM OCR path rather than the local
Paddle OCR pipelines.

### `PERO-OCR`

This is optional and requires a working `pero-ocr` install or local checkout
plus a PERO `config.ini`.

### `Auto Routing`

This route analyzes image quality and chooses between:

- `PP-StructureV3` for cleaner pages
- `PaddleVL` for pages with skew, warping, or illumination problems

Use it mainly for mixed folders. It is less useful for controlled benchmarking.

Current limitation:

- source-image tiling is not currently allowed at run time for `Auto Routing`

## Supported Outputs

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

## Repository Layout

```text
app.py                     Gradio application
launcher.py                Lightweight launcher for app.py
smoke_test.py              CLI smoke test and scripted runs
compose.yaml               PaddleOCR-VL Docker services
setup_wsl.sh               WSL setup helper
requirements.txt           Main Windows / full environment dependencies
requirements-wsl.txt       WSL-friendly dependency set
testdata/source/           Bundled smoke-test corpus
engine/                    Paddle engine setup and compatibility code
services/
  auto_ocr.py              Auto Routing logic
  ocr_pp.py                PP-OCRv5 and PP-StructureV3 execution
  ocr_vl.py                PaddleVL execution
  post_ocr_vl.py           Post-OCR VL checks for PP-OCRv5
  pero_ocr.py              PERO integration
  image_preprocessing.py   Preprocessing steps
  tile_ocr.py              Tiling logic
  pdf_generation.py        Searchable PDF generation
  alto_generation.py       ALTO XML generation
  txt_generation.py        TXT generation
  ocr_defaults.py          Shared defaults
  ocr_logger.py            Logging and run summaries
  fonts/                   Fonts used for PDF generation
utils/                     Shared utilities
logs/                      Run logs
output/                    Typical output folder
```

## Requirements

### Common Requirements

- Python 3
- a virtual environment
- enough disk space for Paddle models and OCR output

Optional components:

- Docker Desktop or Docker Engine for `PaddleVL`
- NVIDIA GPU support for the PaddleOCR-VL stack if you want GPU containers
- Ollama or another OpenAI-compatible vision endpoint for post-OCR VL checks
- PERO-OCR plus a PERO config if you want the PERO backend

The repo is currently pinned around:

- `paddleocr==3.5.0`
- `paddlex[ocr]==3.5.2`
- `PyMuPDF==1.27.2`

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

### 3. Optional: start PaddleVL

If you want `PaddleVL` or `Auto Routing`, start the Docker services:

```powershell
docker compose up -d
```

This uses [compose.yaml](compose.yaml).

The default exposed service URL is:

```text
http://127.0.0.1:8080
```

### 4. Optional: install and run Ollama for post-OCR VL

If you want `PP-OCRv5` post-OCR VL checks through Ollama:

1. install Ollama for Windows
2. start Ollama
3. pull one or more supported models

Examples:

```powershell
ollama pull fredrezones55/chandra-ocr-2
ollama pull richardyoung/olmocr2:7b-q8
ollama pull deepseek-ocr:3b
```

The current app defaults expect an OpenAI-compatible endpoint at:

```text
http://localhost:11434/v1
```

### 5. Run the app

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
- stop PaddleVL with:

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

### 4. Optional: start PaddleVL

If Docker is available inside WSL:

```bash
docker compose up -d
```

This is usually easiest when Docker Desktop is installed on Windows with WSL
integration enabled.

### 5. Optional: connect to Ollama

You can run Ollama either:

- on Windows and call it from WSL using `http://localhost:11434/v1`, or
- directly inside WSL if you prefer

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

## Environment Variables

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

### Ollama OpenAI / post-OCR VL

- `OLLAMA_OPENAI_API_URL`
  - default: `http://localhost:11434/v1`
- `OLLAMA_OPENAI_MODEL`
  - default: `richardyoung/olmocr2:7b-q8`
- `OLLAMA_OPENAI_API_KEY`
  - optional

Legacy compatibility is still accepted:

- `OLMOCR_API_URL`
- `OLMOCR_MODEL`
- `OLMOCR_API_KEY`

### Post-OCR VL behavior

- `POST_OCR_VL_BACKEND`
  - default: `olmocr`

### PERO

- `PERO_OCR_SOURCE_PATH`
- `PERO_OCR_CONFIG_PATH`
- `PERO_OCR_DEVICE`
- `PERO_OCR_GPU_ID`

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

The main method choices are:

- `PP-OCRv5`
- `PP-StructureV3`
- `PERO-OCR`
- `PaddleVL`
- `Auto Routing`

## Smoke Test CLI

The repo includes a bundled test corpus under `testdata/source`.

Default smoke test:

```bash
python smoke_test.py
```

Target a specific image:

```bash
python smoke_test.py \
  --method PP-OCRv5 \
  --input testdata/source/large/oocihm.N_00155_18750610.4.jpg \
  --output output-smoke
```

Run PP-OCRv5 with PaddleVL post-OCR auditing:

```bash
python smoke_test.py \
  --method PP-OCRv5 \
  --input testdata/source/large/oocihm.N_00155_18750610.4.jpg \
  --output output-smoke-post-vl \
  --post-ocr-vl-check \
  --post-ocr-vl-backend paddlevl \
  --post-ocr-vl-candidate-mode suspicious \
  --post-ocr-vl-max-words 25 \
  --post-ocr-vl-no-apply
```

Run PP-OCRv5 with the Ollama/OpenAI-compatible post-OCR backend:

```bash
python smoke_test.py \
  --method PP-OCRv5 \
  --input testdata/source/large/oocihm.N_00155_18750610.4.jpg \
  --output output-smoke-ollama \
  --post-ocr-vl-check \
  --post-ocr-vl-backend olmocr \
  --post-ocr-vl-ollama-openai-url http://localhost:11434/v1 \
  --post-ocr-vl-ollama-openai-model fredrezones55/chandra-ocr-2
```

The CLI still accepts the older aliases:

- `--post-ocr-vl-olmocr-url`
- `--post-ocr-vl-olmocr-model`

## How The Main Options Work

### Generate TXT

`Generate a text file with OCR results for each image` means:

- generate TXT
- skip searchable PDF and ALTO generation

If this is off, the app generates searchable PDF and ALTO instead.

### Preprocessing

Available preprocessing steps:

- `light_clean`
- `background_flatten`
- `contrast`
- `deskew`
- `sauvola_binary`

Useful starting presets:

- clean scans: no preprocessing
- newspaper with dark scanner border:
  - black-border cropping
- uneven background:
  - light clean
  - background flatten
  - contrast

If preprocessing is enabled, the OCR-facing image is written under
`_preprocessed/`.

### Tiling

Tiling is for large pages and dense historical newspapers.

Available strategies in the codebase:

- `grid`
- `grid_3x2`
- `doclayout_v3`
- `grid_doclayout_v3`
- `grid_3x2_doclayout_v3`

Current app behavior:

- `PP-OCRv5`: tiling available
- `PERO-OCR`: tiling available
- `PP-StructureV3`: tiling disabled
- `Auto Routing`: source-image tiling currently not allowed at run time

### Post-OCR VL Word Check

This feature is currently only active for `PP-OCRv5`.

It uses `text_word` and `text_word_boxes` from the PP-OCR result to crop
suspicious words and send them to either:

- `PaddleVL`
- an OpenAI-compatible vision endpoint, labeled in the UI as `Ollama OpenAI`

The post-OCR step records metadata under `post_ocr_vl_check` in the JSON and
can optionally:

- save the cropped tiles
- write a `.post_ocr_vl.csv` audit file
- apply accepted corrections back into `text_word` and `rec_texts`

Current model notes:

- `richardyoung/olmocr2:7b-q8` is the strongest current Ollama option in this app
- `fredrezones55/chandra-ocr-2` now uses a Chandra-specific Ollama fallback path
- `deepseek-ocr:3b` has a model-aware cleanup path, but is still weaker for
  this word-correction use case

### PDF Generation

Searchable PDFs are created with PyMuPDF.

Current behavior:

- the source page image is inserted as the visible page
- invisible OCR text is written on top as the searchable layer
- Latin OCR text now uses built-in `Helvetica`
- CJK text still uses the embedded Noto CJK font

This change was made to improve copy/select behavior in PDF viewers.

## Output Files

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

## Troubleshooting

### The app starts but folder browse does not work

In WSL or headless environments, `tkinter` may be unavailable. Paste the input
and output paths manually.

### `PP-StructureV3` shows fewer advanced options

This is expected. The current app intentionally disables source-image tiling and
post-OCR VL for `PP-StructureV3`.

### PaddleVL methods fail immediately

Make sure:

- Docker is running
- `docker compose up -d` is healthy
- `http://127.0.0.1:8080/health` responds

### Ollama post-OCR checks return empty or weak results

Check:

- the Ollama server is running
- the model name exists exactly as installed
- the endpoint is `http://localhost:11434/v1`

Useful test:

```powershell
ollama list
```

### A PDF opens in VS Code with a binary / encoding warning

That is normal. A PDF is a binary file. Open it in a PDF viewer rather than the
text editor.

### The selected PDF text shows replacement characters

Recent versions of this repo changed the Latin OCR text layer to use built-in
`Helvetica` instead of an embedded custom Latin font. Regenerate the PDF if you
created it before that change.

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
