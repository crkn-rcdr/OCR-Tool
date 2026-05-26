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
- analysis and reporting helpers under `tools/`

## Overview

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
- native PERO `.page.xml` files
- log files in `logs/`

## Quick Start

If you already know which environment you want:

- Windows: go to [Windows Setup](#windows-setup)
- WSL: go to [WSL Setup](#wsl-setup)

After setup:

1. optionally start `PaddleVL` with `docker compose up -d`
2. optionally start Ollama if you want post-OCR VL through an OpenAI-compatible endpoint
3. run `python app.py` or `python launcher.py`
4. open `http://127.0.0.1:7860`

If you want a scriptable non-Gradio path instead, use [Smoke Test CLI](#smoke-test-cli).

## Choose A Method

The main method choices are:

- `PP-OCRv5`
- `PP-StructureV3`
- `PERO-OCR`
- `PaddleVL`
- `Auto Routing`

Practical rationale:

- `PP-OCRv5` is the best general default when you want the most tunable local
  OCR path and access to post-OCR VL correction.
- `PP-StructureV3` is useful when the page is fairly clean and you want a
  simpler structure-oriented Paddle path without the extra post-correction
  layer.
- `PERO-OCR` is worth trying when you already have a strong PERO setup or need
  to compare against an external OCR engine with its own segmentation behavior.
- `PaddleVL` is the better bet for visually difficult pages where a vision
  model may handle warped layout, uneven lighting, or awkward region structure
  better than the local OCR pipeline.
- `Auto Routing` is for mixed batches where manual per-folder method selection
  would be tedious.

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
- the Gradio app hides source-image tiling and forces it off at submit time
- the shared tiling service and `smoke_test.py` still support fixed-grid tiling
  for `PP-StructureV3` (`grid` and `grid_3x2` only)
- post-OCR VL is disabled
- TXT output or PDF/ALTO output still works

The UI now hides tiling and post-OCR VL controls for `PP-StructureV3`, and the
submit path also forces those options off even if they were previously enabled.

### `PaddleVL`

This uses the PaddleOCR-VL service from Docker Compose.

Use it when you want the service-based VLM OCR path rather than the local
Paddle OCR pipelines.

Current app behavior:

- the Gradio app exposes `Use layout detection during inference`
- `services/ocr_vl.py` can forward additional PaddleX request keys when they
  are supplied programmatically, but those extra options are not currently
  surfaced in the Gradio UI

### `PERO-OCR`

This is optional and requires a working `pero-ocr` install or local checkout
plus a PERO `config.ini`.

### `Auto Routing`

This route analyzes image quality and chooses between:

- `PP-StructureV3` for cleaner pages
- `PaddleVL` for pages with skew, warping, or illumination problems

Use it mainly for mixed folders. It is less useful for controlled benchmarking.

Current limitation:

- the Gradio app still shows tiling controls for `Auto Routing`, but the submit
  path rejects source-image tiling at run time
- `smoke_test.py` still accepts tiling flags for `Auto Routing`; those options
  only matter when the router ultimately picks a local OCR backend

## Setup

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

### Gradio server

- `OCR_TOOL_HOST`
  - default: `0.0.0.0`
- `OCR_TOOL_PORT`
  - default: `7860`

### Ollama OpenAI / post-OCR VL

- `OLLAMA_OPENAI_API_URL`
  - default: `http://localhost:11434/v1`
- `OLLAMA_OPENAI_MODEL`
  - default: `richardyoung/olmocr2:7b-q8`
- `OLLAMA_OPENAI_API_KEY`
  - optional

Additional compatibility names are also accepted:

- `OPENAI_COMPAT_API_URL`
- `OPENAI_COMPAT_MODEL`
- `OPENAI_COMPAT_API_KEY`

Legacy compatibility is still accepted:

- `OLMOCR_API_URL`
- `OLMOCR_MODEL`
- `OLMOCR_API_KEY`

### Post-OCR VL behavior

- `POST_OCR_VL_BACKEND`
  - default: `olmocr`

### ALTO output

- `OCR_ALTO_PROFILE`
  - default: `abbyy_like`
  - accepted values: `abbyy_like`, `legacy`

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

## Gradio Option Guide

This section combines the control reference and the rationale for when you
would actually use each option.

### Always-visible controls

These controls are always present in the app:

- `Image Input Folder`
- `Output Folder`
- `Use document orientation classification during inference.`
- `Use document image unwarping during inference.`
- `Generate a text file with OCR results for each image.`
- `Clean punctuation artifacts in TXT, PDF, and ALTO output.`

The current Gradio defaults for local OCR are:

- preprocessing starts with `background_flatten` selected
- `Crop black border before OCR.` starts enabled
- `Enable Post-OCR VL` starts enabled for `PP-OCRv5`

Rationale:

- the defaults lean toward historical newspaper scans, where mild background
  cleanup and black-border cropping often help more than they hurt
- post-OCR VL starts enabled on `PP-OCRv5` because this repo is tuned around
  catching suspicious OCR words after the first pass rather than trusting raw
  OCR output completely

### Generate TXT vs PDF/ALTO

`Generate a text file with OCR results for each image` means:

- generate TXT
- skip searchable PDF and ALTO generation

If this is off, the app generates searchable PDF and ALTO instead.

Rationale:

- choose TXT when you want quick inspection, downstream NLP, or lightweight
  comparison work
- choose PDF and ALTO when you need searchable facsimiles or richer layout
  output for archival workflows

### Clean Punctuation Artifacts

This option is available in the UI as:

- `Clean punctuation artifacts in TXT, PDF, and ALTO output`

Current behavior:

- it applies a conservative text cleanup based on the punctuation issue logic in
  [tools/compare_ocr_errors.py](tools/compare_ocr_errors.py)
- it runs on the saved OCR JSON before output files are generated
- it therefore affects:
  - TXT
  - searchable PDF text layer
  - ALTO XML text
- it does not change source images

The cleanup is intentionally narrow. It mainly targets:

- repeated punctuation like `..`, `,,`, `;;`
- line-start punctuation debris
- quote/punctuation tangles such as `.).)`
- obvious missing spaces after punctuation in OCR text

It is meant to reduce OCR punctuation artifacts without aggressively rewriting
word content.

Rationale:

- this is intentionally conservative because punctuation cleanup is far safer
  than automatic word rewriting
- it is most useful when OCR output is already readable but littered with small
  artifacts that hurt downstream searching or presentation

### Preprocessing

These controls are used by local OCR methods:

- `Image Preprocessing Steps`
  - choices: `light_clean`, `background_flatten`, `contrast`, `deskew`,
    `sauvola_binary`
- `Crop black border before OCR.`

Current defaults differ by entry point:

- Gradio UI: `background_flatten` starts selected
- `smoke_test.py`: defaults to no preprocessing unless you pass a preset or
  explicit `--preprocessing-step` flags

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

Why each preprocessing step exists:

- `light_clean`: removes a small amount of scanner grime and noise without
  changing the page too aggressively
- `background_flatten`: evens out newspaper-style background shading so text
  stands out more consistently
- `contrast`: boosts separation between text strokes and the page background
- `deskew`: helps when the whole page is rotated enough to hurt line finding
- `sauvola_binary`: useful for difficult foreground/background separation, but
  it is more aggressive and can damage subtle detail on already-good scans

Rationale:

- preprocessing exists because OCR quality usually fails first on image quality,
  not language modeling

### Tiling

These controls are used by local OCR methods, with some method-specific
visibility limits:

- `Enable Tiling`
- `Tile Strategy`
  - `grid`
  - `grid_3x2`
  - `doclayout_v3`
  - `grid_doclayout_v3`
  - `grid_3x2_doclayout_v3`
- `Tile Overlap (px)`
- `DocLayout Padding (px)`
- `DocLayout Threshold`
- `Use DocLayout recovery tile.`

Current app behavior:

- `PP-OCRv5`: full tiling UI is available
- `PP-StructureV3`: the Gradio app hides tiling controls and forces tiling off,
  even though the shared tiling service still supports fixed-grid tiling
- `PERO-OCR`: tiling UI is available
- `PaddleVL`: preprocessing controls stay visible but disabled, while tiling
  controls are hidden because the service consumes the original image
- `Auto Routing`: tiling controls are visible, but the submit path currently
  rejects source-image tiling

Why you would pick each tiling strategy:

- `grid`: a simple fixed split for long newspaper columns or tall pages
- `grid_3x2`: better for dense broadsheet-style layouts that need both vertical
  and horizontal subdivision
- `doclayout_v3`: best when page segmentation is the real problem and you want
  OCR to focus only on likely text regions
- `grid_doclayout_v3`: a compromise that first reduces page scale, then applies
  layout segmentation inside each large band
- `grid_3x2_doclayout_v3`: the most aggressive segmentation option and usually
  the best choice for very large, cluttered pages, though it is also slower

What the tiling knobs are for:

- `Tile Overlap (px)` protects against words being cut at tile edges, but too
  much overlap increases duplicate detections
- `DocLayout Padding (px)` gives each layout crop a little extra context so
  letters near edges are less likely to be clipped
- `DocLayout Threshold` tunes how readily layout regions are accepted
  lower values recover more content but can bring in more junk
- `Use DocLayout recovery tile.` adds a fallback crop so missed layout regions
  do not disappear entirely

Rationale:

- tiling exists because large historical pages often exceed the comfortable
  scale for a single OCR pass and benefit from smaller local crops
- the DocLayout-based strategies exist to reduce the amount of blank space,
  separators, illustrations, and unrelated regions sent through OCR

### PP-OCRv5 and PP-StructureV3 recognition controls

These PaddleOCR controls are shown for `PP-OCRv5`, `PP-StructureV3`, and
`Auto Routing`:

- `Select Language`
- `Text Detection Threshold`
- `Text Detection Limit Type`
- `Text Detection Limit Side Length`
- `Text Detection Box Threshold`
- `Text Detection Unclip Ratio`
- `Text Recognition Score Threshold`
- `Text Recognition Batch Size`
- `Use textline orientation classification during inference.`

`Max Side Limit` is only shown for `PP-OCRv5`.

Rationale:

- `Select Language` matters because recognition dictionaries and character
  priors strongly affect OCR accuracy
- the detection threshold controls trade recall against precision
  lower values recover faint text but also admit more false positives
- `Text Detection Limit Type` and `Text Detection Limit Side Length` control
  how aggressively the image is resized before detection, which affects both
  speed and small-text recovery
- `Max Side Limit` is a safety valve for very large pages where PaddleX would
  otherwise spend too much time or memory
- `Text Detection Box Threshold` and `Text Detection Unclip Ratio` tune how
  tight or generous the detected text regions are
- `Text Recognition Score Threshold` is a post-recognition filter; lower values
  keep more questionable text for later review, higher values are stricter
- `Text Recognition Batch Size` is mostly a throughput knob and may need to be
  reduced if GPU memory is tight
- `Use textline orientation classification during inference.` helps when lines
  may be rotated, vertical, or inconsistently aligned, but it adds overhead

### PP-OCRv5 post-OCR VL controls

These controls are shown only for `PP-OCRv5`:

- `Enable Post-OCR VL`
- `Post-OCR VL Backend`
  - `olmocr` via the `Ollama OpenAI` label
  - `paddlevl`
  - `both`
- `Words to Check`
  - `suspicious`
  - `low_confidence`
  - `all`
- `Enable Max Words Check`
- `Max Words to Check`
  - disabled unless `Enable Max Words Check` is turned on
  - when disabled, the effective limit is `0`, meaning no cap
- `Word Tile Padding (px)`
- `Low Confidence Threshold`
- `Apply accepted VL corrections to OCR text.`
- `Save post-OCR VL word tiles.`
- `Group same-line suspicious words for Ollama OpenAI.`
- `PaddleVL API URL`
- `Ollama OpenAI API URL`
- `Ollama OpenAI Model`
  - the dropdown allows custom model names served by the target endpoint

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

Rationale:

- `Enable Post-OCR VL` is for cases where first-pass OCR is decent overall but
  you still want targeted review of suspicious words
- `Post-OCR VL Backend` lets you trade local service behavior against external
  model behavior; `both` is useful for evaluation but slower and more expensive
- `Words to Check` is the main speed/coverage tradeoff
  `suspicious` is the practical default, `low_confidence` is broader, and
  `all` is mostly for debugging or benchmarking
- `Enable Max Words Check` is there to keep difficult pages from exploding into
  hundreds of VL calls
- `Word Tile Padding (px)` adds local visual context around a word crop
  too little padding can clip helpful context, too much can dilute the target
- `Low Confidence Threshold` controls how aggressively lines are treated as
  risky; higher values flag more text
- `Apply accepted VL corrections to OCR text.` is convenient for production
  output, but turning it off is safer when you are auditing model behavior
- `Save post-OCR VL word tiles.` is mainly for debugging and spot-checking
- `Group same-line suspicious words for Ollama OpenAI.` reduces repeated calls
  and gives the model line context, which often improves both speed and
  disambiguation
- the backend URL and model fields exist so you can swap endpoints without
  changing code
- this stage exists because OCR confidence alone does not catch every real
  error, especially on historically noisy pages
- targeted word or line review is cheaper than rerunning a whole page through a
  larger vision-language model
- grouped line crops help because neighboring words often disambiguate a bad
  token better than an isolated crop does

### PaddleVL-only controls

- `Use layout detection during inference.`

This control is shown for `PaddleVL` and `Auto Routing`.

Rationale:

- layout detection helps when the page contains multiple logical regions, mixed
  reading order, or non-body-text blocks
- disabling it can be useful on simpler pages if you want a faster, less
  segmented run

### PERO-only controls

- `PERO Python Module Path`
- `PERO config.ini Path`
- `PERO Device`
- `PERO GPU ID`
- `PERO Min Line Confidence`
- `Save native PERO PAGE XML beside JSON.`

Rationale:

- `PERO Python Module Path` exists because some environments use a local PERO
  checkout instead of an installed package
- `PERO config.ini Path` is required because PERO behavior is driven heavily by
  its config
- `PERO Device` and `PERO GPU ID` are there so you can match the runtime to the
  machine you are using
- `PERO Min Line Confidence` is a recall-vs-cleanliness control; raising it
  drops weaker lines, lowering it preserves more questionable content
- `Save native PERO PAGE XML beside JSON.` is mainly for debugging,
  interoperability, or comparing native PERO output against the repo’s
  normalized JSON

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

`smoke_test.py` covers almost all of the same option families as the Gradio
app, plus a couple of script-only details:

- `--preprocessing-preset` is a legacy single-flag shortcut that maps to
  multiple preprocessing steps unless `--preprocessing-step` is supplied
- `--preprocessing-step` can be repeated to build an explicit preprocessing
  pipeline
- `--disable-doclayout-recovery` turns off the fallback tile used to recover
  regions missed by PP-DocLayoutV3
- `--post-ocr-vl-no-apply`, `--post-ocr-vl-no-save-tiles`, and
  `--post-ocr-vl-no-group-lines` are negative flags for the corresponding
  positive Gradio checkboxes
- `PP-StructureV3` tiling is still reachable from `smoke_test.py`, even though
  the Gradio app hides it

Current smoke-test defaults are slightly different from the Gradio UI:

- preprocessing defaults to `none`
- post-OCR VL defaults to enabled for `PP-OCRv5`
- `--tile-strategy` defaults to `grid_doclayout_v3`

## Analysis And Helper Scripts

### Comparison Tool

The repo includes [tools/compare_ocr_errors.py](tools/compare_ocr_errors.py) for
Paddle-vs-ABBYY review.

Single-file CSV mode:

```bash
python tools/compare_ocr_errors.py \
  --paddle path/to/paddle.txt \
  --abbyy path/to/abbyy.txt \
  --output path/to/errors.csv
```

Folder-to-folder Excel mode:

```bash
python tools/compare_ocr_errors.py \
  --paddle-dir test-results/paddlecomp \
  --abbyy-dir "test-results/large - abby" \
  --excel-output test-results/paddle_vs_abbyy_errors_artifacts.xlsx
```

Current Excel behavior:

- one worksheet per matching Paddle TXT file
- if the target workbook already exists, the script reuses the existing sheet
  names, freeze panes, column widths, and row heights
- backup TXT files that include `b4-cleaned` in the name are ignored during
  directory comparisons

### Other Scripts

Most files under `services/` are imported by the app and CLI rather than used
as standalone commands. The direct script-style entry points in this repo are
primarily `app.py`, `launcher.py`, `smoke_test.py`, the helpers under
`tools/`, and the SMTP utility in `services/email_notify.py`.

#### Tile Preview Helper

[tools/save_tiles.py](tools/save_tiles.py) saves visual tile outputs using the
same tiling helpers the OCR pipeline uses.

Example:

```bash
python tools/save_tiles.py \
  testdata/source/large/oocihm.N_00155_18750610.4.jpg \
  output/tiles-demo \
  --strategies grid content_boxes \
  --grid-rows 3 \
  --grid-cols 2 \
  --overlap-px 96 \
  --padding-px 32
```

#### Human Comparison Workbook

[tools/compare_ocr_to_human.py](tools/compare_ocr_to_human.py) compares a human
transcription against ABBYY plus every `first-article.txt` found under
`test-results/paddlecomp`, then writes a shareable Excel workbook.

Example:

```bash
python tools/compare_ocr_to_human.py \
  --human "test-results/large- human/first-article.txt" \
  --abbyy "test-results/large - abby/first-article.txt" \
  --paddlecomp-dir test-results/paddlecomp \
  --output-dir test-results \
  --prefix first_article_ocr_comparison
```

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
  punctuation_cleanup.py   Conservative OCR text cleanup
  word_box_normalization.py Normalize word-box metadata
  ocr_defaults.py          Shared defaults
  ocr_logger.py            Logging and run summaries
  constants.py             Shared file-extension/constants
  email_notify.py          Optional email log sender
  fonts/                   Fonts used for PDF generation
tools/
  save_tiles.py            Save demo tiles using repo tiling logic
  compare_ocr_errors.py    Paddle vs ABBYY error workbook generator
  compare_ocr_to_human.py  Human-transcription comparison workbook generator
utils/                     Shared utilities
logs/                      Run logs
output/                    Typical output folder
```

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
