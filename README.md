# OCR Pipeline
This project is an OCR tool for internal CRKN staff. It provides PP-OCRv5,
PP-StructureV3, PaddleOCR-VL, and an optional PERO-OCR backend as internal OCR
solutions.

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
  The repo is currently pinned to `paddleocr==3.5.0` and `paddlex==3.5.2`.

## Quick Start
1. Create and activate a virtual environment.
WSL:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Windows:
```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2. Install dependencies.
```bash
pip install -r requirements.txt
```
WSL:
Install PaddlePaddle first, then install the WSL-friendly app requirements.
```bash
bash setup_wsl.sh cpu
# or
bash setup_wsl.sh gpu
```
The WSL setup script intentionally uses a Python 3.10-friendly Gradio pin and replaces the GUI OpenCV wheel with the headless build.
The repo now installs `paddlex[ocr]==3.5.2`, because the plain `paddlex` package is not enough for the OCR pipelines used by this app.
3. Start the PaddleOCR-VL service for `PaddleVL` and `Auto Routing`.
```bash
# Optional: copy .env.example to .env if you want to override the defaults.
docker compose up -d
```
4. Start the Gradio app from the repo root.
WSL:
```bash
python3 app.py
```
Windows:
```powershell
py app.py
```
You can also use `python launcher.py` or a PyInstaller build of `launcher.py` from the repo root.
5. Stop the application with `Ctrl+C`. Stop the VL service with `docker compose down`.

## Smoke Test
The repo now includes the `source` corpus under `testdata/source`, so the smoke test works without any machine-specific path.

To test the OCR pipeline in WSL without the Gradio UI, run:
```bash
python smoke_test.py
```
That defaults to `testdata/source/small/image.jpg`.

To target a different bundled folder or image, use a repo-relative path:
```bash
python smoke_test.py \
  --method PP-OCRv5 \
  --input testdata/source/large/oocihm.N_00155_18750610.4.jpg \
  --output output-smoke
```
Use `PaddleVL` or `Auto Routing` only after `docker compose up -d` is healthy.

## Plain-Language Option Guide

This app has a lot of knobs because newspaper OCR is not one problem. A small,
clean page needs different settings than a giant scan with black borders,
faint text, many columns, and warped paper. Start simple, then add options only
when the output tells you why they are needed.

### Processing Method

**PP-OCRv5**

Use this for most OCR runs. It is the main text-recognition path and usually
the best choice when you care about raw word count, TXT output, ALTO, and
searchable PDF text.

**PP-StructureV3**

Use this when layout structure matters more than plain OCR text, or when you
want to compare Paddle's structure pipeline against PP-OCRv5. It can work with
fixed-grid tiling, but its own layout parser makes PP-DocLayout-inclusive
tiling redundant. It is usually not the first choice for dense historical
newspaper text.

**PERO-OCR**

Use this when you want to compare against PERO-OCR's historical-document OCR
pipeline. PERO runs through its Python `PageParser` module using a PERO
`config.ini`; the app then converts the result into the same JSON, TXT, PDF,
and ALTO flow used by the Paddle backends. Preprocessing, black-border
cropping, fixed-grid tiling, and PP-DocLayoutV3 tiling are still available
before PERO receives each image or tile.

**PaddleVL**

Use this when you specifically want the PaddleOCR-VL service. It requires the
Docker service to be running. It is slower and service-based, but can help with
some layout-heavy or visually difficult documents.

**Auto Routing**

Use this only for mixed batches where you want the app to choose between local
Paddle OCR and PaddleVL based on image quality checks. For controlled testing,
pick the engine directly so results are easier to compare.

### Input And Output

**Image Input Folder**

Choose either one image or a folder. Folder runs preserve the folder structure
under the output directory.

**Output Folder**

Choose where JSON, TXT, PDF, ALTO, logs, and any preprocessed images should be
written.

### Common OCR Options

**Generate a text file with OCR results for each image**

Turn this on when you only want TXT output and want to skip searchable PDF and
ALTO generation. Leave it off when you want PDF and ALTO.

**Use document orientation classification**

Use this if pages may be rotated 90, 180, or 270 degrees. Leave it off for
known upright scans. It adds another model step.

**Use document image unwarping**

Use this for camera photos, book curvature, or visibly warped pages. Leave it
off for flat newspaper scans unless you are testing it directly. It can change
geometry and may not help clean scans.

**Use textline orientation classification**

Use this if individual text lines may be rotated or vertical. Leave it off for
normal horizontal English newspaper pages.

### Image Preprocessing

Preprocessing runs before local OCR and before tiling. If any
preprocessing step is selected, the app saves a PNG under `_preprocessed/` next
to the OCR output so you can inspect what Paddle saw.

**Crop black border before OCR**

Use this for scans with a dark scanner bed or black frame around the page. The
app crops to the detected page area for OCR, then offsets OCR boxes back into
the original image coordinates for PDF and ALTO. This is safer than painting
the border white because it keeps final coordinates aligned to the original
image.

**Light clean**

Use this for mild speckle or scanner noise. It applies a conservative denoise.
It is safe to combine with `Background flatten` and `Contrast / CLAHE`.

**Background flatten**

Use this for uneven page background, shadows, yellowing, or gray paper. This is
often the best first preprocessing step for historical newspapers.

**Contrast / CLAHE**

Use this when text is faint or low contrast. It boosts local contrast. It is a
good partner for `Background flatten`.

**Deskew**

Use this only when the page is visibly tilted. If the page is already straight,
leave it off. The deskew step is conservative, but any rotation can still make
box alignment more delicate.

**Sauvola binary**

Use this as an experiment when faint text is still missed after background
flattening and contrast. It can improve separation between text and background,
but it can also damage punctuation, thin letters, and textured newspaper
print. Do not make this your first default for dense newspaper pages.

Recommended preprocessing starting points:

```text
Clean scans:
  No preprocessing

Historical newspaper with black scanner border:
  Crop black border before OCR

Uneven newspaper background:
  Crop black border before OCR
  Light clean
  Background flatten
  Contrast / CLAHE

Very faint text experiment:
  Crop black border before OCR
  Light clean
  Background flatten
  Sauvola binary
```

### Tiling Options

Tiling splits a large page into smaller OCR jobs, then merges the results back
into one page. Use tiling for large scans, dense newspapers, and any image
where Paddle logs messages like:

```text
Resized image size (...) exceeds max_side_limit of 4000. Resizing to fit within limit.
```

**Tile source images before OCR**

Turn this on for large newspaper scans. Leave it off for small images or quick
sanity checks.

**Tile Strategy**

`Fixed 3x1 grid`

Splits the page into three horizontal bands. Useful for tall pages, but it can
still leave very wide tiles on large broadsheet scans.

`Fixed 3x2 grid`

Splits the page into three rows and two columns. This is a good default for
large newspaper pages because it reduces both tile height and tile width.

`PP-DocLayoutV3 Layout`

Uses Paddle's layout detector to find layout regions and OCR those regions.
This can improve layout targeting, but pure layout-only tiling may miss content
if layout detection misses a region. This option is shown for PP-OCRv5 and
PERO-OCR, not PP-StructureV3.

`3x1 + PP-DocLayoutV3`

First splits into three horizontal bands, then runs DocLayout inside each band.
This was a strong earlier default for tall pages. This option is shown for
PP-OCRv5 and PERO-OCR, not PP-StructureV3.

`3x2 + PP-DocLayoutV3`

First splits into a 3 by 2 grid, then runs DocLayout inside each tile. This is
the best starting point for very large, dense newspaper scans because it keeps
tiles smaller and still uses layout detection inside each tile. This option is
shown for PP-OCRv5 and PERO-OCR, not PP-StructureV3.

**Tile Overlap**

Overlap gives OCR a little repeated context at tile boundaries. Use `96` px as
a good default. If you see duplicated short fragments near tile edges, try
lowering it. If you see missing words at tile boundaries, try raising it.

**DocLayout Padding**

Padding expands each detected layout region before OCR. Use `32` px as a good
default. Raise it if text is clipped at region edges. Lower it if neighboring
columns bleed into each other.

**DocLayout Threshold**

Lower values detect more layout regions. Use `0.20` as a recall-friendly
default. Raise it if DocLayout produces too many noisy regions. Lower it if
content is being missed.

**Use DocLayout recovery tile**

Leave this on for important OCR runs. It adds a fallback OCR pass to recover
areas DocLayout may have missed.

### PERO-OCR Options

These controls map to PERO's Python/module and `parse_folder.py`/`PageParser`
surface rather than invented OCR thresholds.

**PERO Python Module Path**

Path to a local `pero-ocr` checkout when `pero_ocr` is not installed into the
active Python environment. Windows `C:\...` paths are accepted when running
under WSL.

**PERO config.ini Path**

Required. This is the PERO engine configuration passed to
`PageParser(config, config_path=...)`. Relative paths inside the config are
resolved against the config file's folder.

**PERO Device**

Matches PERO's `parse_folder.py --device` choices: `gpu` or `cpu`.

**PERO GPU ID**

Optional. Matches PERO's `parse_folder.py --gpu-id`; leave blank to use
PERO/PyTorch's default CUDA device.

**PERO Min Line Confidence**

Overrides PERO `PAGE_PARSER` option
`FILTER_CONFIDENT_LINES_THRESHOLD`. Use `-1` to leave PERO confidence filtering
disabled.

**Save native PERO PAGE XML**

Writes PERO's own PAGE XML beside the app JSON. The app still generates its
normal PDF and ALTO outputs from the normalized JSON.

### Post-OCR VL Word Check

This optional PP-OCRv5 post-processing step uses PP-OCR word boxes
(`text_word` / `text_word_boxes`) to crop small word tiles from the source page.
Each tile can be sent to either PaddleOCR-VL or an OpenAI-compatible Ollama
server. The checker records all backend responses in the JSON under
`post_ocr_vl_check`, saves the word crops under `_post_ocr_vl_tiles`, and can
rewrite accepted one-word corrections into `text_word` and `rec_texts` before
TXT, PDF, and ALTO are generated.

**Post-OCR VL Backend**

Choose `PaddleVL` to reuse the existing local `/layout-parsing` service, or
`Ollama OpenAI` to call an OpenAI-compatible `/chat/completions` server running
models such as `fredrezones55/chandra-ocr-2`, `richardyoung/olmocr2`, or
`deepseek-ocr:3b`.

**Words to Check**

`Suspicious words` ranks odd-looking or low-confidence word boxes first.
`Low-confidence lines` checks words whose PP-OCR line score is below the
threshold. `All words in reading order` is exhaustive and can be slow.

**Apply accepted VL corrections**

Leave this on to update the normalized OCR JSON when the VL response is a clean
single-word correction. Turn it off to save audit metadata and tiles only.

### Paddle OCR Detection Options

These options affect Paddle's text detector and recognizer. For most newspaper
testing, start with the current defaults and change one thing at a time.

**Language**

Choose the expected OCR language. Use English for English newspaper pages. If
left blank, Paddle's default behavior is used.

**Text Detection Threshold**

Lower means more text candidates are kept. Higher means fewer, cleaner
candidates. Use `0.20` for difficult newspaper scans where recall matters.

**Text Detection Limit Type**

Use `max` for large scans. It limits based on the long side of the image or
tile.

**Text Detection Limit Side Length**

This is Paddle's target side length before detection. Use `9984` when tiling is
on and you want to preserve detail.

**Text Detection Max Side Limit**

This is the hard maximum side limit used to avoid PaddleX resizing warnings and
memory blowups. Use `4000` as the safe default. If you see max-side warnings,
prefer smaller tiles rather than raising this first. This control is shown for
PP-OCRv5 only.

**Text Detection Box Threshold**

Controls which detected text boxes survive post-processing. Use `0.30` as a
stable default. Lower it only if faint words are being missed and noise is
acceptable.

**Text Detection Unclip Ratio**

Expands detected text boxes. Use `2.0` as a default. Raise it if boxes are too
tight and characters are clipped. Lower it if adjacent lines or columns merge.

**Text Recognition Score Threshold**

Controls filtering by recognition confidence. Use `0.0` when word count and
recall are the priority. Raise it only when you want to suppress low-confidence
text and are willing to lose words.

**Text Recognition Batch Size**

Use `32` on CPU or memory-limited machines. On a GPU workstation, try `64` if
memory is comfortable. Batch size mostly affects speed, not OCR quality.

### Recommended Recipes

For `testdata/source/large/oocihm.N_00155_18750610.4.jpg`, start here:

```text
Method: PP-OCRv5
Language: English
Crop black border before OCR: on
Image preprocessing steps: Light clean, Background flatten, Contrast / CLAHE
Tile source images: on
Tile strategy: 3x2 + PP-DocLayoutV3
Tile overlap: 96
DocLayout padding: 32
DocLayout threshold: 0.20
DocLayout recovery tile: on
Text detection threshold: 0.20
Text detection limit type: max
Text detection limit side length: 9984
Text detection max side limit: 4000
Text detection box threshold: 0.30
Text detection unclip ratio: 2.0
Text recognition score threshold: 0.0
Text recognition batch size: 32 on CPU, 64 on GPU
```

For fast debugging:

```text
Method: PP-OCRv5
Generate TXT: on
Tile source images: on
Tile strategy: Fixed 3x2 grid
Image preprocessing: off
```

For comparing against ABBYY:

```text
Generate PDF/ALTO: on
Generate TXT: off
Keep Text recognition score threshold at 0.0
Compare word count first, then reading order and box placement
Change only one option between runs
```

### CLI Examples

Run the large newspaper page with the recommended local OCR settings:

```bash
python smoke_test.py \
  --method PP-OCRv5 \
  --input testdata/source/large/oocihm.N_00155_18750610.4.jpg \
  --output output-smoke-large \
  --language en \
  --crop-black-border \
  --preprocessing-step light_clean \
  --preprocessing-step background_flatten \
  --preprocessing-step contrast \
  --tile-source-images \
  --tile-strategy grid_3x2_doclayout_v3 \
  --tile-overlap-px 96 \
  --tile-doclayout-padding-px 32 \
  --tile-doclayout-threshold 0.20 \
  --text-det-thresh 0.20 \
  --text-det-limit-type max \
  --text-det-limit-side-len 9984 \
  --text-det-max-side-limit 4000
```

Run PP-OCRv5 with the post-OCR VL word checker in audit mode:

```bash
python smoke_test.py \
  --method PP-OCRv5 \
  --input testdata/source/large/oocihm.N_00155_18750610.4.jpg \
  --output output-smoke-post-vl \
  --tile-source-images \
  --tile-strategy grid_3x2_doclayout_v3 \
  --post-ocr-vl-check \
  --post-ocr-vl-backend paddlevl \
  --post-ocr-vl-candidate-mode suspicious \
  --post-ocr-vl-max-words 25 \
  --post-ocr-vl-no-apply
```

Run the same page through PERO-OCR with the shared preprocessing/tiling layer:

```bash
python smoke_test.py \
  --method PERO-OCR \
  --input testdata/source/large/oocihm.N_00155_18750610.4.jpg \
  --output output-smoke-pero \
  --pero-source-path /mnt/c/Users/BrittnyLapierre/Documents/pero-ocr \
  --pero-config-path /path/to/pero-engine/config.ini \
  --pero-device gpu \
  --tile-source-images \
  --tile-strategy grid_3x2_doclayout_v3 \
  --tile-overlap-px 96 \
  --tile-doclayout-padding-px 32 \
  --tile-doclayout-threshold 0.20
```

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

## Environment Notes
- The launcher now resolves the repo directory dynamically, so it no longer depends on `C:\CRKN Projects\paddle_pipeline`.
- `compose.yaml` now has default image tags and mounts the repo-local `pipeline_config_vllm.yaml` and `pipeline_config_fastdeploy.yaml`.
- If the VL API is not local, set `PADDLE_VL_API_URL`, for example `http://127.0.0.1:8080`.
- `requirements-wsl.txt` avoids the Windows-only `safetensors` wheel and uses a headless OpenCV build that is friendlier in WSL.
- The engine includes a small PaddleX compatibility patch so `opencv-contrib-python-headless` is accepted in WSL.
- The Gradio app now starts even if `tkinter` is unavailable; in that case, paste `/mnt/c/...` paths manually instead of using the browse buttons.
- The bundled test corpus lives under `testdata/source`, so smoke tests are portable across machines.
- PERO-OCR is optional. Install `pero-ocr` and its runtime dependencies into
  the active environment or point the app at a local checkout with
  `PERO_OCR_SOURCE_PATH` / the UI field, and provide a PERO engine `config.ini`.
- Post-OCR Ollama OpenAI checks use `OLLAMA_OPENAI_API_URL`,
  `OLLAMA_OPENAI_MODEL`, and optional `OLLAMA_OPENAI_API_KEY` when calling
  external providers. The legacy `OLMOCR_*` env vars still work too.
  
## Project Structure

```text
OCR-Tool/
├── app.py                               # Main application entry point
├── testdata/
│   └── source/                          # Bundled OCR smoke-test corpus
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
│   ├── pero_ocr.py                      # Optional PERO-OCR PageParser integration
│   ├── ocr_vl.py                        # PaddleOCR-VL OCR processing
│   ├── post_ocr_vl.py                   # Optional PP-OCR word-crop VL correction
│   ├── image_preprocessing.py           # Optional crop, cleanup, contrast, binarization, and deskew steps before OCR
│   ├── tile_ocr.py                      # Fixed-grid and PP-DocLayoutV3 tiled OCR logic
│   ├── alto_generation.py               # Generates ALTO XML files from OCR results
│   ├── pdf_generation.py                # Generates searchable PDF files from OCR results
│   ├── txt_generation.py                # Generates TXT output files from OCR results
│   ├── email_notify.py                  # Sends summary and error notification emails
│   ├── ocr_logger.py                    # Logging configuration and OCR log management
│   ├── ocr_defaults.py                  # Shared default values for UI, CLI, logs, and Paddle setup
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
