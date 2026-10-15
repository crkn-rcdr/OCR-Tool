# CRKN Historical OCR & Multimodal Preservation Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PaddleOCR](https://img.shields.io/badge/OCR-PP--OCRv6-orange.svg)](https://github.com/PaddlePaddle/PaddleOCR)
[![Indra OCR](https://img.shields.io/badge/CPU_Engine-Indra_OCR-informational.svg)](https://github.com/crkn-rcdr/paddle_pipeline)
[![Dots OCR](https://img.shields.io/badge/VLM-Dots%20OCR%207B-brightgreen.svg)](https://github.com/redcaps-ai)
[![ALTO XML 4.2](https://img.shields.io/badge/Metadata-ALTO%20XML%204.2-blueviolet.svg)](https://www.loc.gov/standards/alto/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, high-throughput OCR and layout preservation engine developed for the **Canadian Research Knowledge Network (CRKN / RCDR)**. Designed specifically for historical Canadian newspapers, 18th–20th-century monographs, microfilm reels (Heritage collections), and bilingual legislative gazettes.

The pipeline combines ultra-fast TensorRT/FastDeploy OCR (**PP-OCRv6**), high-speed multi-core CPU OCR (**Indra OCR**), autonomous microfilm pre-processing, layout-aware multi-column sorting, and post-OCR visual arbitration using **Dots OCR (7B Vision-Language Model)** and a curated **Canadiana Historical Gazetteer**.

---

## ⚡ Quick Start (60 Seconds)

### Option 1: Interactive Web UI (Gradio)
Launch the user-friendly interface for single pages or folder batches:

```powershell
# In PowerShell (using the project virtual environment):
.\venv\Scripts\python.exe app.py
```
Open **`http://127.0.0.1:7860`** in your browser.

![CRKN Web Interface](images/01_web_interface.png)

> [!TIP]
> **Dynamic Model-Dependent UI**: The interface automatically tailors available options based on the chosen OCR Model:
> - **PP-OCRv6 (GPU/CPU)**: Exposes full controls: Language auto-detection, $3\times 2$ quadrant tiling strategy, adaptive preprocessing (black border, CLAHE, stroke reconnection, deskew, micro-print upscaling), and VLM post-processing arbitration.
> - **Indra OCR (Fast CPU)**: Streamlines the UI to Language, Plain Text, and HTML Results Dashboard (hiding tiling and pre/post-processing toggles since Indra runs native full-page CPU inference).
> - **PaddleVL**: Shows Language, Plain Text, and HTML Results Dashboard.
> - **Dots OCR 7B / OLM OCR**: Displays Plain Text and HTML Results Dashboard (language is inferred visually by the VLM).
> - **HTML Results Dashboard**: Enabled by default; automatically generates a standalone visual inspection dashboard with confidence-colored bounding boxes and opens in a new browser tab upon completion.

---

### Option 2: Full Benchmark & ABBYY Comparison Run (CLI)
Run the automated end-to-end batch script to process the test dataset and immediately generate an updated comparison report:

```powershell
# Default run (targets Newspapers folder):
.\venv\Scripts\python.exe tools/run_full_benchmark.py

# Or target the entire test dataset (all collections including books, serials, and reels):
.\venv\Scripts\python.exe tools/run_full_benchmark.py "C:\Users\blapierre\Documents\PDF Test"
```

This single command:
1. Processes documents from `C:\Users\blapierre\Documents\PDF Test\Newspapers` (or the supplied custom input directory).
2. Generates Searchable PDFs, ALTO XML 4.2, and JSON files in `C:\Users\blapierre\Desktop\PDF Test - Final`.
3. Runs the lexical comparator against baseline ABBYY FineReader XMLs.
4. Generates an interactive visual report at **`compare/index.html`**, summary metrics at `compare/comparison_summary.csv`, and automatically syncs publication dashboards to the `OCR-Evaluations` repository.

---

## Table of Contents

1. [Pipeline Architecture at a Glance](#pipeline-architecture-at-a-glance)
2. [End-to-End Processing Workflow](#end-to-end-processing-workflow)
3. [Supported OCR & Multimodal Engines: Architecture & Logic Deep-Dive](#supported-ocr--multimodal-engines-architecture--logic-deep-dive)
4. [Stage 1: Adaptive Preprocessing Pipeline ("Pre Process")](#stage-1-adaptive-preprocessing-pipeline-pre-process)
5. [Stage 2: Core Recognition, Tiling & Script Dispatch](#stage-2-core-recognition-tiling--script-dispatch)
6. [Stage 3: Spatial & Column-Aware Reading Order Sorter](#stage-3-spatial--column-aware-reading-order-sorter)
7. [Stage 4: Linguistic Arbitration & VLM Delegation ("Post Process")](#stage-4-linguistic-arbitration--vlm-delegation-post-process)
8. [Stage 5: Output Generation & 1:1 Coordinate Fidelity](#stage-5-output-generation--11-coordinate-fidelity)
9. [Interactive HTML Results Dashboard (Standalone Visual Verification)](#interactive-html-results-dashboard-standalone-visual-verification)
10. [Benchmarking & ABBYY FineReader Comparison](#benchmarking--abbyy-finereader-comparison)
11. [Headless Batch Runner CLI](#headless-batch-runner-cli)
12. [Setup & Installation](#setup--installation)
13. [Repository Layout](#repository-layout)

---

## Pipeline Architecture at a Glance

The pipeline operates across 5 interconnected phases designed to outperform traditional OCR engines (such as ABBYY FineReader) on degraded historical print:

```mermaid
flowchart TD
    subgraph Input ["1. Source Material"]
        A["Input Images / Scans<br/>(Microfilm Reels, Broadsheets, Books)"]
    end

    subgraph Preprocessing ["2. Stage 1: Adaptive Preprocessing (Pre Process Checkbox)"]
        B1["Microfilm Signature Detector<br/>(Laplacian Variance + Percentile Analysis)"]
        B2["Black Border / Carrier Cropping<br/>(Pixel Offset [dx, dy] Tracking)"]
        B3["Adaptive Contrast & Background Flattening<br/>(Dynamic CLAHE + Luminance Inversion/Flattening)"]
        B4["Micro-Stroke Reconnection<br/>(2x2 Morphological Closing for Broken Print)"]
        B5["Dynamic Auto-Deskew<br/>(Sub-5 Degree Baseline Straightening)"]
        B6["Table Ruling Line Suppression<br/>(Vertical Ledger Gutter Protection)"]
        B7["Adaptive 1.5x Micro-Print Upscaling<br/>(For Font Heights < 18px + Coordinate Inversion)"]
    end

    subgraph Recognition ["3. Stage 2: Core Recognition, Tiling & Script Dispatch"]
        C1{"Script & Language Detection<br/>services/language_detection.py"}
        C2["Traditional Chinese (sfu.00001)<br/>Bilingual French Superset (sru, qmbn, oocihm.8)<br/>Standard English"]
        C3{"Auto-Detect Tiling?<br/>services/tile_ocr.py<br/>(≥ 18 MP or ≥ 4500x3200px)"}
        C4["3x2 Grid Quadrant Tiling<br/>(15% Overlap Margin + NMS Merging)"]
        C5["Native Full-Page DBNet Inference<br/>(PP-OCRv6 DBNet++ + SVTR-LCNet)"]
    end

    subgraph Spatial ["4. Stage 3: Spatial Layout & Reading Order"]
        D1["Whitespace Gutter Profiler<br/>(1D Horizontal Density Convolution k=31)"]
        D2["Full-Width Banner Isolation<br/>(Spanning >= 75% Active Text Width)"]
        D3["Multi-Column Line Sorter<br/>(Top-to-Bottom within Column, Left-to-Right Flow)"]
    end

    subgraph PostProcessing ["5. Stage 4: Linguistic Arbitration & VLM Delegation (Post Process Checkbox)"]
        E1["Line Quality Evaluator<br/>(Mean Line Score < 0.78)"]
        E2["Line-Level VLM Delegation<br/>(Dots OCR 7B Full-Line Crop)"]
        E3["Token Geometry & Syntax Refiner<br/>(Whitespace / Punctuation Splitting, De-compounding)"]
        E4["Disputed Word Evaluator<br/>(Conf < 0.80 or Lexical Failure)"]
        E5["Word-Level VLM Delegation<br/>(Dots OCR 7B Targeted Token Crop)"]
        E6["Lexicon-Constrained Decoding<br/>(1-Edit Levenshtein + Optical Confusion rn↔m, cl↔d)"]
        E7["Historical Canadiana Gazetteer<br/>(Colonial Settlements, French Accents, Legal Terms)"]
        E8["Hyphen & Accent Consolidator<br/>(Cross-Line Hyphens + Recombines Split Accents)"]
    end

    subgraph Output ["6. Stage 5: Preservation Outputs (1:1 Pixel Coordinates)"]
        F1["Normalized OCR JSON"]
        F2["ALTO XML 4.2<br/>(Auto-Split Words, Reading Order)"]
        F3["Searchable PDF<br/>(Exact 1:1 Pixel Coordinate Alignment)"]
        F4["Plain Text (TXT)"]
    end

    A --> B1
    B1 --> B2 --> B3 --> B4 --> B5 --> B6 --> B7
    B7 --> C1 --> C2 --> C3
    C3 -- "Broadsheet (≥ 18 MP)" --> C4 --> C5
    C3 -- "Standard (< 18 MP)" --> C5
    C5 --> D1 --> D2 --> D3
    D3 --> E1
    E1 -- Line Score < 0.78 --> E2
    E1 -- High Line Score --> E3
    E3 --> E4
    E4 -- Conf < 0.80 / Disputed --> E5
    E4 -- Clean Token --> E6
    E2 & E5 --> E6
    E6 --> E7 --> E8
    E8 --> F1 & F2 & F3 & F4
```

---

## End-to-End Processing Workflow

The following sequence diagram outlines how data flows through the modular Python services during a single-page execution:

```mermaid
sequenceDiagram
    autonumber
    actor Staff as CRKN Staff / CLI Runner
    participant Pre as services/image_preprocessing.py
    participant Lang as services/language_detection.py
    participant Core as services/ocr_pp.py
    participant Layout as services/alto_generation.py
    participant VLM as services/post_ocr_vl.py (Dots OCR)
    participant Norm as services/word_box_normalization.py
    participant Lex as services/historical_lexicon.py
    participant Out as services/pdf_generation.py

    Staff->>Pre: Submit scan with "Pre Process" + "Post Process" enabled
    Pre->>Pre: Detect microfilm profile & flatten background luminance
    Pre->>Pre: Crop black carrier & record pixel offset [dx, dy]
    Pre->>Pre: If character height < 18px, upscale 1.5x with bicubic interpolation
    Pre->>Pre: Reconnect broken letter strokes (2x2 morphological closing)
    Pre->>Pre: Straighten baseline skew (0.35° <= theta <= 5.0°)
    Pre->>Lang: Inspect file path & collection markers (Bilingual FR, Chinese CHT, EN)
    Lang-->>Core: Assign model ("chinese_cht", "fr" Latin superset, or "en")
    Core->>Core: Execute PP-OCRv6 DBNet++ & SVTR-LCNet recognition
    Core->>Core: Invert 1.5x upscale coordinates back to original image scale
    Core->>Layout: Pass raw bounding boxes & confidence scores
    Layout->>Layout: Convolve whitespace gutters & organize lines into columns
    Layout->>VLM: Evaluate line scores (< 0.78) -> delegate full line to Dots OCR 7B
    Layout->>Norm: Split run-together tokens (whitespace, punctuation, compound words)
    Norm->>Lex: Regularize low-conf tokens (< 0.85) via 1-edit Levenshtein & confusion matrix
    Lex->>VLM: Disputed tokens failing lexicon (< 0.80) -> targeted Dots OCR VLM crop
    Norm->>Norm: Recombine split French accents (talage + é -> étalage) & cross-line hyphens
    Norm->>Core: Return normalized word boxes & verified transcriptions
    Core->>Out: Build Searchable PDF with invisible text layer
    Core->>Layout: Build schema-valid ALTO XML 4.2 with discrete word tokens
    Out-->>Staff: Deliver outputs with 100% pixel coordinate precision
```

---

## Supported OCR & Multimodal Engines: Architecture & Logic Deep-Dive

The pipeline integrates five distinct recognition engines tailored for different preservation requirements, hardware configurations, and document degradation levels:

```mermaid
flowchart TD
    In["Input Document Scan"] --> EngineSwitch{"Engine Selection<br/>(UI / CLI --method)"}

    EngineSwitch -->|"PP-OCRv6 (Default)"| PP["PaddleOCR Engine (services/ocr_pp.py)<br/>• FastDeploy DBNet++ & SVTR-LCNet (GPU/CPU)<br/>• Full Stage 1 Archival Preprocessing Pipeline<br/>• Auto-Detect 3x2 Quadrant Grid Tiling (≥18 MP)<br/>• Whitespace Gutter Profiler & Reading Order Sorter<br/>• Dual-Tier Dots OCR 7B VLM Post-Arbitration<br/>• Canadiana Historical Gazetteer Regularization<br/>• ALTO XML 4.2 + PDF + JSON + HTML Dashboard"]

    EngineSwitch -->|"Indra OCR (Fast CPU)"| Indra["Indra OCR Engine (services/ocr_indra.py)<br/>• Highly Optimized Multi-Process ONNX Runtime CPU Engine<br/>• Zero NVIDIA GPU / CUDA Requirement (0.5–1.5s / page)<br/>• Native Full-Page Text Detection & Recognition<br/>• Self-Contained Direct Word-Box Coordinate Alignment<br/>• ALTO XML 4.2 + PDF + JSON + HTML Dashboard"]

    EngineSwitch -->|"Dots OCR (7B VLM)"| Dots["Dots OCR Engine (services/ocr_dots.py)<br/>• 7B Parameter Vision-Language Model (OpenAI API)<br/>• Holistic Contextual Semantic Decoding<br/>• Ideal for Severely Degraded Scans, Faded Ink, Broken Type<br/>• ALTO XML 4.2 + PDF + JSON + HTML Dashboard"]

    EngineSwitch -->|"PaddleVL"| VL["PaddleOCR-VL Engine (services/ocr_vl.py)<br/>• PaddleX Multimodal Layout Parsing Server<br/>• Interleaved Table, Graphic, and Ad Boundary Segmentation<br/>• ALTO XML 4.2 + PDF + JSON + HTML Dashboard"]

    EngineSwitch -->|"olmOCR"| OLM["olmOCR Engine (services/ocr_olm.py)<br/>• 7B Vision LLM fine-tuned for Document Transcription (vLLM)<br/>• Direct Pixel-to-Markdown Reading Order Synthesis<br/>• ALTO XML 4.2 + PDF + JSON + HTML Dashboard"]
```

### 1. PaddleOCR (PP-OCRv6) *(Recommended Production Default)*
- **Primary Modules**: `services/ocr_pp.py`, `services/image_preprocessing.py`, `services/tile_ocr.py`, `services/post_ocr_vl.py`.
- **Underlying Architecture**: Dual-stage neural pipeline pairing **DBNet++** (Differentiable Binarization text detector with deformable convolutions) and **SVTR-LCNet** (Lightweight Vision Transformer text recognizer).
- **Execution & Acceleration**: Accelerated via NVIDIA CUDA, TensorRT, and Paddle FastDeploy with automatic multi-threaded CPU fallback.
- **Execution Logic**:
  1. **Stage 1 Preprocessing**: Automatically engages microfilm background flattening, carrier black border removal, $2\times 2$ stroke reconnection, deskew ($0.35^\circ$–$5.0^\circ$), ledger ruling line suppression, and $1.5\times$ micro-print upscaling when enabled.
  2. **Script & Language Dispatch**: Heuristically selects models: Traditional Chinese (`chinese_cht`) for Chinese-Canadian diaspora presses, Bilingual French Latin superset (`fr`) with full diacritics for Canadian federal/provincial records, or English (`en`).
  3. **Resolution-Aware Quadrant Tiling**: Evaluates image resolution. Pages $\ge 18\text{ MP}$ or $\ge 4500\times 3200\text{px}$ (e.g. broadsheet newspapers) are dynamically sliced into a $3\times 2$ quadrant grid with a 15% overlap margin and merged via Non-Maximum Suppression (NMS). Standard pages ($< 18\text{ MP}$) are executed natively full-page.
  4. **Spatial Layout & Column Sorting**: Projects word bounding boxes horizontally through a 31-pixel Gaussian convolution filter to identify whitespace column gutters and isolate full-width banners.
  5. **Post-OCR Arbitration & Gazetteer**: Degraded lines ($< 0.78$) and disputed words ($< 0.80$) are submitted to Dots OCR 7B. Tokens undergo punctuation-bound splitting, display ad de-compounding, cross-line hyphen reconciliation, 1-edit Levenshtein lexical regularizing, and verification against the Canadiana Historical Gazetteer with 54-language zero-corruption guardrails.
  6. **Preservation Outputs**: Generates sub-pixel Searchable PDFs, schema-valid ALTO XML 4.2, normalized JSON, plain text, and the interactive HTML Results Dashboard.

### 2. Indra OCR (Fast CPU)
- **Primary Modules**: `services/ocr_indra.py`.
- **Underlying Architecture**: Lightweight, high-throughput multi-process ONNX Runtime CPU engine.
- **Execution & Acceleration**: Pure CPU execution utilizing an asynchronous multi-process worker pool across available system CPU cores (~0.5–1.5s per page). Requires zero NVIDIA GPU or CUDA drivers.
- **Execution Logic**:
  1. **Native Full-Page Processing**: Executes text detection and character recognition in a single unified pass over the full-resolution image without requiring external quadrant slicing or seam merging.
  2. **Self-Contained Pipeline**: Bypasses external microfilm preprocessing and secondary VLM delegation, eliminating pipeline overhead and external microservice dependencies.
  3. **Direct Coordinate Mapping**: Directly captures line geometries and word token boundaries with confidence scores, normalizing coordinates with sub-pixel fidelity.
  4. **Preservation Output Generation**: Directly interfaces with `services/pdf_generation.py`, `services/alto_generation.py`, and `services/results_dashboard.py` to produce standard Library of Congress ALTO XML 4.2, Searchable PDFs, normalized JSON, optional plain text, and the standalone HTML Results Dashboard.
  5. **Best Used For**: High-throughput processing on CPU workstations, laptops without discrete NVIDIA GPUs, headless CI/CD environments, and rapid bulk digitizing where 1–2 second per-page CPU throughput is essential.

### 3. Dots OCR (7B Vision-Language Model)
- **Primary Modules**: `services/ocr_dots.py`, `services/post_ocr_vl.py`.
- **Underlying Architecture**: 7-Billion parameter multimodal Vision-Language Model (VLM) running via an OpenAI-compatible local server.
- **Execution & Acceleration**: Neural vision-language transformer inference (~150ms per crop, ~4.0–8.0s per full page).
- **Execution Logic**:
  1. **Standalone OCR Engine**: Analyzes full-page images holistically, translating visual features into rich formatted text and markdown. Overcomes severe paper yellowing, smudged ink, bleed-through, and archaic typefaces where conventional bounding-box detectors fail.
  2. **PP-OCRv6 Dual-Tier Post-Processor**: Operates as an automated visual arbiter for PP-OCRv6:
     - *Line-Level Delegation*: Lines with mean confidence $< 0.78$ are cropped and sent to Dots OCR to decode full sentential context.
     - *Word-Level Crop Delegation*: Individual disputed tokens failing lexicon validation with confidence $< 0.80$ are extracted with context-padded crops and resolved.
  3. **Preservation Outputs & Audit Logging**: Emits schema-compliant ALTO XML 4.2, Searchable PDFs, normalized JSON, optional plain text, structured markdown, the standalone HTML Results Dashboard, and generates an audit log (`.post_ocr_vl.csv`) documenting every line or token modified by the model.

### 4. PaddleVL (PaddleOCR-VL / PaddleX)
- **Primary Modules**: `services/ocr_vl.py`.
- **Underlying Architecture**: PaddleX multimodal layout parser and visual document information extractor.
- **Execution & Acceleration**: Containerized Docker microservice (`docker compose up -d`) running on port `8080`.
- **Execution Logic**:
  1. **Visual Region Classification**: Detects and segments high-level document structures: tabular data, classified advertisement frames, illustrative figures, titles, and text paragraphs.
  2. **Interleaved Content Parsing**: Separates tabular columns and multi-tiered classified ads from continuous narrative body text.
  3. **Preservation Outputs**: Generates sub-pixel Searchable PDFs, schema-valid ALTO XML 4.2, normalized JSON, optional plain text, structured layout blocks, and the standalone HTML Results Dashboard.

### 5. OLM OCR (olmOCR 7B)
- **Primary Modules**: `services/ocr_olm.py`.
- **Underlying Architecture**: 7B Vision-Language Model fine-tuned specifically for dense historical document transcription and reading order synthesis, served via a local vLLM endpoint on port `8000`.
- **Execution & Acceleration**: Autoregressive multimodal sequence-to-sequence generation (~4.0–8.0s per page).
- **Execution Logic**:
  1. **Direct Pixel-to-Markdown Synthesis**: Bypasses heuristic coordinate sorting by training directly on human reading sequences, generating natural flow across multi-column newspapers, academic journals, marginalia, and footnoted texts.
  2. **Structural Formatting**: Faithfully reproduces inline mathematical formulas, tables, and section hierarchies in standard Markdown.
  3. **Preservation Outputs**: Produces Searchable PDFs, ALTO XML 4.2, normalized JSON, optional plain text, structured markdown, and the interactive HTML Results Dashboard.

---

## Stage 1: Adaptive Preprocessing Pipeline ("Pre Process")

Historical Canadian microfilms (such as Library and Archives Canada / LAC reels) exhibit optical artifacts that severely degrade standard OCR: dark film carrier borders, low contrast, broken printer type, baseline skew, and dense micro-print.

The preprocessing pipeline automatically senses image properties and applies targeted enhancements without requiring manual parameter tuning:

| Enhancement | Module Function | Archival Purpose |
| :--- | :--- | :--- |
| **Black Border Cropping** | `find_black_borders()` | Eliminates dark microfilm borders that cause false positive text detections while preserving `(dx, dy)` offsets. |
| **Microfilm Detection** | `detect_microfilm_degradation()` | Autonomous trigger analyzing Laplacian variance ($< 75.0$), histogram percentiles, and contrast spread. |
| **Adaptive Background Flattening** | `detect_microfilm_degradation()` / CLAHE | Automatically detects high background variance or dark corner vignetting and normalizes luminance, making faint strokes (*of*, *said*, *fishery*) crisp and sharp. |
| **Dynamic CLAHE** | `cv2.createCLAHE()` | Local adaptive histogram equalization to recover faint text in underexposed or yellowed film regions. |
| **Micro-Stroke Reconnection** | `reconnect_broken_strokes()` | $2 \times 2$ morphological closing bridges hairline breaks (e.g. `'e'` splitting into `'c'`, `'m'` into `'rn'`). |
| **Dynamic Auto-Deskew** | `estimate_skew_angle()` | Detects page tilt ($0.35^\circ$–$5.0^\circ$) via projection profiles to prevent detectors from jumping column gutters. |
| **Table Ruling Line Suppression** | `suppress_ruling_lines_if_present()` | Detects ledger gridlines, inpaints intersecting lines, and tightens box dilation (`table_unclip_ratio = 1.25`). |
| **Adaptive 1.5× Upscaling** | `estimate_median_character_height()` | Upscales small print ($< 18\text{px}$) with bicubic interpolation, then scales coordinates back $1 / 1.5$ for exact alignment. |

### Technical Deep Dive: Microfilm Background Flattening & Dynamic CLAHE
On challenging historical microfilm reels (such as Library and Archives Canada / LAC `oocihm.lac_reel_c6063`), images suffer from non-uniform dark corner vignetting, low-contrast ink, and negative film grain.
- **The Process**: When average background variance or darkness exceeds historical archival thresholds, adaptive background flattening and local Contrast Limited Adaptive Histogram Equalization (CLAHE) are automatically engaged.
- **The Impact**: Overcomes traditional commercial OCR engines on degraded film, transforming faint, unreadable characters into crisp, distinct strokes for PP-OCRv6 without clipping character baselines.

### Technical Deep Dive: Micro-Print Upscaling & Coordinate Inversion
Micro-print on historical broadsheets (classified advertisements, shipping manifests, legal deeds) often measures only 10–14 pixels in character height. When processed at native resolution, DBNet text detection struggles with letter boundaries.
- **The Process**: Connected-component analysis estimates median character height. If $< 18\text{px}$, the image is upscaled by $1.5\times$ using bicubic interpolation before inference.
- **Coordinate Inversion (`services/ocr_pp.py` & `services/tile_ocr.py`)**: All predicted bounding box coordinates are subsequently scaled by $1 / 1.5$, ensuring all coordinate layers in ALTO XML and Searchable PDFs remain identical to the original raw page dimensions.

---

## Stage 2: Core Recognition, Tiling & Script Dispatch

The engine supports ultra-fast local recognition alongside multimodal Vision-Language Models:

| Engine | Execution Mode | Best For | Typical Latency | Outputs Supported |
| :--- | :--- | :--- | :--- | :--- |
| **PP-OCRv6** *(Default)* | Local GPU / CPU (FastDeploy / TensorRT) | Broad volume, monographs, historical broadsheet newspapers | **0.8 – 2.5s / page** | Searchable PDF, ALTO XML 4.2, JSON, TXT, HTML Dashboard |
| **Indra OCR (Fast CPU)** | Multi-core CPU ONNX Runtime | Standard laptops, CPU servers, rapid high-throughput processing without GPU | **0.5 – 1.5s / page** | Searchable PDF, ALTO XML 4.2, JSON, TXT, HTML Dashboard |
| **Dots OCR 7B** | Local OpenAI-compatible VLM Server | Post-OCR arbitration, severely degraded scans, damaged typography | **150ms / crop** / **4–8s / page** | Searchable PDF, ALTO XML 4.2, JSON, TXT, HTML Dashboard |
| **PaddleOCR-VL 1.6** | Docker VLM Service (PaddleX) | Multimodal page parsing with interleaved tables & display ads | **3.0 – 5.0s / page** | Searchable PDF, ALTO XML 4.2, JSON, TXT, HTML Dashboard |
| **olmOCR 7B** | Local vLLM Service | Complex multi-modal academic layouts & natural reading-order synthesis | **4.0 – 8.0s / page** | Searchable PDF, ALTO XML 4.2, JSON, TXT, HTML Dashboard |

### Autonomous Script & Language Resolution (`services/language_detection.py`)
CRKN collections span multilingual archives: 19th-century English newspapers, French Canadian records, and early 20th-century Chinese broadsheets.
- **Traditional Chinese (`chinese_cht`)**: Automatically detected for Chinese broadsheets (e.g., `sfu.00001` *The Chinese Times / 大漢公報*, `dahan`, `cht`). Recognizes over 6,000 Traditional Chinese characters with near 100% lexical precision.
- **Bilingual French Superset (`fr`)**: Automatically detected for Canadian federal publications, provincial gazettes, and historical Québec serials (`sru.00007`, `sru.00004`, `oocihm.8_06619`, `oocihm.8_`, `qmbn` BAnQ serials, RCMP forms, Hansard). When `lang="fr"` is dispatched, Paddle's Latin/French model natively processes accented characters (`é`, `è`, `ê`, `à`, `ç`, `ô`, `î`, `ù`, `ë`, `ï`, `œ`, `æ`) as first-class tokens rather than treating them as anomalous English noise, preventing dropouts (*qualité* $\rightarrow$ *qualit*, *société* $\rightarrow$ *societ*).
- **Standard English (`en`)**: Default for unflagged historical serials and monographs.

### CJK Directional Character Tokenization & Box Partitioning (`services/word_box_normalization.py`)
Historical Chinese Canadian newspapers (such as `sfu.00001` *The Chinese Times / 大漢公報*) present a complex mix of vertical columnar body text and horizontal display banners:
- **The Challenge**: Unlike Western scripts with spaces, Traditional Chinese characters are non-delimited. PaddleOCR returns full multicharacter line boxes (`rec_texts`). In naive export, entire column strips are lumped into single string entries, breaking word-level search, indexing, and alignment.
- **Orientation Sensing**: The normalizer evaluates the bounding box aspect ratio:
  - **Vertical Text Columns ($H > 1.2 \times W$)**: The bounding box is sliced vertically along the Y-axis into $N$ equal character slots with interpolated sub-pixel coordinates.
  - **Horizontal Banner Headers ($W > 1.2 \times H$)**: The bounding box is sliced horizontally along the X-axis into $N$ character slots.
- **ALTO & Searchable PDF Integration**: Emits discrete, individual `<String>` tokens in ALTO XML 4.2 with accurate `HPOS`, `VPOS`, `WIDTH`, and `HEIGHT`, and embeds per-glyph invisible text boxes in the Searchable PDF layer.

### Auto-Detect Tiling Strategy (`services/tile_ocr.py`)
Historical print in the CRKN archive spans from massive broadsheets ($6000 \times 9000\text{px}$) to standard microfilm monographs ($1800 \times 2400\text{px}$). Empirical testing proved that a single static tiling policy is suboptimal:

1. **Large Broadsheet Newspapers ($\ge 18\text{ MP}$ or $\max \ge 4500\text{px} \land \min \ge 3200\text{px}$)**:
   - On scans such as *The Ottawa Citizen* (`ome.00003`, 31–37 MP), classified advertisements have character heights of only 10–14px.
   - Processing untiled forces DBNet to downsample the image, missing thousands of micro-print words.
   - Slicing into a **$3 \times 2$ Grid** with a 15% overlap margin and Non-Maximum Suppression (NMS) merging yields **+328 words (+10.0% word recall over untiled OCR and +396 words over ABBYY FineReader)** on a single page.
2. **Standard Microfilms & Monographs ($< 18\text{ MP}$)**:
   - Books, annual reports, and Heritage microfilm reels (`oocihm.72118`, `lac_reel_c6063`, 3.8–7.8 MP) fit natively inside DBNet's receptive field (up to 9984 side len).
   - Running full-page inference without tiling avoids horizontal/vertical slice seams, eliminates duplicate edge words, and runs **$4\times$ faster**.
   - $3 \times 1$ slicing was empirically retired as obsolete, as horizontal cuts severed vertical Chinese columns and split multi-column paragraphs.

#### Supported Tiling Modes (`tile_strategy`)
- **`auto` *(Default)***: Automatically inspects image resolution. Broadsheets ($\ge 18\text{ MP}$) are routed to $3 \times 2$ quadrant grid tiling; books and standard pages ($< 18\text{ MP}$) run at native full-image resolution.
- **`3x2` / `grid_3x2`**: Forces $3 \times 2$ quadrant grid tiling across all documents.
- **`none`**: Forces native full-resolution OCR across all documents with zero tiling.

> Available as the **Image Tiling Strategy** dropdown in the Gradio Web UI and via `--tile-strategy auto|3x2|none` in the batch runner CLI.

---

## Stage 3: Spatial & Column-Aware Reading Order Sorter

### Why Naive Sorting Fails
Standard OCR engines sort lines purely by vertical baseline coordinate ($Y$). On multi-column newspapers, this interleaves text across columns:

```text
[Column 1 Line 1]  --->  [Column 2 Line 1]  (Naive Y-sort jumps across gutter!)
[Column 1 Line 2]  --->  [Column 2 Line 2]
```

### The Solution: Whitespace Gutter Profiling
`sort_lines_reading_order()` in `services/alto_generation.py` and `sort_items_reading_order()` in `services/pdf_generation.py` execute layout-aware column partitioning:

1. **Full-Width Banner Isolation**: Lines spanning $\ge 75\%$ of active text width (headlines, mastheads) are isolated and sorted top-to-bottom first.
2. **Gutter Profiling**: Remaining bounding boxes are projected onto the horizontal axis. A 1D density histogram is convolved with a 31-pixel Gaussian smoothing window; deep valleys identify column gutters.
3. **Partitioned Sorting**: Text lines are grouped into discrete column boundaries. Lines within each column are sorted strictly top-to-bottom, and columns are assembled in natural left-to-right reading order.

---

## Stage 4: Linguistic Arbitration & VLM Delegation ("Post Process")

When **`Post Process`** is checked, high-speed dictionary heuristics cooperate with targeted Vision-Language Model arbitration across multiple specialized passes:

### 1. Dual-Tier Dots OCR 7B VLM Delegation (`services/post_ocr_vl.py`)
- **Line-Level Delegation**: When an entire line suffers from severe baseline degradation (mean confidence $< 0.78$), the full line is cropped and dispatched to **Dots OCR 7B**. The VLM leverages sentence-level contextual syntax to resolve smudged microfilm lines.
- **Targeted Word-Level Crop Delegation**: For individual disputed tokens where confidence drops below **$< 0.80$** and lexical validation fails (or on disputed French vocabulary), the pipeline extracts a tight, context-padded word crop. Dots OCR's vision-language transformer easily deciphers broken microfilm characters, degraded ink, and ornate display typography that character engines fail on.

### 2. Accented Word Token Reconstruction (`services/word_box_normalization.py`)
In degraded French print, floating accents frequently get segmented as detached tokens or split words in two (`talage` + `é`, `galvanis` + `é`).
- **The Rule**: If an isolated diacritic letter (`é`, `à`, `è`, `ê`, `ç`, etc.) is adjacent to a word token, or if a word was split around an accent, the pipeline tests whether the combined token matches the raw line text in `rec_texts`.
- **The Impact**: Automatically consolidates these tokens into full, valid accented words (`étalage`, `galvanisé`) before ALTO XML generation, turning thousands of potential OCR dropouts into direct matches.

### 3. Punctuation-Bound & Whitespace Token Splitting (`services/word_box_normalization.py`)
- **Punctuation-Bound Splitting**: Detects words fused across punctuation marks (`r"([A-Za-zÀ-ÿ]+)[,;:\.]([A-Za-zÀ-ÿ]+)"`, e.g. *Montréal,Avril*, *situé,la*, *quincailliers.Des*). Automatically splits them into discrete tokens (*Montréal,* and *Avril*) and calculates proportional horizontal bounding boxes based on relative character count.
- **Whitespace Token Auto-Splitting**: Standardizes word tokenization by ensuring no `<String>` ALTO element contains internal spaces. Whenever a recognized token contains whitespace (e.g. `"anything to"`), it is automatically split into discrete word tokens (`"anything"` and `"to"`) with partitioned bounding boxes.

### 4. Display-Font Compound Word De-compounding (`services/word_box_normalization.py`)
In 19th-century newspaper advertisements and classified notices, compressed display type often fuses words together without spaces (*ESTPERMANENT*, *Commodepour*, *belétalageau*).
- **The Process**: If an all-caps or title-cased token fails lexical validation and has length $\ge 8$, the de-compounder tests split points against the Canadian historical lexicon.
- **The Resolution**: If both halves validate as historical words, the token is split with horizontally interpolated bounding boxes, eliminating commercial OCR's advantage on dense display ads.

### 5. Cross-Line Hyphen Reconciliation (`services/word_box_normalization.py` & `services/alto_generation.py`)
In historical newspapers and legal gazettes, trailing hyphens split long words across line breaks (*occupa-* on line 1, *tion* on line 2).
- **The Resolution**: Reconciles soft hyphens across sequential lines inside layout processing so the unified root word (*occupation*) is evaluated and preserved as a full lexical entity.

### 6. Lexicon-Constrained Decoding & Optical Confusion Matrix (`services/historical_lexicon.py`)
- **Optical Pair Confusion Matrix**: Targets classic historical print misreadings caused by broken lead type and microfilm spread:
  - $rn \leftrightarrow m$ (e.g. *modem* $\leftrightarrow$ *modern*)
  - $cl \leftrightarrow d$
  - $vv \leftrightarrow w$
  - $fi$ / $fl$ ligature fractures
  - $e \leftrightarrow c \leftrightarrow o$
  - $li \leftrightarrow h$ / $b$
- **1-Edit Levenshtein Regularizer**: When a token's confidence is $< 0.85$ and it is within 1 edit substitution of a known Canadian historical term (*heeb* $\rightarrow$ *hereby*, *resering* $\rightarrow$ *reserving*), it is regularized automatically.
- **Common Historical Optical Omissions**: Resolves frequent historical abbreviations and faint ink dropouts (*sd* $\rightarrow$ *said*, *thie* $\rightarrow$ *this*, *tis* $\rightarrow$ *this*, *oupa* $\rightarrow$ *occupa-*).
- **French Terminal Accent Regularizer**: Corrects high-frequency terminal accent dropouts (*qualit* $\rightarrow$ *qualité*, *societ* $\rightarrow$ *société*, *activit* $\rightarrow$ *activité*, *comptabilit* $\rightarrow$ *comptabilité*, *fraichement* $\rightarrow$ *fraîchement*, *deyenir* $\rightarrow$ *devenir*).

### 7. Canadiana Historical Gazetteer & Whitelist (`services/historical_lexicon.py`)
Prevents automated spellcheckers from misidentifying genuine historical proper nouns as OCR errors:
- **Territories & Settlements**: *Upper Canada*, *Lower Canada*, *Assiniboia*, *Athabaska*, *Keewatin*, *Rupert's Land*, *Hochelaga*, *Bytown*, *Stadacona*, *Cataraqui*, *Glengarry*, *Gaspésie*, *Kamouraska*, *Batoche*.
- **French Canadian Seigniorial Terms**: *seigneurie*, *censive*, *censitaire*, *lods*, *corvée*, *arpents*, *bailliage*, *prévôté*, *greffier*, *marguillier*, *voltigeurs*, *fencibles*.
- **Colonial Abbreviations**: *Vic.* (Victorian statutes), *Geo.*, *Edw.*, *Esq.*, *Honble.*, *M.P.P.*, *M.L.A.*, *Q.C.*, *inst.*, *ult.*, *prox.*, *sess.*
- **Canadian Historical Surnames & Commercial Entities (`CANADIAN_HISTORICAL_NAMES_AND_SURNAMES`)**: Historical romanized Chinese Canadian surnames (*Cho*, *Chow*, *Chan*, *Wong*, *Lee*, *Lim*, *Yip*, *Ho*, *Wu*, *Cheung*, *Yuen*, *Kwan*) and archival business entities (*Co*, *Ltd*, *Inc*). Eliminates false-win attributions to ABBYY on historical commercial advertisements (e.g. validly recognizing `"CHO CO."` where ABBYY incorrectly substituted `"So"`).

### 8. Word-Box & Historical Ligature Consolidation (`services/word_box_normalization.py`)
Historical typography contains contractions and ligatures that standard OCR engines fracture into separate bounding boxes:
- **Historical Contractions**: Automatically merges fragmented `["&", "c", "."]` into single valid token `"&c."`.
- **Standard Abbreviations**: Fuses `["No", "."]` $\rightarrow$ `"No."`, `["Co", "."]` $\rightarrow$ `"Co."`, `["Reg", "."]` $\rightarrow$ `"Reg."`, `["Esq", "."]` $\rightarrow$ `"Esq."`.
- **Orphan Punctuation Attachment**: Trailing punctuation (commas, periods, colons, semicolons) mistakenly segmented as independent boxes is merged back into preceding word boxes, eliminating spurious OCR word count penalties.

### 9. Archaic Long-s (`ſ` $\rightarrow$ `s`) Resolution
In 18th- and early 19th-century Canadian gazettes, the historical medial long-s (`ſ`) is transcribed by modern OCR models as `'f'` (e.g., *firft* $\rightarrow$ *first*, *houfe* $\rightarrow$ *house*, *caufe* $\rightarrow$ *cause*, *congrefs* $\rightarrow$ *congress*).
- Modern English collisions (*fame*, *cafe*, *fend*) are protected. Phrase-level bigrams (*"the fame"* $\rightarrow$ *"the same"*, *"in cafe"* $\rightarrow$ *"in case"*) and unambiguous words (*"bufinefs"* $\rightarrow$ *"business"*, *"majefty"* $\rightarrow$ *"majesty"*) are resolved automatically.

### 10. Official Canadiana 54-Language Census & Golden Spellcheck Policy (`services/canadiana_languages.py`)
The CRKN collections encompass 54 distinct languages, from predominant national collections down to single-document historical records:
- **Census Volume**: English (287,983 docs), French (87,877), Undetermined (46,263), Traditional Chinese (23,006), German (5,540), Yiddish (5,516), Ukrainian (3,245), Italian (1,747), Finnish (606), Latin (546), Icelandic (459), Hungarian (383), Indigenous languages (Cree, Chinook Jargon, Mi'kmaq, Ojibwa, Mohawk, Inuktitut, Tsimshian, Siksika, Chipewyan, Delaware, Haida, Slave), Dutch (108), Spanish (76), Russian (28), Gaelic (23), and more.
- **The Golden Spellcheck Policy**: Full dictionary regularization runs **strictly** for verified languages (`en`, `fr`, `it`, `de`, `es`, `nl`, `pt`, `ru`, bilingual `en`/`fr`).
- **Zero-Corruption Guardrail for 46 Canadiana Languages**: For Indigenous languages, Undetermined records, Ukrainian, Yiddish, Latin, and all unverified languages, automated spellchecking is **completely bypassed**. Raw OCR tokens are preserved intact without risk of being corrupted into foreign English or French words (e.g. Italian `località` is never mutated to English `locality`).

---

## Stage 5: Output Generation & 1:1 Coordinate Fidelity

For every processed image, the pipeline generates four synchronized output files:

```text
output/
├── document_name.json                    Normalized OCR geometry, words & confidences
├── document_name.txt                     Clean text with preserved reading-order line breaks
├── document_name.pdf                     Searchable PDF with invisible text layer
├── document_name.xml                     ALTO XML 4.2 conforming to Library of Congress schema
├── document_name.post_ocr_vl.csv         Audit log of all VLM corrections applied
└── _post_ocr_vl_tiles/                   Image crops of inspected words/lines (optional audit)
```

### Searchable PDF Generation (`services/pdf_generation.py`)
- Injects an invisible searchable text layer behind the original high-resolution scan.
- Uses built-in `Helvetica` glyph rendering to guarantee universal PDF viewer compatibility across Windows, macOS, and Linux without font embedding errors.
- **Strict 1:1 Pixel Alignment**: Word coordinates are translated with sub-pixel precision so mouse selection highlights the exact printed words on the page image.

### ALTO XML 4.2 Generation (`services/alto_generation.py`)
- Generates fully schema-valid ALTO XML 4.2 (`<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">`).
- Encapsulates exact line boxes (`<TextLine>`), discrete word tokens (`<String CONTENT="..." HPOS="..." VPOS="..." WIDTH="..." HEIGHT="..." WC="..."/>`), and spaces (`<SP/>`).
- **Whitespace Token Auto-Splitting**: Recognized tokens containing internal whitespace are partitioned into individual `<String>` items with proportional horizontal bounding boxes, eliminating multi-word string anomalies.
- Organizes text into column-ordered layout blocks (`<TextBlock>`), ensuring screen readers and automated ingest tools read broadsheets in natural reading sequence.

---

## Interactive HTML Results Dashboard (Standalone Visual Verification)

Every regular OCR run (via the Web UI or batch pipeline with `--results-dashboard`) automatically generates a self-contained, interactive HTML results dashboard in the selected output directory under `HTML_Results_Dashboard/index.html`. It automatically launches in your default browser once processing finishes.

Unlike benchmark comparisons that require a baseline ABBYY XML scan, the **Results Dashboard** works out of the box for any arbitrary batch of documents or single images across all supported engines (**PP-OCRv6**, **Indra OCR**, etc.).

```text
<Output_Folder>/
├── <Item_Name>/                     <-- Searchable PDF, ALTO XML 4.2, JSON, TXT
│   ├── <Item_Name>.1.pdf
│   ├── <Item_Name>.1.xml
│   ├── <Item_Name>.1.json
│   └── <Item_Name>.1.txt
└── HTML_Results_Dashboard/          <-- Standalone visual verification suite
    ├── index.html                   <-- Master overview dashboard
    ├── pages/<Item_Name>/           <-- High-precision bounding box viewers
    └── images/<Item_Name>/          <-- Portable high-resolution scan assets
```

### Master Overview Dashboard (`index.html`)
- **Key Performance Indicators (KPIs)**: Total pages processed, total recognized word count, total execution time, and mean confidence score.
- **Document Breakdown**: Folder-level aggregation reporting word counts, average processing speed, and confidence per item.
- **Page-Level Interactive Table**: Searchable, filterable, and sortable page index with status badges and links to open any page in the visual viewer.

### Single-Page Visual Bounding Box Viewer
- **Scanned Image + ALTO Bounding Boxes**: Overlays exact word bounding boxes onto the original high-resolution scan with 1:1 sub-pixel coordinate fidelity.
- **Three-Tier Confidence Color Coding**:
  - 🟢 **High Confidence ($\ge 90\%$)**: Word recognized with high certainty.
  - 🟠 **Medium Confidence ($75\% - 89\%$)**: Acceptable confidence, minor noise.
  - 🔴 **Low Confidence / Needs Review ($< 75\%$)**: Highlighted in red for immediate human review.
- **Page Metadata Badges**: Displays exact word count on the page, engine processing latency (in seconds), and average recognition confidence.
- **Sequential Navigation**: Jump seamlessly between pages using `< Previous` and `Next >` navigation buttons or keyboard **Left / Right arrow keys**.
- **Interactive Inspection Controls**:
  - Smooth pan-and-zoom with mouse wheel and drag (with a one-click **Fit** button).
  - Checkbox filters to isolate High, Medium, or Low confidence word boxes.
  - Background image opacity slider to verify bounding box alignment against the scan.
  - Interactive sidebar word list with live search and click-to-highlight functionality.

---

## Benchmarking & ABBYY FineReader Comparison

The pipeline includes an automated benchmarking suite to evaluate performance against ABBYY FineReader Server output and Ground Truth transcripts:

```mermaid
flowchart LR
    PaddleALTO["Paddle Pipeline Output ALTO"] --> Arbiter["tools/compare_alto.py<br/>Lexical Quality Arbiter"]
    AbbyyALTO["ABBYY FineReader ALTO"] --> Arbiter
    Arbiter --> Lexicon["Canadian Historical Lexicon<br/>+ English/French SpellCheckers<br/>+ Phrase-Aware QA"]
    Lexicon --> Output["Interactive HTML Dashboard<br/>(compare/index.html)<br/>+ Concordance, Word Recall & Win Rates"]
    Output --> Sync["Granular Publication Push<br/>(tools/sync_evaluations.py)<br/>to OCR-Evaluations Repository"]
```

### Key Evaluation Capabilities
- **Phrase-Aware Lexical QA**: If a comparison token contains spaces, the arbiter evaluates whether all constituent words are valid dictionary entries (`all(is_valid(w) for w in token.split())`), preventing false error penalties on compound or hyphenated phrases.
- **Word Recall & Total Word Count Reporting**: Accurately tracks recognized word count and word recall rates alongside dictionary precision, providing a balanced, transparent comparison against ABBYY FineReader.
- **Interactive Word & Discrepancy Adjudication**: In each page comparison viewer (`*_comparison.html`), reviewers can click any word bounding box or discrepancy sidebar card to select/override the winner (Paddle Win, ABBYY Win, Both Valid, Neither Valid). Concordance rates, win totals, and progress meters update in real-time with browser storage persistence.
- **Resilient Multi-Repository Synchronization (`--sync-docs` & `tools/sync_evaluations.py`)**:
  - Evaluation results and interactive inspection viewers can be synchronized directly to the CRKN `OCR-Evaluations` repository.
  - To prevent GitHub HTTPS timeouts on multi-gigabyte historical scan datasets (~7 GiB), synchronization stages and pushes in granular publication-by-publication batches with exponential backoff and retry.
  - Local `compare/` directories are untracked in `.gitignore` to keep the main pipeline repository clean and lightweight.

### Running the Full Automated Benchmark
To run OCR on the test directory and update the comparison dashboard:

```powershell
# Default run (targets Newspapers collection: 68 broadsheet & tabloid pages):
.\venv\Scripts\python.exe tools/run_full_benchmark.py

# Or target the entire test dataset (all collections including books, serials, and reels):
.\venv\Scripts\python.exe tools/run_full_benchmark.py "C:\Users\blapierre\Documents\PDF Test"
```

The script executes 3 automated phases:
1. **Phase 1: Full OCR Run**: Runs PP-OCRv6 with `tile_strategy="auto"`, preprocessing (border crop, CLAHE, micro-print upscaling), and post-OCR VLM arbitration.
2. **Phase 2: ABBYY Benchmarking**: Generates token concordance diffs, dictionary win rates, and interactive HTML dashboards under `compare/`.
3. **Phase 3: Resilient GitHub Sync**: Pushes updated dashboards and metrics to the `OCR-Evaluations` repository.

### Running the Comparison Dashboard Standalone
To compare an existing folder of ALTO XML files against ABBYY FineReader baseline files:

```powershell
.\venv\Scripts\python.exe tools/compare_alto.py `
  --pipeline "C:\Users\blapierre\Desktop\PDF Test - Final\lac_reel_c6063" `
  --benchmark "C:\Users\blapierre\Documents\PDF Test\lac_reel_c6063" `
  --output-dir "compare" `
  --title "Heritage Reel c6063 Benchmark"
```

The tool produces:
- `compare/index.html`: An interactive HTML dashboard with side-by-side diffs, token concordance rates, and win statistics.
- `compare/comparison_summary.csv`: Machine-readable token statistics, word count recall, and dictionary validity metrics per page.

#### Executive Benchmark Overview
![OCR Quality Benchmark Dashboard](images/02_comparison_dashboard.png)

#### Interactive Side-by-Side Word Concordance Diff
![Interactive Side-by-Side Word Concordance Diff](images/03_side_by_side_diff.png)

---

## Headless Batch Runner CLI

For unattended processing of full microfilm reels (hundreds or thousands of pages) across multiple volumes:

```powershell
.\venv\Scripts\python.exe tools/batch_runner.py `
  --input "C:\CRKN Projects\Heritage\oocihm.lac_reel_c10143" `
  --output "C:\CRKN Projects\output\lac_reel_c10143" `
  --method PP-OCRv6 `
  --tile-strategy auto `
  --pre-process `
  --post-process `
  --format pdf_alto `
  --resume `
  --concurrency 8
```

### Key CLI Flags
- `--tile-strategy`: Tiling policy: `auto` (default, evaluates resolution $\ge 18\text{ MP}$), `3x2` (forces 3x2 quadrant tiling), or `none` (native full resolution).
- `--resume` / `-r`: Resumes an interrupted run, skipping pages that already have complete output in the target destination folder.
- `--pre-process`: Enables automatic microfilm detection, black border crop, CLAHE contrast boost, stroke reconnection, and auto-deskew.
- `--post-process`: Enables post-OCR VLM arbitration, historical lexicon whitelisting, and orthography cleanup.
---

## Setup & Installation

### Windows Native Setup

1. **Clone the repository**:
   ```powershell
   git clone https://github.com/crkn-rcdr/paddle_pipeline.git
   cd paddle_pipeline
   ```

2. **Create and activate a virtual environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Launch the Application**:
   ```powershell
   # Gradio Web UI
   .\venv\Scripts\python.exe app.py

   # Or Desktop Launcher
   .\venv\Scripts\python.exe launcher.py
   ```

### Docker VLM Microservices (Optional)

If running visual language model verification via local containerized backends:

```bash
docker compose up -d
```
Starts:
- **PaddleOCR-VL 1.6 API** on port `8080`.
- **olmOCR vLLM service** on port `8000`.

---

## Repository Layout

```text
paddle_pipeline/
├── app.py                         Gradio UI application entry point
├── launcher.py                    Desktop launcher wrapper
├── smoke_test.py                  Headless sanity check script
├── compose.yaml                   Docker Compose configuration for VLM containers
├── requirements.txt               Python dependencies for Windows / GPU
├── services/
│   ├── image_preprocessing.py     Border crop, CLAHE, stroke reconnection, deskew, table suppression, micro-print upscaling
│   ├── language_detection.py      Multilingual script detection (Traditional Chinese, bilingual French, English)
│   ├── ocr_pp.py                  PP-OCRv6 FastDeploy engine execution & coordinate mapping
│   ├── ocr_indra.py               Indra OCR multi-process ONNX CPU engine execution
│   ├── ocr_dots.py                Dots OCR 7B standalone vision-language engine
│   ├── ocr_vl.py                  PaddleVL (PaddleX) multimodal layout parsing client
│   ├── ocr_olm.py                 olmOCR 7B vision-to-markdown transcription client
│   ├── results_dashboard.py       Standalone interactive HTML results dashboard generator
│   ├── post_ocr_vl.py             Dots OCR VLM arbitration & whole-line delegation
│   ├── historical_lexicon.py      Canadian gazetteer, long-s & French diacritics
│   ├── canadiana_languages.py     Canadiana 54-language census & zero-corruption spellcheck policy
│   ├── word_box_normalization.py  Word bounding box geometry normalizer & ligature consolidator
│   ├── punctuation_cleanup.py     Post-processing text & punctuation cleaning
│   ├── alto_generation.py         ALTO XML 4.2 export & multi-column sorting
│   ├── pdf_generation.py          Searchable PDF export & sub-pixel coordinate alignment
│   ├── tile_ocr.py                Large broadsheet tiling & coordinate scaling
│   ├── txt_generation.py          Plain text exporter
│   ├── ocr_defaults.py            Configuration constants & parameter defaults
│   └── fonts/                     Helvetica metric definitions for PDF text layer
├── tools/
│   ├── run_full_benchmark.py      Automated end-to-end full run & comparison dashboard updater
│   ├── fast_paddle_pipeline.py    Staged high-throughput GPU batch processing pipeline
│   ├── batch_runner.py            Headless CLI batch processor with resume support
│   ├── compare_alto.py            ABBYY FineReader comparison & HTML dashboard generator
│   ├── evaluate_ocr.py            CER, WER, and Bag-of-Words F1 benchmark against Ground Truth
│   └── sync_evaluations.py        Resilient publication-level batch pusher for OCR-Evaluations repository
├── windows_bundle/                Non-technical end-user package (README-For-Users.md, launch scripts, images)
└── compare/                       Local evaluation output directory (untracked in git, synced to OCR-Evaluations)
```

---

## License & Attribution

Developed for the **Canadian Research Knowledge Network (CRKN / RCDR)**. Released under the [MIT License](LICENSE).  
Contributed algorithms utilize [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [Dots OCR](https://github.com/redcaps-ai), and [ALTO XML](https://www.loc.gov/standards/alto/).
