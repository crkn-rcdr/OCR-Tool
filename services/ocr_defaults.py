"""Shared OCR defaults for UI, CLI, logging, and engine initialization."""

import os

DEFAULT_PREPROCESSING_PRESET = "none"
DEFAULT_PREPROCESSING_STEPS = ["background_flatten"]
DEFAULT_IGNORE_BLACK_BORDER = True
PREPROCESSING_STEP_CHOICES = [
    ("Light clean", "light_clean"),
    ("Background flatten", "background_flatten"),
    ("Contrast / CLAHE", "contrast"),
    ("Deskew", "deskew"),
    ("Sauvola binary", "sauvola_binary"),
]
PREPROCESSING_STEP_LABELS = {
    value: label for label, value in PREPROCESSING_STEP_CHOICES
}
PREPROCESSING_PRESET_CHOICES = [
    ("None", "none"),
    ("Light clean", "light_clean"),
    ("Contrast / CLAHE", "contrast"),
    ("Background flatten", "background_flatten"),
    ("Sauvola binary", "sauvola_binary"),
    ("Deskew + contrast", "deskew_contrast"),
]
PREPROCESSING_PRESET_LABELS = {
    value: label for label, value in PREPROCESSING_PRESET_CHOICES
}
PREPROCESSING_PRESET_TO_STEPS = {
    "none": [],
    "light_clean": ["light_clean"],
    "contrast": ["light_clean", "contrast"],
    "background_flatten": ["light_clean", "background_flatten", "contrast"],
    "sauvola_binary": ["light_clean", "background_flatten", "sauvola_binary"],
    "deskew_contrast": ["light_clean", "contrast", "deskew"],
}

DEFAULT_TILE_OVERLAP_PX = 96
DEFAULT_DOCLAYOUT_PADDING_PX = 32
DEFAULT_DOCLAYOUT_THRESHOLD = 0.20

DEFAULT_TEXT_DET_THRESH = 0.20
DEFAULT_TEXT_DET_LIMIT_TYPE = "max"
DEFAULT_TEXT_DET_LIMIT_SIDE_LEN = 9984
DEFAULT_TEXT_DET_MAX_SIDE_LIMIT = 4000
DEFAULT_TEXT_DET_BOX_THRESH = 0.30
DEFAULT_TEXT_DET_UNCLIP_RATIO = 2.0
DEFAULT_TEXT_REC_SCORE_THRESH = 0.0
DEFAULT_TEXT_RECOGNITION_BATCH_SIZE = 32

DEFAULT_PERO_SOURCE_PATH = os.getenv(
    "PERO_OCR_SOURCE_PATH",
    "/mnt/c/Users/BrittnyLapierre/Documents/pero-ocr",
)
DEFAULT_PERO_CONFIG_PATH = os.getenv("PERO_OCR_CONFIG_PATH", "")
DEFAULT_PERO_DEVICE = os.getenv("PERO_OCR_DEVICE", "gpu")
DEFAULT_PERO_GPU_ID = os.getenv("PERO_OCR_GPU_ID", "")
DEFAULT_PERO_MIN_LINE_CONFIDENCE = -1.0
DEFAULT_CLEAN_PUNCTUATION_ISSUES = False

DEFAULT_POST_OCR_VL_ENABLED = True
DEFAULT_POST_OCR_VL_BACKEND = os.getenv("POST_OCR_VL_BACKEND", "olmocr")
DEFAULT_POST_OCR_VL_CANDIDATE_MODE = "suspicious"
DEFAULT_POST_OCR_VL_MAX_WORDS = 0
DEFAULT_POST_OCR_VL_PADDING_PX = 5
DEFAULT_POST_OCR_VL_LOW_CONFIDENCE_THRESHOLD = 0.90
DEFAULT_POST_OCR_VL_APPLY_CORRECTIONS = True
DEFAULT_POST_OCR_VL_SAVE_TILES = False
DEFAULT_POST_OCR_VL_GROUP_LINES = True
DEFAULT_PADDLE_VL_API_URL = os.getenv("PADDLE_VL_API_URL", "http://127.0.0.1:8080")
DEFAULT_OLLAMA_OPENAI_API_URL = os.getenv(
    "OLLAMA_OPENAI_API_URL",
    os.getenv("OPENAI_COMPAT_API_URL", os.getenv("OLMOCR_API_URL", "http://localhost:11434/v1")),
)
DEFAULT_OLLAMA_OPENAI_MODEL = os.getenv(
    "OLLAMA_OPENAI_MODEL",
    os.getenv("OPENAI_COMPAT_MODEL", os.getenv("OLMOCR_MODEL", "richardyoung/olmocr2:7b-q8")),
)
DEFAULT_OLMOCR_API_URL = DEFAULT_OLLAMA_OPENAI_API_URL
DEFAULT_OLMOCR_MODEL = DEFAULT_OLLAMA_OPENAI_MODEL
