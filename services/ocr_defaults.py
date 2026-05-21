"""Shared OCR defaults for UI, CLI, logging, and Paddle initialization."""

DEFAULT_PREPROCESSING_PRESET = "none"
DEFAULT_PREPROCESSING_STEPS = []
DEFAULT_IGNORE_BLACK_BORDER = False
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
