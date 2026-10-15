import threading
from .paddlex_compat import apply_headless_opencv_patch
from services.ocr_defaults import (
    DEFAULT_OCR_VERSION,
    DEFAULT_TEXT_DET_BOX_THRESH,
    DEFAULT_TEXT_DET_LIMIT_SIDE_LEN,
    DEFAULT_TEXT_DET_LIMIT_TYPE,
    DEFAULT_TEXT_DET_MAX_SIDE_LIMIT,
    DEFAULT_TEXT_DET_THRESH,
    DEFAULT_TEXT_DET_UNCLIP_RATIO,
    DEFAULT_TEXT_REC_SCORE_THRESH,
    DEFAULT_TEXT_RECOGNITION_BATCH_SIZE,
)

apply_headless_opencv_patch()
try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

PADDLE_OCR_METHODS = {"PP-OCRv6"}



def option_or_default(opt: dict, key: str, default):
    value = opt.get(key)
    return default if value is None else value


def normalize_text_det_limit_type(value):
    value = str(value or DEFAULT_TEXT_DET_LIMIT_TYPE).strip().lower()
    if value not in {"max", "min"}:
        return DEFAULT_TEXT_DET_LIMIT_TYPE
    return value


class PaddleEngine:
    """
    Single-instance PaddleOCR inference engine.

    - Initializes PaddleOCR based on runtime options
    - Reuses the OCR instance if initialization parameters are unchanged
    """ 

    def __init__(self):
        self.pipeline = None
        self._init_params = None
        self.method = None
        self._lock = threading.Lock()
    
    def _create_params(self,opt:dict):
        """
        Extract OCR initialization parameters from options.

        Args:
            opt (dict): OCR configuration options.

        Returns:
            tuple: A tuple of OCR initialization parameters.
        """
       
        return {
            "method": opt.get("method"),
            "ocr_version": DEFAULT_OCR_VERSION,
            "device": opt.get("device"),
            "use_doc_orientation_classify": opt.get(
                "use_doc_orientation_classify", False
            ),
            "use_doc_unwarping": opt.get(
                "use_doc_unwarping", False
            ),
            "use_textline_orientation": opt.get(
                "use_textline_orientation", False
            ),
            "lang": opt.get("lang"),
            "text_det_thresh": option_or_default(
                opt,
                "text_det_thresh",
                DEFAULT_TEXT_DET_THRESH,
            ),
            "text_det_limit_side_len": int(
                option_or_default(
                    opt,
                    "text_det_limit_side_len",
                    DEFAULT_TEXT_DET_LIMIT_SIDE_LEN,
                )
            ),
            "text_det_limit_type": normalize_text_det_limit_type(
                option_or_default(
                    opt,
                    "text_det_limit_type",
                    DEFAULT_TEXT_DET_LIMIT_TYPE,
                )
            ),
            "text_det_max_side_limit": int(
                option_or_default(
                    opt,
                    "text_det_max_side_limit",
                    DEFAULT_TEXT_DET_MAX_SIDE_LIMIT,
                )
            ),
            "text_det_box_thresh": option_or_default(
                opt,
                "text_det_box_thresh",
                DEFAULT_TEXT_DET_BOX_THRESH,
            ),
            "text_det_unclip_ratio": option_or_default(
                opt,
                "text_det_unclip_ratio",
                DEFAULT_TEXT_DET_UNCLIP_RATIO,
            ),
            "text_rec_score_thresh": option_or_default(
                opt,
                "text_rec_score_thresh",
                DEFAULT_TEXT_REC_SCORE_THRESH,
            ),
            "text_recognition_batch_size": int(
                option_or_default(
                    opt,
                    "text_recognition_batch_size",
                    DEFAULT_TEXT_RECOGNITION_BATCH_SIZE,
                )
            ),
        }

    def single_instance_engine(self,opt:dict):
        """
        Initialize or reuse a PaddleOCR instance (singleton behavior).

        If the OCR instance already exists and the parameters
        have not changed, it will be reused.

        Args:
            opt (dict): OCR configuration options.
        """
        params = self._create_params(opt)
        method = params["method"]

        with self._lock:
            if self.pipeline is not None and params == self._init_params:
                return False
        
        kwargs = {
            "use_doc_orientation_classify": params["use_doc_orientation_classify"],
            "use_doc_unwarping": params["use_doc_unwarping"],
            "use_textline_orientation": params["use_textline_orientation"],
            "text_det_thresh": params["text_det_thresh"],
            "text_det_limit_side_len": params["text_det_limit_side_len"],
            "text_det_limit_type": params["text_det_limit_type"],
            "text_det_box_thresh": params["text_det_box_thresh"],
            "text_det_unclip_ratio": params["text_det_unclip_ratio"],
            "text_rec_score_thresh": params["text_rec_score_thresh"],
            "text_recognition_batch_size": params["text_recognition_batch_size"],
        }

        if params["device"] is not None:
            kwargs["device"] = params["device"]
        if params["lang"] is not None:
            kwargs["lang"] = params["lang"]
        
        if method not in PADDLE_OCR_METHODS:
            raise ValueError(f"Unsupported OCR method: {method}")

        kwargs["ocr_version"] = params["ocr_version"]
        kwargs["return_word_box"] = True
        self.pipeline = PaddleOCR(**kwargs)

        self._init_params = params
        self.method = method
        return True
        
    def infer(self,image_path:str,opt:dict):
        self.single_instance_engine(opt)
        params = self._create_params(opt)
        predict_kwargs = {
            "use_doc_orientation_classify": params["use_doc_orientation_classify"],
            "use_doc_unwarping": params["use_doc_unwarping"],
            "use_textline_orientation": params["use_textline_orientation"],
            "text_det_limit_side_len": params["text_det_limit_side_len"],
            "text_det_limit_type": params["text_det_limit_type"],
            "text_det_thresh": params["text_det_thresh"],
            "text_det_box_thresh": params["text_det_box_thresh"],
            "text_det_unclip_ratio": params["text_det_unclip_ratio"],
            "text_rec_score_thresh": params["text_rec_score_thresh"],
        }

        rec_batch_size = params["text_recognition_batch_size"]
        if hasattr(self.pipeline, "paddlex_pipeline") and hasattr(self.pipeline.paddlex_pipeline, "_pipeline"):
            text_rec_model = getattr(self.pipeline.paddlex_pipeline._pipeline, "text_rec_model", None)
            if text_rec_model and hasattr(text_rec_model, "batch_sampler") and hasattr(text_rec_model.batch_sampler, "batch_size"):
                text_rec_model.batch_sampler.batch_size = rec_batch_size

        if self.method in PADDLE_OCR_METHODS and hasattr(self.pipeline, "paddlex_pipeline"):
            return self.pipeline.paddlex_pipeline.predict(
                image_path,
                **predict_kwargs,
                text_det_max_side_limit=params["text_det_max_side_limit"],
                return_word_box=True,
            )

        return self.pipeline.predict(image_path, **predict_kwargs)
