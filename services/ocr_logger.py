import logging 
from datetime import datetime
from pathlib import Path

class OCRLoggerManger:
    """
    Singleton-style logger manager shared across ocr_pp.py, ocr_vl.py, and auto_ocr.py.
    """
    _initialized = False
    _summary_logger = None
    _error_logger = None
    _summary_log_path = None
    _error_log_path = None
    _run_timestamp = None

    @classmethod
    def setup(cls,methond:str,log_dir:str="logs"):
        if cls._initialized:
            return(
                cls._summary_logger,
                cls._error_logger,
                cls._summary_log_path,  
                cls._error_log_path
            )

        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        cls._run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        method_sanitized = methond.replace("-", "_")

        summary_log_path = log_dir_path / f"{method_sanitized}_summary_{cls._run_timestamp}.log"
        error_log_path = log_dir_path / f"{method_sanitized}_error_{cls._run_timestamp}.log"

        formatter = logging.Formatter('%(asctime)s | %(levelname)s |%(message)s')

        summary_logger = logging.getLogger("ocr_summary")
        summary_logger.setLevel(logging.INFO)
        summary_logger.handlers.clear()
        summary_logger.propagate = False

        summary_handler = logging.FileHandler(summary_log_path,encoding='utf-8')
        summary_handler.setFormatter(formatter)
        summary_logger.addHandler(summary_handler)

        error_logger = logging.getLogger("ocr_error")
        error_logger.setLevel(logging.ERROR)
        error_logger.handlers.clear()
        error_logger.propagate = False

        error_handler = logging.FileHandler(error_log_path,encoding='utf-8')
        error_handler.setFormatter(formatter)
        error_logger.addHandler(error_handler)

        cls._summary_log_path = summary_log_path
        cls._error_log_path = error_log_path
        cls._summary_logger = summary_logger
        cls._error_logger = error_logger
        cls._initialized = True

        return (
            cls._summary_logger,
            cls._error_logger,
            cls._summary_log_path,
            cls._error_log_path
        )