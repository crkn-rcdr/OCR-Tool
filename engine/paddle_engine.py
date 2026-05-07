from paddleocr import PPStructureV3
import threading
from paddleocr import PaddleOCR  

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
       
        return (
            opt.get("method"),
            opt.get("use_doc_orientation_classify", False),
            opt.get("use_doc_unwarping", False),
            opt.get("use_textline_orientation", False),
            opt.get("lang"),
            
            
        )

    def single_instance_engine(self,opt:dict):
        """
        Initialize or reuse a PaddleOCR instance (singleton behavior).

        If the OCR instance already exists and the parameters
        have not changed, it will be reused.

        Args:
            opt (dict): OCR configuration options.
        """
        params = self._create_params(opt)
        method = params[0]

        with self._lock:
            if self.pipeline is not None and params == self._init_params:
                return False
        
        kwargs = {
            "use_doc_orientation_classify": params[1],
            "use_doc_unwarping": params[2],
            "use_textline_orientation": params[3],
           
            #"use_chart_recognition": False,
        }

        if params[4]:
            kwargs["lang"] = params[4]
        print("[DEBUG] init kwargs:", kwargs, flush=True)
        
        if method == "PP-OCRv5":
            kwargs["return_word_box"] = True
            self.pipeline = PaddleOCR(**kwargs)
        elif method in ["PP-StructureV3","Auto Routing"]:
            
            self.pipeline = PPStructureV3(**kwargs)

        self._init_params = params
        return True
        
    def infer(self,image_path:str,opt:dict):
        self.single_instance_engine(opt)
        return self.pipeline.predict(image_path)