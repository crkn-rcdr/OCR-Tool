"""
from paddleocr import PaddleOCR  

ocr = PaddleOCR(
   text_det_thresh=0.2,
   text_det_limit_type="min",
   text_det_limit_side_len=2000000,
)

result = ocr.predict("./oocihm.22250.5.jpg")  
for res in result:  
    res.print()  
    res.save_to_img("output")  
    res.save_to_json("output")  

"""
from paddleocr import PPStructureV3

pipeline = PPStructureV3(use_doc_unwarping=False,use_textline_orientation=False,use_doc_orientation_classify=False) # Set the text_det_thresh parameter to adjust the text detection confidence threshold. The default value is 0.5. Lowering this threshold may increase recall but may also introduce more false positives. Adjust this value based on your specific use case and document types.
# pipeline = PPStructureV3(lang="en") # Set the lang parameter to use the English text recognition model. For other supported languages, see Section 5: Appendix. By default, both Chinese and English text recognition models are enabled.
# pipeline = PPStructureV3(use_doc_orientation_classify=True) # Use use_doc_orientation_classify to enable/disable document orientation classification model
# pipeline = PPStructureV3(use_doc_unwarping=True) # Use use_doc_unwarping to enable/disable document unwarping module
# pipeline = PPStructureV3(use_textline_orientation=True) # Use use_textline_orientation to enable/disable textline orientation classification model
# pipeline = PPStructureV3(device="gpu") # Use device to specify GPU for model inference
output = pipeline.predict("./oocihm.22250.7.jpg")
for res in output:
    res.print() ## Print the structured prediction output
    res.save_to_json(save_path="output")