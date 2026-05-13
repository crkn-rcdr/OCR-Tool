from paddleocr import PaddleOCR  

ocr = PaddleOCR(
   text_det_thresh=0.2,
   text_det_limit_type="min",
   text_det_limit_side_len=2000000,
)

result = ocr.predict("./oocihm.N_00155_18750610.4.jpg")  
for res in result:  
    res.print()  
    res.save_to_img("output")  
    res.save_to_json("output")  