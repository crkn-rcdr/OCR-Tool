from pathlib import Path
import json
def generate_txt_file(json_path:Path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if "overall_ocr_res" in data:
        ocr_res = data["overall_ocr_res"]
        texts = ocr_res["rec_texts"]
    elif "rec_texts" in data:
        texts = data["rec_texts"]

    elif "result" in data:
        parsing_res_list = data["result"]["layoutParsingResults"][0]["prunedResult"]["parsing_res_list"]
        texts = []

        for block in parsing_res_list:
            text = block.get("block_content")
            texts.append(text)

    txt_path = json_path.with_suffix(".txt")

    with open(txt_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(f"{text}\n")
    
    return txt_path