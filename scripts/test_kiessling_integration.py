import os
import sys
import time
import pathlib
import numpy as np
import cv2

# Ensure NVIDIA CUDA DLLs are found for ONNXRuntime
venv_dir = pathlib.Path(r"C:\CRKN Projects\paddle_pipeline\venv")
cuda_dirs = [
    str(venv_dir / "Lib/site-packages/nvidia/cublas/bin"),
    str(venv_dir / "Lib/site-packages/nvidia/cufft/bin"),
    str(venv_dir / "Lib/site-packages/nvidia/cuda_runtime/bin"),
    str(venv_dir / "Lib/site-packages/nvidia/cudnn/bin"),
    str(venv_dir / "Lib/site-packages/onnxruntime/capi"),
]
os.environ["PATH"] = ";".join(cuda_dirs) + ";" + os.environ.get("PATH", "")
for d in cuda_dirs:
    if os.path.isdir(d):
        try:
            os.add_dll_directory(d)
        except Exception:
            pass

import onnxruntime as ort
from kraken.models.loaders import load_safetensors
from paddleocr import PaddleOCR

ONNX_MODEL_PATH = r"C:\CRKN Projects\paddle_pipeline\output\test_kiessling.onnx"
SAFETENSORS_PATH = r"C:\Users\blapierre\.cache\huggingface\hub\models--small-models-for-glam--kraken-ppocrv6-medium\snapshots\01b574f071bf21cff7d7e2a4f38966925dddc30a\medium.safetensors"
TEST_IMG = r"C:\Users\blapierre\Documents\PDF Test\Newspapers\ome.00003_19290103\ome.00003_19290103.1.jpg"

print("=" * 70)
print("1. Initializing Kiessling Historical Model on NVIDIA RTX 5090 CUDA...")
print("=" * 70)
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
k_model = load_safetensors(SAFETENSORS_PATH)[0]
codec = k_model.codec
in_name = session.get_inputs()[0].name
print(f"CUDA Provider Active: {session.get_providers()[0]}")
print(f"Historical Vocabulary: {len(codec.c2l)} characters")

print("\n" + "=" * 70)
print("2. Initializing Native PaddleOCR (DBNet Detection Engine)...")
print("=" * 70)
ocr = PaddleOCR(ocr_version="PP-OCRv6", return_word_box=True)

# Run full prediction with base PaddleOCR
print(f"\nProcessing newspaper page: {pathlib.Path(TEST_IMG).name}")
t0 = time.time()
base_res = list(ocr.predict(TEST_IMG, return_word_box=True))[0]
base_time = time.time() - t0
print(f"Base PaddleOCR full page time: {base_time:.3f}s")
print(f"Detected lines: {len(base_res['rec_texts'])}")

# Extract crops according to dt_polys
img = cv2.imread(TEST_IMG)
dt_polys = base_res["dt_polys"]

def get_rotate_crop_image(img, points):
    points = np.array(points, dtype=np.float32)
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3]),
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2]),
        )
    )
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img

print("\n" + "=" * 70)
print("3. Running Kiessling PP-OCRv6 CUDA Recognition on Line Crops...")
print("=" * 70)

# Preprocess crops for Kiessling model: height=96, normalized, batched
def resize_norm_crop(crop_img, target_h=96):
    h, w = crop_img.shape[:2]
    ratio = w / float(h)
    new_w = max(16, int(target_h * ratio))
    # round up to nearest multiple of 32
    new_w = int(np.ceil(new_w / 32) * 32)
    resized = cv2.resize(crop_img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
    # convert BGR to RGB and normalize float32 [-1, 1] or [0, 1]
    # Kraken uses RGB standard [0, 1] normalized with mean/std
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # NCHW
    chw = np.transpose(rgb, (2, 0, 1))
    return chw

def ctc_decode_single(b_probs, codec):
    argmax_indices = np.argmax(b_probs, axis=0) # (seq_len,)
    locs = []
    prev_idx = -1
    for t in range(len(argmax_indices)):
        idx = int(argmax_indices[t])
        if idx != 0 and idx != prev_idx:
            conf = float(b_probs[idx, t])
            locs.append((idx, t, t + 1, conf))
        prev_idx = idx
    if not locs:
        return "", 0.0
    decoded = codec.decode(locs)
    text = "".join(char for char, _, _, _ in decoded)
    avg_conf = float(np.mean([conf for _, _, _, conf in decoded])) if decoded else 0.0
    return text, avg_conf

kiessling_texts = []
t_start_k = time.time()

# Recognize crops in batches
BATCH_SIZE = 32
crops = [get_rotate_crop_image(img, poly) for poly in dt_polys]

for i in range(0, len(crops), BATCH_SIZE):
    batch_crops = crops[i:i + BATCH_SIZE]
    # pad batch to max width
    processed = [resize_norm_crop(c, target_h=96) for c in batch_crops]
    max_w = max(p.shape[2] for p in processed)
    batch_tensor = np.zeros((len(processed), 3, 96, max_w), dtype=np.float32)
    for idx, p in enumerate(processed):
        batch_tensor[idx, :, :, :p.shape[2]] = p
        
    out = session.run(None, {in_name: batch_tensor})[0] # (B, 1623, 1, seq_len)
    if out.ndim == 4:
        out = out.squeeze(2)
        
    # Softmax
    exp = np.exp(out - np.max(out, axis=1, keepdims=True))
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    
    for b in range(len(batch_crops)):
        text, conf = ctc_decode_single(probs[b], codec)
        kiessling_texts.append((text, conf))

t_end_k = time.time()
print(f"Kiessling GPU Batch Recognition Time: {(t_end_k - t_start_k):.3f}s for {len(crops)} lines")
print(f"Speed: {len(crops) / (t_end_k - t_start_k):.1f} lines/sec on RTX 5090")

print("\n" + "=" * 70)
print("4. Side-by-Side Comparison Sample (Base Paddle vs. Kiessling Historical):")
print("=" * 70)
sample_count = min(15, len(crops))
for idx in range(sample_count):
    paddle_text = base_res["rec_texts"][idx]
    if isinstance(paddle_text, tuple):
        paddle_text = paddle_text[0]
    paddle_conf = base_res["rec_scores"][idx]
    k_text, k_conf = kiessling_texts[idx]
    
    print(f"Line {idx+1:2d}:")
    print(f"  Paddle Base   [{paddle_conf:.2f}]: {paddle_text}")
    print(f"  Kiessling GLAM [{k_conf:.2f}]: {k_text}")

