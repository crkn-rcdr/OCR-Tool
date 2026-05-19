#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-cpu}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d ".venv" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

case "$MODE" in
  cpu)
    python -m pip install paddlepaddle==3.2.1 \
      -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
    ;;
  gpu)
    python -m pip install paddlepaddle-gpu==3.2.1 \
      -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
    ;;
  *)
    echo "Usage: $0 [cpu|gpu]" >&2
    exit 1
    ;;
esac

python -m pip install -r requirements-wsl.txt

# PaddleX pulls the GUI OpenCV wheel, which needs libGL in WSL.
# Keep the headless build instead so cv2 imports cleanly without extra system packages.
python -m pip uninstall -y opencv-contrib-python || true
python -m pip install --force-reinstall opencv-contrib-python-headless==4.10.0.84

python - <<'PY'
import cv2
import fitz
import gradio
import paddleocr

print("cv2", cv2.__version__)
print("fitz", fitz.__doc__.splitlines()[0])
print("gradio", gradio.__version__)
print("paddleocr", paddleocr.__version__)
PY
