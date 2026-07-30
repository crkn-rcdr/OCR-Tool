$ErrorActionPreference = "Stop"

$repoRoot = Split-Path $PSScriptRoot -Parent
$python = Join-Path $repoRoot "venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Expected Python at $python"
}

$env:OPENAI_COMPAT_API_URL = "http://127.0.0.1:8000/v1"
$env:OPENAI_COMPAT_MODEL = "mradermacher/chandra-ocr-2-GGUF:Q4_K_M"

& $python (Join-Path $repoRoot "app.py")
