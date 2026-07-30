$ErrorActionPreference = "Stop"

$port = 8000
$model = "mradermacher/chandra-ocr-2-GGUF:Q4_K_M"
$logDir = Join-Path $PSScriptRoot "..\logs"
$logPath = Join-Path $logDir "chandra_server.log"
$errPath = Join-Path $logDir "chandra_server.err.log"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath) {
    $env:PATH += ";$userPath"
}

$existing = Get-Process llama-server -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "llama-server is already running."
    exit 0
}

$llamaServer = (Get-Command llama-server -ErrorAction Stop).Source

if (Test-Path $logPath) {
    Remove-Item -LiteralPath $logPath -Force -ErrorAction SilentlyContinue
}
if (Test-Path $errPath) {
    Remove-Item -LiteralPath $errPath -Force -ErrorAction SilentlyContinue
}

Start-Process `
    -FilePath $llamaServer `
    -ArgumentList "--hf-repo", $model, "--host", "127.0.0.1", "--port", "$port", "--gpu-layers", "999" `
    -RedirectStandardOutput $logPath `
    -RedirectStandardError $errPath `
    -WindowStyle Hidden

Write-Host "Starting Chandra server on http://127.0.0.1:$port/v1"
Write-Host "Log: $logPath"
