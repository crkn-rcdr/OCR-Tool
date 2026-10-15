@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%scripts\Stop-OcrTool.ps1"
if errorlevel 1 (
  echo.
  echo OCR Tool did not stop cleanly. Please send the message above to support.
  pause
)

