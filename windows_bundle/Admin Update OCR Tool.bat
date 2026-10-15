@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%scripts\Update-OcrTool.ps1"
if errorlevel 1 (
  echo.
  echo Update did not complete cleanly. Please send the message above to support.
  pause
)

