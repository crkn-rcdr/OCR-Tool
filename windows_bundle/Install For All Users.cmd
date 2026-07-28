@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%scripts\Install-ForAllUsers.ps1"
if errorlevel 1 (
  echo.
  echo Install did not complete cleanly. Please send the message above to support.
  pause
)
