# Managed Windows Workstation Bundle

This folder packages the OCR app for low-tech users on a shared Windows PC.

## Files

- `Start OCR Tool.cmd`: starts Docker PaddleOCR-VL if available, starts Gradio, opens the browser.
- `Stop OCR Tool.cmd`: stops the Gradio app and Docker services.
- `Install For All Users.cmd`: creates Public Desktop and Start Menu shortcuts for every user.
- `Admin Update OCR Tool.cmd`: pulls repo changes, reinstalls requirements, and pulls Docker images.

## Install For All Users

1. Put the repo in a stable path, for example:

   ```text
   C:\CRKN Projects\paddle_pipeline
   ```

2. Confirm the repo has a working virtual environment:

   ```text
   venv\Scripts\python.exe
   ```

   or:

   ```text
   .venv\Scripts\python.exe
   ```

3. Right-click `Install For All Users.cmd` and choose **Run as administrator**.

The installer creates:

- `C:\Users\Public\Desktop\CRKN OCR Tool.lnk`
- Start Menu entries under `CRKN OCR Tool`
- shared logs under `C:\ProgramData\CRKN OCR Tool\logs`

It grants local `Users` modify access to:

- `C:\ProgramData\CRKN OCR Tool`
- repo `logs`
- repo `output`
- repo `.paddlex`

## Runtime Defaults

The start script sets:

```text
OCR_TOOL_HOST=127.0.0.1
OCR_TOOL_PORT=7860
PADDLE_VL_API_URL=http://127.0.0.1:8080
```

The app is intentionally local to the workstation. Other Windows users on the
same PC can launch it from their own account using the shared shortcut.

## Logs

Wrapper logs:

```text
C:\ProgramData\CRKN OCR Tool\logs
```

App run logs:

```text
<repo>\logs
```
