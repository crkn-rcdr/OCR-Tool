# Managed Windows Workstation Bundle (Administrator Guide)

This package manages the CRKN OCR preservation pipeline on a shared Windows workstation for library and digitization staff.

---

## 📁 Bundle Scripts & Files

| Script (`.cmd` / `.bat`) | PowerShell Backend | Purpose |
| :--- | :--- | :--- |
| **`Start OCR Tool`** | `scripts/Start-OcrTool.ps1` | Verifies shared model cache, starts Docker services (if enabled), launches Gradio UI under the local user, and opens the default browser to `http://127.0.0.1:7860`. |
| **`Stop OCR Tool`** | `scripts/Stop-OcrTool.ps1` | Stops the running Gradio process, removes the PID lock, and stops Docker containers. |
| **`Install For All Users`** | `scripts/Install-ForAllUsers.ps1` | Creates public shortcuts on `C:\Users\Public\Desktop`, creates Start Menu entries, creates `C:\ProgramData\CRKN OCR Tool`, and applies multi-user NTFS ACLs. |
| **`Admin Update OCR Tool`** | `scripts/Update-OcrTool.ps1` | Pulls the latest Git commits, upgrades `requirements.txt` packages, and pulls updated Docker images. |

---

## 🛠️ Workstation Setup for All Users

### 1. Repository Location
The pipeline must reside in a stable, shared local path accessible by all workstation users:
```text
C:\CRKN Projects\paddle_pipeline
```

### 2. Virtual Environment
Ensure the Python virtual environment is active at:
```text
C:\CRKN Projects\paddle_pipeline\venv\Scripts\python.exe
```

### 3. Run the Installer Once as Administrator
Right-click **`Install For All Users.cmd`** (or `Install For All Users.bat`) and select **Run as administrator**.

The installer:
* Creates **`C:\Users\Public\Desktop\CRKN OCR Tool.lnk`** (visible on every user's desktop).
* Creates Start Menu shortcuts under `CRKN OCR Tool`.
* Sets up the shared directory **`C:\ProgramData\CRKN OCR Tool\logs`**.
* Grants the local `Users` group (`*S-1-5-32-545`) Modify permissions to:
  * `C:\ProgramData\CRKN OCR Tool`
  * `C:\CRKN Projects\paddle_pipeline\logs`
  * `C:\CRKN Projects\paddle_pipeline\output`
  * `C:\CRKN Projects\paddle_pipeline\.paddlex` (Shared Model Cache)

---

## 👥 Multi-User Configuration (e.g. `JLoitzenbauer`, `blapierre`)

### 1. Docker Desktop Permissions (`docker-users`)
Docker Desktop is installed **once** on the workstation. Individual staff accounts do **not** need separate Docker installations.

To allow staff accounts (such as `CRKN\JLoitzenbauer`) to use Docker OCR microservices (PaddleVL and OLM OCR), ensure they belong to the local `docker-users` group.

Run in **Windows PowerShell (Admin)**:
```powershell
Add-LocalGroupMember -Group "docker-users" -Member "CRKN\JLoitzenbauer"
Add-LocalGroupMember -Group "docker-users" -Member "CRKN\blapierre"
Add-LocalGroupMember -Group "docker-users" -Member "CRKN\mzheng"
```

To verify membership:
```powershell
Get-LocalGroupMember -Group "docker-users"
```

> [!IMPORTANT]
> Newly added users must **sign out of Windows and sign back in** for `docker-users` group membership to take effect.

---

### 2. Shared Offline Model Cache (`.paddlex`)
PaddleX normally caches official models inside `~/.paddlex` (the user's private profile). Because Windows restricts access between different user directories (`C:\Users\<user>`), running under a new user would cause PaddleX to fail or attempt to re-download 500MB+ of model weights from external servers.

To prevent this:
* Official model weights are stored in the shared project cache:
  ```text
  C:\CRKN Projects\paddle_pipeline\.paddlex\official_models
  ├── PP-DocLayoutV3
  ├── PP-OCRv6_medium_det
  ├── PP-OCRv6_medium_rec
  ├── PP-LCNet_x1_0_doc_ori
  ├── PP-LCNet_x1_0_textline_ori
  └── UVDoc
  ```
* All startup scripts ([`Start-OcrTool.ps1`](file:///c:/CRKN%20Projects/paddle_pipeline/windows_bundle/scripts/Start-OcrTool.ps1), [`launcher.py`](file:///c:/CRKN%20Projects/paddle_pipeline/launcher.py), and [`engine/paddlex_compat.py`](file:///c:/CRKN%20Projects/paddle_pipeline/engine/paddlex_compat.py)) automatically configure:
  ```text
  PADDLE_PDX_CACHE_HOME=C:\CRKN Projects\paddle_pipeline\.paddlex
  PADDLE_HOME=C:\CRKN Projects\paddle_pipeline\.paddlex
  ```
* Any user (including `JLoitzenbauer`) will run OCR completely offline without downloading or encountering permission errors.

---

## ⚡ Automated Benchmarking & CLI Execution

Admins and researchers can run unattended batch jobs and automated ABBYY FineReader comparison benchmarks via the command line:

### 1. Automated Full Benchmark & ABBYY Comparison
```powershell
.\venv\Scripts\python.exe tools/run_full_benchmark.py
```
* **Input**: `C:\Users\blapierre\Documents\PDF Test`
* **Output**: `C:\Users\blapierre\Desktop\PDF Test - Final`
* **Report**: Updates the interactive dashboard at **`compare/index.html`** and writes summary metrics to `compare/comparison_summary.csv`.

### 2. Bulk Microfilm Reel Runner
```powershell
.\venv\Scripts\python.exe tools/batch_runner.py `
  --input "C:\CRKN Projects\Heritage\oocihm.lac_reel_c10143" `
  --output "C:\CRKN Projects\output\lac_reel_c10143" `
  --method PP-OCRv6 `
  --pre-process `
  --post-process `
  --format pdf_alto `
  --resume `
  --concurrency 8
```

---

## 📊 Diagnostic Logs

When troubleshooting workstation issues, consult the logs:

1. **Launcher & Service Logs**:
   ```text
   C:\ProgramData\CRKN OCR Tool\logs
   ├── start.log
   ├── app.stdout.log
   ├── app.stderr.log
   ├── docker-compose.stdout.log
   └── docker-compose.stderr.log
   ```

2. **OCR Engine Application Logs**:
   ```text
   C:\CRKN Projects\paddle_pipeline\logs
   ├── PP_OCRv6_summary_YYYYMMDD_HHMMSS.log
   └── PP_OCRv6_error_YYYYMMDD_HHMMSS.log
   ```
