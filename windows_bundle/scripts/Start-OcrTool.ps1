param(
    [int]$Port = 7860,
    [switch]$SkipDockerServices,
    [switch]$SkipPaddleVl,
    [switch]$SkipOlmOcr
)

$ErrorActionPreference = "Stop"

function Resolve-ProjectRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

function Write-Status {
    param([string]$Message)
    Write-Host "[CRKN OCR] $Message"
}

function Test-HttpReady {
    param(
        [string]$Url,
        [int]$TimeoutSeconds = 2
    )
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec $TimeoutSeconds
        return [int]$response.StatusCode -ge 200 -and [int]$response.StatusCode -lt 500
    } catch {
        return $false
    }
}

function Test-DockerEngineReady {
    param([string]$DockerPath)
    try {
        & $DockerPath info *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Start-DockerDesktopIfNeeded {
    param(
        [string]$DockerPath,
        [string]$LogFile
    )

    if (Test-DockerEngineReady -DockerPath $DockerPath) {
        return $true
    }

    $desktopExe = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (Test-Path -LiteralPath $desktopExe) {
        Write-Status "Docker engine is not running. Starting Docker Desktop..."
        Start-Process `
            -FilePath $desktopExe `
            -WindowStyle Hidden
    } else {
        Write-Status "Docker Desktop was not found at $desktopExe."
        Add-Content -Path $LogFile -Value "Docker Desktop executable missing: $desktopExe"
        return $false
    }

    $deadline = (Get-Date).AddMinutes(5)
    while ((Get-Date) -lt $deadline) {
        if (Test-DockerEngineReady -DockerPath $DockerPath) {
            Write-Status "Docker engine is ready."
            return $true
        }
        Start-Sleep -Seconds 5
    }

    Write-Status "Docker engine did not become ready in time. Docker OCR services may be unavailable."
    Write-Status "Confirm this Windows user is in the local docker-users group, then sign out and back in."
    Add-Content -Path $LogFile -Value "Docker engine did not become ready within 5 minutes."
    Add-Content -Path $LogFile -Value "Check docker-users membership for the current Windows user and sign out/in after changes."
    return $false
}

function Resolve-RepoPython {
    param([string]$ProjectRoot)
    $candidates = @(
        (Join-Path $ProjectRoot "venv\Scripts\python.exe"),
        (Join-Path $ProjectRoot ".venv\Scripts\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    throw "No repo virtual environment was found. Expected venv\Scripts\python.exe or .venv\Scripts\python.exe under $ProjectRoot."
}

function Wait-DockerHttpReady {
    param(
        [string]$Url,
        [string]$ReadyMessage,
        [string]$TimeoutMessage,
        [int]$Minutes = 8
    )

    $deadline = (Get-Date).AddMinutes($Minutes)
    while ((Get-Date) -lt $deadline) {
        if (Test-HttpReady -Url $Url -TimeoutSeconds 3) {
            Write-Status $ReadyMessage
            return $true
        }
        Start-Sleep -Seconds 5
    }

    Write-Status $TimeoutMessage
    return $false
}

function Start-DockerServices {
    param(
        [string]$ProjectRoot,
        [string]$LogFile,
        [switch]$Wait
    )

    $docker = Get-Command docker -ErrorAction SilentlyContinue
    if ($null -eq $docker) {
        Write-Status "Docker was not found. The app will run with PaddleOCR and Indra OCR."
        return
    }

    if (-not (Test-DockerEngineReady -DockerPath $docker.Source)) {
        $desktopExe = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        if (Test-Path -LiteralPath $desktopExe) {
            Write-Status "Starting Docker Desktop in background for optional VLM services..."
            Start-Process -FilePath $desktopExe -WindowStyle Hidden
        } else {
            Write-Status "Docker Desktop was not found at $desktopExe."
            Add-Content -Path $LogFile -Value "Docker Desktop executable missing: $desktopExe"
        }
        return
    }

    Write-Status "Starting OCR model services with Docker Compose..."
    $dockerOutLog = Join-Path (Split-Path -Parent $LogFile) "docker-compose.stdout.log"
    $dockerErrLog = Join-Path (Split-Path -Parent $LogFile) "docker-compose.stderr.log"

    if ($Wait) {
        $compose = Start-Process `
            -FilePath $docker.Source `
            -ArgumentList @("compose", "up", "-d") `
            -WorkingDirectory $ProjectRoot `
            -RedirectStandardOutput $dockerOutLog `
            -RedirectStandardError $dockerErrLog `
            -WindowStyle Hidden `
            -Wait `
            -PassThru

        if ($compose.ExitCode -ne 0) {
            Write-Status "Docker Compose did not start cleanly. The app is still usable. See $dockerErrLog"
            Add-Content -Path $LogFile -Value "docker compose up -d failed with exit code $($compose.ExitCode). See $dockerErrLog"
            return
        }

        if (-not $SkipOlmOcr) {
            Wait-DockerHttpReady `
                -Url "http://127.0.0.1:8000/v1/models" `
                -ReadyMessage "OLM OCR service is ready." `
                -TimeoutMessage "OLM OCR did not report healthy yet. OLM OCR may be unavailable." `
                -Minutes 5 | Out-Null
        }

        if (-not $SkipPaddleVl) {
            Wait-DockerHttpReady `
                -Url "http://127.0.0.1:8080/health" `
                -ReadyMessage "PaddleOCR-VL service is ready." `
                -TimeoutMessage "PaddleOCR-VL did not report healthy yet. The app is still usable without post process." `
                -Minutes 5 | Out-Null
        }
    } else {
        Start-Process `
            -FilePath $docker.Source `
            -ArgumentList @("compose", "up", "-d") `
            -WorkingDirectory $ProjectRoot `
            -RedirectStandardOutput $dockerOutLog `
            -RedirectStandardError $dockerErrLog `
            -WindowStyle Hidden
    }
}

$projectRoot = Resolve-ProjectRoot
$appPath = Join-Path $projectRoot "app.py"
if (-not (Test-Path -LiteralPath $appPath)) {
    throw "Could not find app.py under $projectRoot."
}

$sharedRoot = Join-Path $env:ProgramData "CRKN OCR Tool"
$logRoot = Join-Path $sharedRoot "logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$startLog = Join-Path $logRoot "start.log"
$appOutLog = Join-Path $logRoot "app.stdout.log"
$appErrLog = Join-Path $logRoot "app.stderr.log"
$pidFile = Join-Path $sharedRoot "app.pid"

$url = "http://127.0.0.1:$Port"

Write-Status "Project: $projectRoot"
Write-Status "URL: $url"

# Check if the OCR Tool is already running on the port
if (Test-HttpReady -Url $url -TimeoutSeconds 2) {
    Write-Status "OCR Tool is already running. Opening browser..."
    Start-Process $url
    exit 0
}

# Resolve Python environment and prepare environment variables
$python = Resolve-RepoPython -ProjectRoot $projectRoot

$env:OCR_TOOL_PROJECT_DIR = $projectRoot
$env:OCR_TOOL_HOST = "127.0.0.1"
$env:OCR_TOOL_PORT = [string]$Port
$env:PADDLE_PDX_CACHE_HOME = Join-Path $projectRoot ".paddlex"
$env:PADDLE_HOME = Join-Path $projectRoot ".paddlex"
$env:INDRA_OCR_MODEL_DIR = Join-Path $projectRoot "models\indra"
$env:PADDLE_VL_API_URL = "http://127.0.0.1:8080"
$env:OLMOCR_API_URL = "http://127.0.0.1:8000/v1"
if (-not $env:OLMOCR_MODEL) {
    $env:OLMOCR_MODEL = "allenai/olmOCR-2-7B-1025-FP8"
}

Write-Status "Starting OCR Tool app..."
$process = Start-Process `
    -FilePath $python `
    -ArgumentList @("`"$appPath`"") `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $appOutLog `
    -RedirectStandardError $appErrLog `
    -WindowStyle Hidden `
    -PassThru

Set-Content -Path $pidFile -Value $process.Id -Encoding ASCII

# Wait for Gradio web server to be ready
$deadline = (Get-Date).AddMinutes(2)
while ((Get-Date) -lt $deadline) {
    if ($process.HasExited) {
        throw "OCR Tool exited immediately. Check $appErrLog."
    }
    if (Test-HttpReady -Url $url -TimeoutSeconds 2) {
        Write-Status "OCR Tool is ready. Opening browser..."
        Start-Process $url
        break
    }
    Start-Sleep -Seconds 2
}

if (-not (Test-HttpReady -Url $url -TimeoutSeconds 2)) {
    throw "OCR Tool did not become ready in time. Check $appErrLog."
}

# Optionally start Docker services in background for VLM models
if (-not $SkipDockerServices) {
    Start-DockerServices -ProjectRoot $projectRoot -LogFile $startLog
}

exit 0
