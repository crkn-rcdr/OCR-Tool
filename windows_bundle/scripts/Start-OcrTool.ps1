param(
    [int]$Port = 7860,
    [switch]$SkipPaddleVl
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

function Start-PaddleVlService {
    param(
        [string]$ProjectRoot,
        [string]$LogFile
    )

    $docker = Get-Command docker -ErrorAction SilentlyContinue
    if ($null -eq $docker) {
        Write-Status "Docker was not found. The app will start, but the post process checkbox needs Docker/PaddleVL."
        return
    }

    Write-Status "Starting PaddleOCR-VL service with Docker Compose..."
    $dockerOutLog = Join-Path (Split-Path -Parent $LogFile) "docker-compose.stdout.log"
    $dockerErrLog = Join-Path (Split-Path -Parent $LogFile) "docker-compose.stderr.log"
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
        Write-Status "Docker Compose did not start cleanly. The app will still start, but post process may be unavailable."
        Add-Content -Path $LogFile -Value "docker compose up -d failed with exit code $($compose.ExitCode). See $dockerErrLog"
        return
    }

    $healthUrl = "http://127.0.0.1:8080/health"
    $deadline = (Get-Date).AddMinutes(8)
    while ((Get-Date) -lt $deadline) {
        if (Test-HttpReady -Url $healthUrl -TimeoutSeconds 3) {
            Write-Status "PaddleOCR-VL service is ready."
            return
        }
        Start-Sleep -Seconds 5
    }

    Write-Status "PaddleOCR-VL did not report healthy yet. The app is still usable without post process."
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

if (Test-HttpReady -Url $url -TimeoutSeconds 2) {
    Write-Status "OCR Tool is already running. Opening browser..."
    Start-Process $url
    exit 0
}

if (-not $SkipPaddleVl) {
    Start-PaddleVlService -ProjectRoot $projectRoot -LogFile $startLog
}

$python = Resolve-RepoPython -ProjectRoot $projectRoot

$env:OCR_TOOL_PROJECT_DIR = $projectRoot
$env:OCR_TOOL_HOST = "127.0.0.1"
$env:OCR_TOOL_PORT = [string]$Port
$env:PADDLE_VL_API_URL = "http://127.0.0.1:8080"

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

$deadline = (Get-Date).AddMinutes(2)
while ((Get-Date) -lt $deadline) {
    if ($process.HasExited) {
        throw "OCR Tool exited immediately. Check $appErrLog."
    }
    if (Test-HttpReady -Url $url -TimeoutSeconds 2) {
        Write-Status "OCR Tool is ready. Opening browser..."
        Start-Process $url
        exit 0
    }
    Start-Sleep -Seconds 2
}

throw "OCR Tool did not become ready in time. Check $appErrLog."
