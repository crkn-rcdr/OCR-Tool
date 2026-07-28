param(
    [switch]$KeepPaddleVl
)

$ErrorActionPreference = "Stop"

function Resolve-ProjectRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

function Write-Status {
    param([string]$Message)
    Write-Host "[CRKN OCR] $Message"
}

$projectRoot = Resolve-ProjectRoot
$sharedRoot = Join-Path $env:ProgramData "CRKN OCR Tool"
$pidFile = Join-Path $sharedRoot "app.pid"

if (Test-Path -LiteralPath $pidFile) {
    $pidValue = (Get-Content -LiteralPath $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($pidValue -match "^\d+$") {
        $process = Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue
        if ($null -ne $process) {
            Write-Status "Stopping OCR Tool app process $pidValue..."
            Stop-Process -Id ([int]$pidValue) -Force
        }
    }
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
}

$processes = Get-CimInstance Win32_Process |
    Where-Object {
        $_.CommandLine -and
        $_.CommandLine -match "app\.py" -and
        $_.CommandLine -match [regex]::Escape($projectRoot)
    }

foreach ($item in $processes) {
    Write-Status "Stopping leftover OCR Tool process $($item.ProcessId)..."
    Stop-Process -Id $item.ProcessId -Force -ErrorAction SilentlyContinue
}

if (-not $KeepPaddleVl) {
    $docker = Get-Command docker -ErrorAction SilentlyContinue
    if ($null -ne $docker) {
        Write-Status "Stopping PaddleOCR-VL Docker services..."
        Push-Location $projectRoot
        try {
            & docker compose stop
        } finally {
            Pop-Location
        }
    }
}

Write-Status "Stopped."
