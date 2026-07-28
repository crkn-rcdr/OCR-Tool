param(
    [switch]$SkipGitPull,
    [switch]$SkipDockerPull
)

$ErrorActionPreference = "Stop"

function Resolve-ProjectRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
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
    throw "No repo virtual environment was found. Expected venv or .venv under $ProjectRoot."
}

$projectRoot = Resolve-ProjectRoot
$python = Resolve-RepoPython -ProjectRoot $projectRoot

Push-Location $projectRoot
try {
    if (-not $SkipGitPull -and (Test-Path -LiteralPath ".git")) {
        Write-Host "[CRKN OCR] Pulling latest repo changes..."
        & git pull
    }

    Write-Host "[CRKN OCR] Installing Python requirements..."
    & $python -m pip install -r requirements.txt

    if (-not $SkipDockerPull) {
        $docker = Get-Command docker -ErrorAction SilentlyContinue
        if ($null -ne $docker) {
            Write-Host "[CRKN OCR] Pulling Docker images..."
            & docker compose pull
        } else {
            Write-Host "[CRKN OCR] Docker was not found; skipping Docker image update."
        }
    }
} finally {
    Pop-Location
}

Write-Host "[CRKN OCR] Update complete."
