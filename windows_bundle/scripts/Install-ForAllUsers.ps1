param(
    [switch]$NoAclChanges
)

$ErrorActionPreference = "Stop"

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Resolve-ProjectRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

function New-Shortcut {
    param(
        [string]$ShortcutPath,
        [string]$TargetPath,
        [string]$WorkingDirectory,
        [string]$Description
    )
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($ShortcutPath)
    $shortcut.TargetPath = $TargetPath
    $shortcut.WorkingDirectory = $WorkingDirectory
    $shortcut.Description = $Description
    $shortcut.Save()
}

if (-not (Test-IsAdministrator)) {
    Write-Host "[CRKN OCR] Requesting administrator approval..."
    Start-Process powershell.exe `
        -Verb RunAs `
        -ArgumentList @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", "`"$PSCommandPath`""
        )
    exit 0
}

$projectRoot = Resolve-ProjectRoot
$bundleRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$startCmd = Join-Path $bundleRoot "Start OCR Tool.cmd"
$stopCmd = Join-Path $bundleRoot "Stop OCR Tool.cmd"
$updateCmd = Join-Path $bundleRoot "Admin Update OCR Tool.cmd"

if (-not (Test-Path -LiteralPath (Join-Path $projectRoot "app.py"))) {
    throw "Could not find app.py under $projectRoot."
}

$sharedRoot = Join-Path $env:ProgramData "CRKN OCR Tool"
$logRoot = Join-Path $sharedRoot "logs"
$outputRoot = Join-Path $projectRoot "output"
$repoLogRoot = Join-Path $projectRoot "logs"
$modelCacheRoot = Join-Path $projectRoot ".paddlex"

New-Item -ItemType Directory -Force -Path $sharedRoot, $logRoot, $outputRoot, $repoLogRoot, $modelCacheRoot | Out-Null

if (-not $NoAclChanges) {
    Write-Host "[CRKN OCR] Granting local Users modify access to shared logs/output..."
    $usersSid = "*S-1-5-32-545"
    & icacls $sharedRoot /grant "${usersSid}:(OI)(CI)M" /T | Out-Null
    & icacls $outputRoot /grant "${usersSid}:(OI)(CI)M" /T | Out-Null
    & icacls $repoLogRoot /grant "${usersSid}:(OI)(CI)M" /T | Out-Null
    & icacls $modelCacheRoot /grant "${usersSid}:(OI)(CI)M" /T | Out-Null
}

$publicDesktop = [Environment]::GetFolderPath("CommonDesktopDirectory")
$commonPrograms = [Environment]::GetFolderPath("CommonPrograms")
$startMenuFolder = Join-Path $commonPrograms "CRKN OCR Tool"
New-Item -ItemType Directory -Force -Path $startMenuFolder | Out-Null

New-Shortcut `
    -ShortcutPath (Join-Path $publicDesktop "CRKN OCR Tool.lnk") `
    -TargetPath $startCmd `
    -WorkingDirectory $bundleRoot `
    -Description "Start the CRKN OCR Tool"

New-Shortcut `
    -ShortcutPath (Join-Path $startMenuFolder "Start CRKN OCR Tool.lnk") `
    -TargetPath $startCmd `
    -WorkingDirectory $bundleRoot `
    -Description "Start the CRKN OCR Tool"

New-Shortcut `
    -ShortcutPath (Join-Path $startMenuFolder "Stop CRKN OCR Tool.lnk") `
    -TargetPath $stopCmd `
    -WorkingDirectory $bundleRoot `
    -Description "Stop the CRKN OCR Tool"

New-Shortcut `
    -ShortcutPath (Join-Path $startMenuFolder "Admin Update CRKN OCR Tool.lnk") `
    -TargetPath $updateCmd `
    -WorkingDirectory $bundleRoot `
    -Description "Update the CRKN OCR Tool"

Write-Host "[CRKN OCR] Installed shortcuts for all users."
Write-Host "[CRKN OCR] Public Desktop shortcut: $(Join-Path $publicDesktop 'CRKN OCR Tool.lnk')"
Write-Host "[CRKN OCR] Shared logs: $logRoot"
