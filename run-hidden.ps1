# Veo3 Pro Generator - hidden GUI launcher (PowerShell variant).
# Run from a terminal:
#   powershell -ExecutionPolicy Bypass -File run-hidden.ps1
# Prefer double-clicking VeoGUI.vbs - simpler for non-technical users.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$venvPyw = Join-Path $PSScriptRoot '.venv\Scripts\pythonw.exe'
if (-not (Test-Path $venvPyw)) {
    Write-Host 'First run: creating virtual environment - please wait...'
    & cmd /c ("`"$PSScriptRoot\run.bat`"")
    if (-not (Test-Path $venvPyw)) { Write-Error 'Setup failed.'; exit 1 }
}

Start-Process -FilePath $venvPyw -ArgumentList 'veo_gui.py' -WorkingDirectory $PSScriptRoot
