# Run the agent-automaton dev server.
# Usage (from the project root):  .\scripts\run_dev.ps1
# If Group Policy blocks unsigned scripts (AllSigned), run instead:
#   Get-Content -Raw .\scripts\run_dev.ps1 | Invoke-Expression

$ErrorActionPreference = "Stop"
# $PSScriptRoot is empty when piped through Invoke-Expression; assume cwd = project root then.
if ($PSScriptRoot) { $projectRoot = Split-Path -Parent $PSScriptRoot } else { $projectRoot = (Get-Location).Path }
Set-Location $projectRoot

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $python = $venvPython
} else {
    Write-Host "No .venv found - using system python. Run scripts\setup_windows.ps1 first for an isolated setup." -ForegroundColor Yellow
    $python = "python"
}

& $python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
