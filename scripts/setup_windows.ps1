# One-time setup for agent-automaton on Windows.
# Usage (from the project root):  .\scripts\setup_windows.ps1
# If Group Policy blocks unsigned scripts (AllSigned), run instead:
#   Get-Content -Raw .\scripts\setup_windows.ps1 | Invoke-Expression

$ErrorActionPreference = "Stop"
# $PSScriptRoot is empty when piped through Invoke-Expression; assume cwd = project root then.
if ($PSScriptRoot) { $projectRoot = Split-Path -Parent $PSScriptRoot } else { $projectRoot = (Get-Location).Path }
Set-Location $projectRoot

Write-Host "== agent-automaton setup ==" -ForegroundColor Cyan

# 1. Virtual environment
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment (.venv)..."
    python -m venv .venv
} else {
    Write-Host ".venv already exists, skipping."
}

# 2. Dependencies
Write-Host "Installing dependencies..."
& ".venv\Scripts\python.exe" -m pip install --upgrade pip
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt

# 3. Local config
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example"
} else {
    Write-Host ".env already exists, skipping."
}

Write-Host ""
Write-Host "Done. Start the server with:  .\scripts\run_dev.ps1" -ForegroundColor Green
