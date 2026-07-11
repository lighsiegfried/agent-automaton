# Run the test suite. Extra arguments are passed to pytest.
# Usage (from the project root):  .\scripts\test.ps1 [-- pytest args]
# If Group Policy blocks unsigned scripts (AllSigned), run instead:
#   Get-Content -Raw .\scripts\test.ps1 | Invoke-Expression

$ErrorActionPreference = "Stop"
# $PSScriptRoot is empty when piped through Invoke-Expression; assume cwd = project root then.
if ($PSScriptRoot) { $projectRoot = Split-Path -Parent $PSScriptRoot } else { $projectRoot = (Get-Location).Path }
Set-Location $projectRoot

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $python = $venvPython
} else {
    Write-Host "No .venv found - using system python. Run scripts\setup_windows.ps1 first." -ForegroundColor Yellow
    $python = "python"
}

& $python -m pytest @args
# Propagate pytest's exit code, but never `exit` an Invoke-Expression'd shell.
if ($PSScriptRoot) { exit $LASTEXITCODE }
