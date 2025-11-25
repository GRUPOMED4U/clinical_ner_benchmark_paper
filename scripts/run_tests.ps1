# PowerShell test runner for the repository
# Usage: Open PowerShell in repo root and run: ./scripts/run_tests.ps1

$venvPath = "$PSScriptRoot\..\.venv"
$python = "$venvPath\Scripts\python.exe"

# Create venv if missing
if (-Not (Test-Path $python)) {
    Write-Host "Creating virtual environment at $venvPath"
    python -m venv "$venvPath"
}

# Upgrade pip and install minimal test deps
& $python -m pip install --upgrade pip setuptools wheel
& $python -m pip install pytest pytest-mock numpy pydantic jsonlines pandas tqdm -q

# Run pytest
& $python -m pytest -q
