$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Weather Trader desktop launcher (Windows PowerShell)
# Backend is auto-started by app_new.py if ../weather-engine is present.

if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv run python app_new.py
    exit $LASTEXITCODE
}

if (Get-Command python -ErrorAction SilentlyContinue) {
    python app_new.py
    exit $LASTEXITCODE
}

Write-Host "[ERROR] Neither 'uv' nor 'python' found in PATH." -ForegroundColor Red
Write-Host "Install Python 3.12+ and uv, then retry."
exit 1
