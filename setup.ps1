$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

function Has-Command($Name) {
    $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

Write-Host "== Local Live Captions setup =="

if (-not (Has-Command python)) {
    Write-Host ""
    Write-Host "Python not found."
    Write-Host "Install Python 3.10+ from https://www.python.org/downloads/ and enable 'Add python.exe to PATH'."
    exit 1
}

$version = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "Python: $version"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating .venv..."
    python -m venv .venv
}

$python = Join-Path $PWD ".venv\Scripts\python.exe"

Write-Host "Installing Python packages..."
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt

Write-Host ""
if (Has-Command ffmpeg) {
    Write-Host "ffmpeg: ok"
} else {
    Write-Host "ffmpeg: missing"
    Write-Host "Install with: winget install Gyan.FFmpeg"
}

if (Has-Command ollama) {
    Write-Host "Ollama: ok"
    Write-Host "Recommended model: ollama pull qwen2.5:7b"
} else {
    Write-Host "Ollama: missing"
    Write-Host "Install with: winget install Ollama.Ollama"
}

Write-Host ""
Write-Host "Done. Start the app with:"
Write-Host "  .\start_live_caption.bat"
