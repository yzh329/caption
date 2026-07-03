$ErrorActionPreference = "Stop"

Write-Host "Starting local live caption UI..."
Write-Host "Open: http://127.0.0.1:7860"
python -u live_caption_app.py
