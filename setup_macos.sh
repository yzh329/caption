#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "== Local Live Captions macOS setup =="

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 not found. Install it with: brew install python"
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "Creating .venv..."
  python3 -m venv .venv
fi

echo "Installing Python packages..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install mlx-whisper

if command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg: ok"
else
  echo "ffmpeg: missing. Install with: brew install ffmpeg"
fi

if command -v ollama >/dev/null 2>&1; then
  echo "Ollama: ok"
  echo "Ollama uses the Mac GPU through Metal automatically."
  echo "M2 16GB safe model: ollama pull qwen2.5:7b"
  echo "M1 32GB better summary model: ollama pull qwen3:14b"
else
  echo "Ollama: missing. Install with: brew install ollama"
fi

echo ""
echo "Done. Start with:"
echo "  ./run_live_ui.command"
