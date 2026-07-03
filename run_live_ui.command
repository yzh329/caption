#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python3"
fi

if command -v ollama >/dev/null 2>&1; then
  if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    ollama serve >/tmp/local-live-captions-ollama.log 2>&1 &
    sleep 2
  fi
fi

echo "Open: http://127.0.0.1:7860"
"$PYTHON" -u live_caption_app.py
