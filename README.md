# 本地实时会议字幕 / Local Live Meeting Captions

本项目在本机完成会议转写、中文翻译和会议总结。音频、字幕和总结都保存在本地。不需要云端 API，也不需要会议 bot 加入 Zoom / Teams / Meet。

English version is below.

## 中文说明

### 功能

- 默认使用 Chrome/system audio 录音，也可以同时包含麦克风。
- 本地 Whisper 转写英文。
- 可选中文翻译和段落修复。
- 会议结束后一键生成 short / deep summary。
- Windows 和 Apple Silicon Mac 都可用。
- 默认 `gpu`：Mac 自动用 MLX/Metal，Windows NVIDIA 自动用 CUDA。

### 快速开始

Windows 10/11:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
start_live_caption.bat
```

如果提示缺少依赖：

```powershell
winget install Gyan.FFmpeg
winget install Ollama.Ollama
```

macOS Apple Silicon:

```bash
brew install python ffmpeg ollama
chmod +x setup_macos.sh run_live_ui.command
./setup_macos.sh
./run_live_ui.command
```

打开：<http://127.0.0.1:7860>

### 使用流程

1. 点 `Preload all`，提前下载/加载模型。
2. `Audio source` 默认是 Chrome/system audio；Chrome 分享窗口里要勾选音频共享。
3. `Include mic` 默认开启，会把麦克风也混进去。
4. 点 `Start` 开始，点 `Stop` 结束。
5. 输出文件在 `output/live/...`。

### 模型选择

Whisper:

- `small`: 更快，质量一般。
- `medium`: 默认推荐，速度和质量平衡。
- `large-v3` / `distil-large-v3`: 更准，但更慢。

翻译模型：默认 `qwen2.5:7b`，适合实时翻译。

Summary chunk model:

- small: `gemma3:12b`
- medium: `qwen3:14b`
- large: `gpt-oss:20b`

Final summary model:

- small: `qwen3:14b`
- medium: `gpt-oss:20b`
- large: `qwen3:30b`

推荐：

- M2 16GB: Whisper `medium`, translation `qwen2.5:7b`, summary `qwen3:14b` 或 `gpt-oss:20b`。
- M1 32GB: 可以尝试 final summary `qwen3:30b`。
- Windows NVIDIA: Whisper `large-v3` 或 `distil-large-v3`。

### 常见问题

Chrome 没录到声音：重新分享屏幕/标签页，并确认勾选 `Share audio`。

Ollama 模型缺失：

```bash
ollama pull qwen2.5:7b
ollama pull qwen3:14b
ollama pull gpt-oss:20b
```

Mac 没装 mlx-whisper：

```bash
.venv/bin/python -m pip install mlx-whisper
```

生成文件已被 `.gitignore` 忽略：`logs/`, `output/`, `__pycache__/`, `*.pyc`。

## English

This repo runs local meeting transcription, Chinese translation, and summaries on Windows or macOS. Audio and outputs stay on your machine.

### Quick Start

Windows:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
start_live_caption.bat
```

macOS Apple Silicon:

```bash
brew install python ffmpeg ollama
chmod +x setup_macos.sh run_live_ui.command
./setup_macos.sh
./run_live_ui.command
```

Open: <http://127.0.0.1:7860>

### Usage

1. Click `Preload all`.
2. Keep the default Chrome/system audio source and enable audio sharing in Chrome.
3. `Include mic` is enabled by default.
4. Click `Start`, then `Stop`.
5. Outputs are saved in `output/live/...`.

### Models

- Whisper: `medium` is the default balance; use `small` for speed, `large-v3` or `distil-large-v3` for quality.
- Translation: `qwen2.5:7b` is the default low-latency choice.
- Summary chunk: `gemma3:12b`, `qwen3:14b`, or `gpt-oss:20b`.
- Final summary: `qwen3:14b`, `gpt-oss:20b`, or `qwen3:30b`.

The UI uses `gpu` by default. Apple Silicon maps to MLX/Metal when available; Windows NVIDIA maps to CUDA; CPU fallback is available.

### Main Files

- `live_caption_app.py`: local web app/API.
- `static/app.js`: browser recording UI logic.
- `templates/index.html`: controls.
- `transcribe.py`: offline transcription.
- `summarize.py`: summary generation.
- `setup.ps1`, `setup_macos.sh`: setup scripts.
