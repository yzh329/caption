# 本地实时会议字幕 / Local Live Meeting Captions

本项目在本机完成会议转写、中文翻译和会议总结。音频、字幕和总结都保存在本地。不需要云端 API，也不需要会议 bot 加入 Zoom / Teams / Meet。

English version is below.

## 中文说明

### 功能

- 默认使用 Chrome/system audio 录音，也可以同时包含麦克风。
- 本地 Whisper 转写英文。
- 连续录音，默认每 3 秒刷新字幕，使用最近约 15 秒音频作为上下文；标记为 `Draft` 的末尾字幕可修订。
- 连续两轮一致的文字确认后才写入正式转录并送去翻译；未稳定音频会保留更久，点 `Stop` 后保存剩余尾段。
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
4. 点 `Start` 开始，点 `Stop` 后等待 `Final subtitles saved`；上传失败时保留页面，恢复连接后再点 `Stop`。
5. 输出文件在 `output/live/...`。

### 模型选择

Whisper:

- `small`: 更快，质量一般。
- `medium`: 较小模型。
- `distil-large-v3`: 当前默认，保留实时识别速度。
- `large-v3`: 完整大模型，可自行比较质量和耗时。

翻译模型：默认 `qwen2.5:7b`，适合实时翻译。

可选试用 `qwen3.5:9b-q4_K_M`（约 6.6GB 下载）；Qwen3 翻译关闭思考，翻译请求关闭 presence penalty。当前会议样本尚未显示稳定质量优势，因此保留原默认值。

Summary chunk model:

- 默认：`hf.co/unsloth/Qwen3.8-27B-GGUF:UD-IQ3_S`（27B，3-bit，模型约 12GB）
- small: `gemma3:12b`
- medium: `qwen3:14b`
- large: `gpt-oss:20b`

Final summary model:

- 默认：同上，Qwen3.8 27B 3-bit
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
ollama pull hf.co/unsloth/Qwen3.8-27B-GGUF:UD-IQ3_S
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

- Whisper: `distil-large-v3` is the default; `medium`, `small`, and full `large-v3` remain available.
- Translation: `qwen2.5:7b` is the default low-latency choice.
- Summary chunk and final summary default: `hf.co/unsloth/Qwen3.8-27B-GGUF:UD-IQ3_S` (27B, 3-bit, about 12GB on disk). CLI and backfill use the same default.
- Alternatives remain available in the UI: `gemma3:12b`, `qwen3:14b`, `gpt-oss:20b`, and `qwen3:30b`.

The UI uses `gpu` by default. Apple Silicon maps to MLX/Metal when available; Windows NVIDIA maps to CUDA; CPU fallback is available.

Audio capture runs continuously. Subtitles refresh every 3 seconds using about 15 seconds of context. The trailing `Draft` can change; text confirmed by two updates is saved and queued for translation. Unstable audio is retained beyond the target window rather than discarded. Stop flushes the remaining audio and pending text before auto-summary. New source audio is saved once per chunk as `audio_*.wav`; `stream_state.npz` permits retry/resume after a server restart.

Run the regression checks with `python -m pytest -q test_live_caption_app.py` and `node test_audio_capture.js`.

### Main Files

- `live_caption_app.py`: local web app/API.
- `static/app.js`: browser recording UI logic.
- `templates/index.html`: controls.
- `transcribe.py`: offline transcription.
- `summarize.py`: summary generation.
- `setup.ps1`, `setup_macos.sh`: setup scripts.
