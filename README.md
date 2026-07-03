# 本地会议记录系统：Windows 11 + RTX 5080

这套方案不使用云端 OpenAI API，不需要会议 bot 加入 Zoom / Teams / Google Meet。会议声音先由本机录下来，再用本地 ASR 转写，最后用本地 Ollama LLM 生成会议纪要。

## A. 推荐架构图

```text
Zoom / Teams / Meet / Browser / 系统声音
        |
        v
OBS Studio 或 VB-Cable / VoiceMeeter
        |
        v
本地音频文件 .mkv / .m4a / .wav / .mp4
        |
        v
Python transcribe.py
  - 优先 faster-whisper + NVIDIA CUDA
  - GPU 不兼容时自动 CPU fallback
        |
        v
transcript.md
        |
        v
Python summarize.py -> Ollama 本地 LLM
        |
        v
meeting_notes.md
```

推荐音频捕获路线：

- 最简单：OBS Studio 录制“桌面音频”和“麦克风/Aux”，导出 `.mkv` 或 `.mp4`。
- 更稳定：VB-Cable 或 VoiceMeeter 把会议软件声音和麦克风混到一个虚拟输入，再由 OBS 录制。
- 只要最终有本地音频/视频文件，本项目即可处理。

## B. 一步步安装指南

### 1. 安装 NVIDIA 驱动并验证 RTX 5080

RTX 5080 是 Blackwell，NVIDIA 官方列出的 CUDA Capability 是 12.0。CUDA 12.8 文档加入了 SM_120 编译支持，所以不要默认旧 CUDA / 旧 PyTorch / 旧二进制包一定可用。

PowerShell：

```powershell
nvidia-smi
```

你应该看到 `GeForce RTX 5080`、驱动版本、CUDA Version。建议安装较新的 NVIDIA Studio Driver 或 Game Ready Driver。

### 2. 安装 Python

建议 Python 3.10 或 3.11。验证：

```powershell
python --version
```

### 3. 创建虚拟环境

在本目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

验证 faster-whisper 能导入：

```powershell
python -c "import faster_whisper; print('faster-whisper ok')"
python -c "import ctranslate2; print('ctranslate2', ctranslate2.__version__)"
```

### 4. 安装 FFmpeg

推荐 winget：

```powershell
winget install Gyan.FFmpeg
```

关闭并重新打开 PowerShell 后验证：

```powershell
ffmpeg -version
```

### 5. 安装 Ollama

下载并安装 Ollama for Windows：

```powershell
winget install Ollama.Ollama
```

启动/验证：

```powershell
ollama --version
ollama serve
```

另开一个 PowerShell 拉取本地模型。RTX 5080 16GB 显存建议：

```powershell
ollama pull qwen2.5:14b
ollama run qwen2.5:14b "用一句话说：本地会议纪要系统已就绪。"
```

显存或内存紧张时用更小模型：

```powershell
ollama pull qwen2.5:7b
```

英文会议较多也可用：

```powershell
ollama pull llama3.1:8b
```

### 6. OBS 录音设置

1. 安装 OBS Studio。
2. 设置 -> 音频：
   - 桌面音频：默认设备，或选择你的耳机/扬声器。
   - 麦克风/Aux：选择你的麦克风。
3. 设置 -> 输出 -> 录制：
   - 录制格式建议 `.mkv`，崩溃时更不容易损坏。
   - 音频码率 160 或 192 kbps 即可。
4. 开会前点击“开始录制”，结束后“停止录制”。
5. 得到本地 `.mkv` / `.mp4` 文件后交给脚本处理。

## C. 常用 PowerShell 命令

激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

转写一个录音：

```powershell
python transcribe.py "C:\Meetings\meeting.mkv" -o output\transcript.md --model large-v3 --device cuda --compute-type float16 --vad-filter
```

总结：

```powershell
python summarize.py output\transcript.md -o output\meeting_notes.md --model qwen2.5:14b
```

一键流水线：

```powershell
.\run_pipeline.ps1 -InputFile "C:\Meetings\meeting.mkv" -OutDir output -WhisperModel large-v3 -WhisperDevice cuda -ComputeType float16 -OllamaModel qwen2.5:14b -VadFilter
```

如果 GPU 报错，先跑 CPU：

```powershell
.\run_pipeline.ps1 -InputFile "C:\Meetings\meeting.mkv" -OutDir output_cpu -WhisperModel medium -WhisperDevice cpu -ComputeType int8 -OllamaModel qwen2.5:7b
```

## D. Python 项目结构

```text
caption/
  requirements.txt
  transcribe.py
  summarize.py
  run_pipeline.ps1
  README.md
  output/
    transcript.md
    meeting_notes.md
```

## E. requirements.txt

见 `requirements.txt`。

## F. transcribe.py

见 `transcribe.py`。它会输出带时间戳的 `transcript.md`，并记录使用的模型、设备、compute type。

## G. summarize.py

见 `summarize.py`。它调用本地 Ollama 的 `http://localhost:11434/api/chat`，自动切分长 transcript，再合并为：

- meeting summary
- key decisions
- action items
- open questions
- timeline / topics
- risks / follow-ups

## H. run_pipeline.ps1

见 `run_pipeline.ps1`。

## I. 1-2 分钟音频测试方法

### 方法 1：OBS 真实录音测试

1. 打开 OBS，开始录制。
2. 播放一段中英文混合音频，自己再对着麦克风说几句中文和英文。
3. 录 1-2 分钟后停止，假设文件是 `C:\Meetings\test.mkv`。
4. 执行：

```powershell
.\run_pipeline.ps1 -InputFile "C:\Meetings\test.mkv" -OutDir test_output -WhisperModel medium -WhisperDevice cuda -ComputeType float16 -OllamaModel qwen2.5:7b -VadFilter
```

检查：

```powershell
notepad test_output\transcript.md
notepad test_output\meeting_notes.md
```

### 方法 2：不用 OBS，手头已有音频

```powershell
python transcribe.py ".\sample.wav" -o test_output\transcript.md --model medium --device cuda --compute-type float16
python summarize.py test_output\transcript.md -o test_output\meeting_notes.md --model qwen2.5:7b
```

## J. 常见错误排查

### CUDA not available

验证：

```powershell
nvidia-smi
python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"
```

处理：

- 更新 NVIDIA 驱动。
- RTX 5080 是新架构，旧 CUDA/CTranslate2/PyTorch 轮子可能不支持。
- 先用 CPU 跑通：

```powershell
python transcribe.py "C:\Meetings\meeting.mkv" -o output\transcript.md --model medium --device cpu --compute-type int8
```

### cuDNN / CTranslate2 error

常见表现：`CUDNN_STATUS...`、`CUDA driver version is insufficient`、`no kernel image is available`。

处理顺序：

```powershell
pip install --upgrade faster-whisper ctranslate2
nvidia-smi
python -c "import ctranslate2; print(ctranslate2.__version__, ctranslate2.get_cuda_device_count())"
```

如果仍失败：

```powershell
python transcribe.py "C:\Meetings\meeting.mkv" -o output\transcript.md --model large-v3 --device cpu --compute-type int8
```

或者改用 whisper.cpp fallback。思路：

```powershell
winget install Git.Git
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
```

然后按 whisper.cpp 官方 Windows CUDA/Vulkan 文档编译。RTX 5080 上优先选支持新 CUDA/Blackwell 的最新版源码；如果 CUDA 编译失败，先用 CPU 版验证流程。

### ffmpeg not found

验证：

```powershell
ffmpeg -version
```

处理：

```powershell
winget install Gyan.FFmpeg
```

重新打开 PowerShell，再验证。

### Ollama connection refused

验证：

```powershell
ollama list
```

如果服务没启动：

```powershell
ollama serve
```

另开 PowerShell：

```powershell
ollama run qwen2.5:7b "hello"
python summarize.py output\transcript.md -o output\meeting_notes.md --model qwen2.5:7b
```

### 中文识别质量差

建议：

- 模型从 `medium` 换成 `large-v3`。
- 音频尽量用 WAV/M4A 高码率，避免会议软件降噪过重。
- 麦克风和系统声音分轨录制，再混音前确保音量不过载。
- 中英混合时不要强制 `--language zh`，让 Whisper 自动识别。
- 如果会议几乎全中文，可试：

```powershell
python transcribe.py "C:\Meetings\meeting.mkv" -o output\transcript.md --model large-v3 --device cuda --compute-type float16 --language zh --vad-filter
```

### 长会议切分问题

`transcribe.py` 会按模型输出段落写入 transcript；`summarize.py` 会按字符数切分。长会议建议：

```powershell
python summarize.py output\transcript.md -o output\meeting_notes.md --model qwen2.5:14b --max-chars 12000
```

如果转写本身太慢或内存紧张，先把音频切成 30 分钟段：

```powershell
ffmpeg -i "C:\Meetings\long.mkv" -f segment -segment_time 1800 -c copy "C:\Meetings\part_%03d.mkv"
```

逐个转写后把多个 transcript 合并再总结。

## 推荐默认配置

RTX 5080 + 16GB 显存：

```powershell
.\run_pipeline.ps1 -InputFile "C:\Meetings\meeting.mkv" -OutDir output -WhisperModel large-v3 -WhisperDevice cuda -ComputeType float16 -OllamaModel qwen2.5:14b -VadFilter
```

稳定优先：

```powershell
.\run_pipeline.ps1 -InputFile "C:\Meetings\meeting.mkv" -OutDir output -WhisperModel medium -WhisperDevice cpu -ComputeType int8 -OllamaModel qwen2.5:7b -VadFilter
```

## 实时字幕 / 实时滚动文档 UI

如果会议主要是英文，但你想要中文翻译，可以使用本地 Web UI：

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull qwen2.5:7b
.\run_live_ui.ps1
```

然后在浏览器打开：

```text
http://127.0.0.1:7860
```

也可以直接双击：

```text
start_live_caption.bat
```

这个启动器会自动进入项目目录、激活 `.venv`、检查依赖、打开浏览器并启动本地 UI。使用时请保持黑色命令行窗口打开；关闭窗口就会停止服务。

如果你不想看到命令行窗口，可以双击：

```text
start_live_caption_hidden.vbs
```

隐藏启动方式不会显示日志；如果启动失败，建议改用 `start_live_caption.bat` 看错误信息。

页面里选择：

- `Microphone (browser permission popup)`：通过 Chrome 录一个输入设备。允许麦克风权限后，可在 `Browser input` 里选真实麦克风、VB-Cable、VoiceMeeter 等设备。
- `System audio via Chrome screen sharing`：通过 Chrome 屏幕共享录系统声音。Chrome 弹窗里必须选择带音频的共享目标，并勾选 `Share audio` / `Share system audio`。
- `Whisper model`：先用 `medium` 测试，质量不够再用 `large-v3`。
- `Device`：RTX 5080 优先选 `cuda`；如果报 CUDA/CTranslate2 错误，选 `cpu`。
- `Translate to Chinese`：开启后会调用本地 Ollama，把英文段落翻译成中文。

UI 会实时往下追加：

```text
时间
English transcript
中文翻译
```

同时保存到：

```text
output/live/YYYYMMDD_HHMMSS/live_transcript.md
```

实时模式不是逐词字幕，而是“分段准实时”：默认每 6 秒识别一次。你可以把 `Chunk seconds` 改成 3-5 秒降低延迟，但准确率可能略降；改成 8-12 秒通常更稳。

### 实时 UI 验证命令

```powershell
python -m py_compile live_caption_app.py
python live_caption_app.py
```

打开页面后：

1. 选择 `Microphone (browser permission popup)`。
2. 播放一段 1-2 分钟英文视频或会议录音。
3. 点击 `Start`。
4. 看页面是否开始出现英文和中文。
5. 点击 `Stop`，检查 `output/live/.../live_transcript.md`。

### Chrome 提示 No audio track selected

如果看到 `No audio track selected`，说明 Chrome 的屏幕共享没有拿到音频轨道。处理方式：

- 共享 Chrome 标签页时，勾选 `Share tab audio`。
- 共享桌面应用时，优先选 `Entire Screen`，不要选单个窗口，并勾选 `Share system audio`。
- 如果弹窗里没有音频勾选项，Chrome 无法捕获这个目标的声音。
- Zoom 桌面客户端最稳方案是 VB-Cable / VoiceMeeter：把 Zoom speaker 输出路由到虚拟输入，然后在 UI 里选择 `Microphone (browser permission popup)`，并在 `Browser input` 里选 VB-Cable / VoiceMeeter。

### 实时 UI 常见问题

如果页面没有音频源：

```powershell
python -c "import soundcard as sc; print(sc.all_speakers()); print(sc.all_microphones())"
```

如果只有系统声音、没有你的声音：

- 在会议软件里确保麦克风声音没有只发给对方而不经过本机播放设备。
- 简单方案：OBS/VoiceMeeter 混音后录文件，再跑离线 pipeline。
- 进阶方案：VoiceMeeter 把会议软件声音和麦克风混到同一个虚拟输出，再在 UI 里选择对应 `System audio`。

如果翻译一直失败：

```powershell
ollama serve
ollama run qwen2.5:7b "Translate to Chinese: The meeting starts now."
```

## 官方兼容性依据

- NVIDIA RTX 5080 官方规格页列出 Blackwell 架构与 CUDA Capability 12.0。
- NVIDIA CUDA 12.8 Features Archive 写明 CUDA 12.8 增加了 SM_100、SM_101、SM_120 编译支持。
