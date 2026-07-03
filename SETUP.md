# Setup for Friends

Windows 10/11:

1. Install Python 3.10+ and check "Add python.exe to PATH".
2. Download this repo and open PowerShell in the folder.
3. Run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

4. If setup says `ffmpeg` is missing:

```powershell
winget install Gyan.FFmpeg
```

5. If setup says Ollama is missing:

```powershell
winget install Ollama.Ollama
ollama pull qwen2.5:7b
```

6. Start the app:

```text
start_live_caption.bat
```

Open: <http://127.0.0.1:7860>

If install fails, send the contents of `logs/live_caption_launcher.log` and the PowerShell error.

macOS Apple Silicon:

Ollama uses the Mac GPU through Metal automatically. The UI uses `gpu` by default: Apple Silicon maps to MLX/Metal when `mlx-whisper` is installed, and Windows NVIDIA maps to CUDA. No one needs to choose ASR backend or MLX repo/path in normal use.

For M2 16GB, use default `gpu`, Whisper `medium`, live translation `qwen2.5:7b`, and summary `qwen3:14b` or `gpt-oss:20b`.

For M1 32GB, use default `gpu`; summary can try `qwen3:30b` or `gemma3:27b` if speed is acceptable.

```bash
brew install python ffmpeg ollama
chmod +x setup_macos.sh run_live_ui.command
./setup_macos.sh
ollama pull qwen2.5:7b
./run_live_ui.command
```