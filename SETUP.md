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

Ollama uses the Mac GPU through Metal automatically. For Whisper transcription, choose `auto` or `mlx-whisper` on Apple Silicon; `cuda` is only for NVIDIA Windows/Linux.

For M2 16GB, use `cpu` + `int8`, Whisper `medium`, live translation `qwen2.5:7b`, and summary `qwen2.5:7b` or `qwen3:8b`.

For M1 32GB, use the same Whisper settings; summary can use `qwen3:14b` if speed is acceptable.

```bash
brew install python ffmpeg ollama
chmod +x setup_macos.sh run_live_ui.command
./setup_macos.sh
ollama pull qwen2.5:7b
./run_live_ui.command
```