@echo off
setlocal

cd /d "%~dp0"
title Local Live Meeting Captions

if not exist "logs" mkdir "logs"
set "LAUNCH_LOG=logs\live_caption_launcher.log"
set "TORCH_CUDA_DLLS=%USERPROFILE%\miniconda3\envs\dl\Lib\site-packages\torch\lib"
set "OLLAMA_EXE="

echo Starting Local Live Meeting Captions...
echo [%date% %time%] Starting Local Live Meeting Captions...>> "%LAUNCH_LOG%"
echo.

if exist "%TORCH_CUDA_DLLS%\cublas64_12.dll" (
  set "PATH=%TORCH_CUDA_DLLS%;%PATH%"
  echo Using CUDA DLLs from: %TORCH_CUDA_DLLS%
  echo [%date% %time%] Added CUDA DLL path: %TORCH_CUDA_DLLS%>> "%LAUNCH_LOG%"
)

set "PYTHON_EXE=python"
if exist "%USERPROFILE%\miniconda3\python.exe" (
  set "PYTHON_EXE=%USERPROFILE%\miniconda3\python.exe"
)
if exist ".venv\Scripts\python.exe" (
  set "PYTHON_EXE=%cd%\.venv\Scripts\python.exe"
)

echo Using Python: %PYTHON_EXE%
"%PYTHON_EXE%" -c "import sys, os; print('Python:', sys.executable); print('cublas path exists:', os.path.exists(r'%TORCH_CUDA_DLLS%\cublas64_12.dll'))"

for %%P in (
  "%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
  "%ProgramFiles%\Ollama\ollama.exe"
  "%ProgramFiles(x86)%\Ollama\ollama.exe"
) do (
  if exist "%%~P" set "OLLAMA_EXE=%%~P"
)

if not defined OLLAMA_EXE (
  for /f "delims=" %%P in ('where ollama 2^>nul') do (
    if not defined OLLAMA_EXE set "OLLAMA_EXE=%%P"
  )
)

if defined OLLAMA_EXE (
  powershell -NoProfile -Command "try { $c = Test-NetConnection -ComputerName 127.0.0.1 -Port 11434 -WarningAction SilentlyContinue; if ($c.TcpTestSucceeded) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>nul
  if errorlevel 1 (
    echo Starting Ollama server...
    echo [%date% %time%] Starting Ollama: %OLLAMA_EXE%>> "%LAUNCH_LOG%"
    start "Ollama Server" /min "%OLLAMA_EXE%" serve
    timeout /t 3 /nobreak >nul
  ) else (
    echo Ollama server already running.
    echo [%date% %time%] Ollama already running.>> "%LAUNCH_LOG%"
  )
) else (
  echo Ollama not found. Translation and summary need Ollama.
  echo Install from https://ollama.com/download/windows
  echo [%date% %time%] Ollama not found.>> "%LAUNCH_LOG%"
)

"%PYTHON_EXE%" -c "import flask, requests, faster_whisper" >nul 2>nul
if errorlevel 1 (
  echo Missing Python dependencies.
  echo.
  echo Installing from requirements.txt...
  "%PYTHON_EXE%" -m pip install --upgrade pip
  "%PYTHON_EXE%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo Install failed. Check your Python and network, then run this file again.
    pause
    exit /b 1
  )
)

echo.
echo UI: http://127.0.0.1:7860
echo Keep this window open while using live captions.
echo Close this window to stop the local UI.
echo.

set /a RESTARTS=0

:run_app
echo [%date% %time%] Launching Python app. Restart count: %RESTARTS%>> "%LAUNCH_LOG%"
"%PYTHON_EXE%" -u live_caption_app.py
set "EXIT_CODE=%ERRORLEVEL%"
echo [%date% %time%] Python app exited with code %EXIT_CODE%.>> "%LAUNCH_LOG%"
echo Python app exited with code %EXIT_CODE%.

if "%EXIT_CODE%"=="0" (
  if %RESTARTS% LSS 1 (
    set /a RESTARTS+=1
    echo The app stopped unexpectedly. Restarting once for diagnostics...
    echo [%date% %time%] Restarting once for diagnostics.>> "%LAUNCH_LOG%"
    timeout /t 2 /nobreak >nul
    goto run_app
  )
)

echo.
echo Local UI stopped.
echo Logs:
echo   %cd%\logs\live_caption_app.log
echo   %cd%\logs\live_caption_launcher.log
pause
