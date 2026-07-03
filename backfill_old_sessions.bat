@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0"
title Backfill Meeting Outputs

set "PYTHON_EXE=python"
if exist "%USERPROFILE%\miniconda3\python.exe" (
  set "PYTHON_EXE=%USERPROFILE%\miniconda3\python.exe"
)
if exist ".venv\Scripts\python.exe" (
  set "PYTHON_EXE=%cd%\.venv\Scripts\python.exe"
)

echo Backfill or regenerate paragraph_transcript.md, short/deep notes, and Chinese short/deep notes.
echo Ollama must be installed and running.
echo.

echo Choose a mode:
echo   1. Fill missing outputs for all live sessions
echo   2. Redo short/deep summaries for all live sessions
echo   3. Redo short/deep summaries for one session folder
echo   4. Choose a transcript/review file and redo short/deep summaries
echo.
echo Optional prompt glossary: edit summary_terms.md before running.
echo Default models: chunks=gpt-oss:20b, final=gpt-oss:20b.
echo.
set /p "CHOICE=Enter choice [1-4]: "
echo.

if "%CHOICE%"=="1" (
  "%PYTHON_EXE%" backfill_outputs.py --all
  goto done
)

if "%CHOICE%"=="2" (
  "%PYTHON_EXE%" backfill_outputs.py --all --redo-summary
  goto done
)

if "%CHOICE%"=="3" (
  set /p "SESSION_NAME=Session folder name under output\live: "
  if not defined SESSION_NAME (
    echo No session entered.
    goto done
  )
  "%PYTHON_EXE%" backfill_outputs.py --session "!SESSION_NAME!" --redo-summary
  goto done
)

if "%CHOICE%"=="4" (
  set "SOURCE_FILE="
  for /f "usebackq delims=" %%F in (`powershell -NoProfile -Sta -ExecutionPolicy Bypass -Command "Add-Type -AssemblyName System.Windows.Forms; $dlg = New-Object System.Windows.Forms.OpenFileDialog; $start = Join-Path (Get-Location) 'output\live'; if (Test-Path $start) { $dlg.InitialDirectory = (Resolve-Path $start).Path } else { $dlg.InitialDirectory = (Get-Location).Path }; $dlg.Title = 'Choose transcript or review source file'; if ($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $dlg.FileName }"`) do set "SOURCE_FILE=%%F"
  if not defined SOURCE_FILE (
    echo No file selected.
    goto done
  )
  "%PYTHON_EXE%" backfill_outputs.py --file "!SOURCE_FILE!"
  goto done
)

echo Unknown choice: %CHOICE%

:done
echo.
pause
