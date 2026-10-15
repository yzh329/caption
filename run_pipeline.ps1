param(
    [Parameter(Mandatory = $true)]
    [string]$InputFile,

    [string]$OutDir = "output",
    [string]$AsrBackend = "auto",
    [string]$WhisperModel = "large-v3",
    [string]$WhisperDevice = "cuda",
    [string]$ComputeType = "float16",
    [string]$OllamaModel = "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-IQ3_S",
    [string]$FinalOllamaModel = "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-IQ3_S",
    [switch]$VadFilter
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $InputFile)) {
    throw "Input file not found: $InputFile"
}

if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir | Out-Null
}

$transcript = Join-Path $OutDir "transcript.md"
$shortNotes = Join-Path $OutDir "meeting_notes_short.md"
$deepNotes = Join-Path $OutDir "meeting_notes_deep.md"
$shortNotesZh = Join-Path $OutDir "meeting_notes_short_zh.md"
$deepNotesZh = Join-Path $OutDir "meeting_notes_deep_zh.md"

Write-Host "[1/3] Checking Python..."
python --version

Write-Host "[2/3] Transcribing..."
$transcribeArgs = @(
    "transcribe.py",
    $InputFile,
    "-o", $transcript,
    "--model", $WhisperModel,
    "--device", $WhisperDevice,
    "--compute-type", $ComputeType
)
if ($VadFilter) {
    $transcribeArgs += "--vad-filter"
}
python @transcribeArgs

Write-Host "[3/3] Summarizing with Ollama..."
python summarize.py $transcript --mode both --output-short $shortNotes --output-deep $deepNotes --output-short-zh $shortNotesZh --output-deep-zh $deepNotesZh --model $OllamaModel --final-model $FinalOllamaModel

Write-Host ""
Write-Host "[ok] Done"
Write-Host "Transcript: $transcript"
Write-Host "Short notes: $shortNotes"
Write-Host "Deep notes: $deepNotes"
Write-Host "Chinese short notes: $shortNotesZh"
Write-Host "Chinese deep notes: $deepNotesZh"
