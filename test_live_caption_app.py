import requests

from live_caption_app import format_ollama_http_error, mlx_whisper_repo, normalize_asr_runtime, resolve_asr_backend, selected_ollama_models
from summarize import transcript_signal_text


def test_ollama_404_names_missing_model():
    response = requests.Response()
    response.status_code = 404
    response._content = b'{"error":"model not found"}'
    exc = requests.HTTPError(response=response)

    message = format_ollama_http_error(exc, "qwen3:8b")

    assert "Ollama HTTP 404" in message
    assert "ollama pull qwen3:8b" in message



def test_empty_transcribe_metadata_is_not_content():
    text = """# Transcript

- Source: `empty.wav`
- Model: `large-v3`
- Device: `cuda`
- Compute type: `float16`
- Detected language: `en` (0.50)
- Duration: `0:00:00`

## Segments

## Plain Text
"""

    assert transcript_signal_text(text) == ""


def test_selected_ollama_models_deduplicates_in_order():
    assert selected_ollama_models("qwen2.5:7b", "gpt-oss:20b", "gpt-oss:20b", "") == [
        "qwen2.5:7b",
        "gpt-oss:20b",
    ]




def test_asr_backend_aliases():
    assert resolve_asr_backend("faster-whisper") == "faster_whisper"
    assert resolve_asr_backend("mlx") == "mlx_whisper"


def test_mlx_whisper_repo_maps_common_names():
    assert mlx_whisper_repo("medium") == "mlx-community/whisper-medium"
    assert mlx_whisper_repo("medium", "mlx-community/custom") == "mlx-community/custom"


def test_normalize_asr_runtime_gpu_means_platform_gpu():
    assert normalize_asr_runtime("faster_whisper", "gpu", "auto") == ("cuda", "float16")
    assert normalize_asr_runtime("faster_whisper", "cpu", "auto") == ("cpu", "int8")
    assert normalize_asr_runtime("mlx_whisper", "gpu", "auto") == ("metal", "mlx")
if __name__ == "__main__":
    test_ollama_404_names_missing_model()
    test_empty_transcribe_metadata_is_not_content()
    test_selected_ollama_models_deduplicates_in_order()
    print("ok")
