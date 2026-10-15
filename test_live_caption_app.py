import io
import re
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import requests

from live_caption_app import app, confirm_stream_words, format_ollama_http_error, mlx_whisper_repo, normalize_asr_runtime, resolve_asr_backend, selected_ollama_models, stream_states
from summarize import DEFAULT_FINAL_MODEL, DEFAULT_MODEL, transcript_signal_text


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


def test_qwen_translation_disables_thinking_without_changing_other_models():
    from live_caption_app import ollama_chat

    with patch("live_caption_app.requests.post") as post:
        post.return_value.json.return_value = {"message": {"content": "OK"}}
        for model in ("qwen3.5:9b-q4_K_M", "qwen3:8b", "qwen2.5:7b", "gpt-oss:20b"):
            assert ollama_chat(model, "http://localhost:11434", "test") == "OK"
            payload = post.call_args.kwargs["json"]
            assert payload["options"]["presence_penalty"] == 0
            if model.startswith("qwen3"):
                assert payload["think"] is False
            else:
                assert "think" not in payload
        assert b'qwen3.5:9b-q4_K_M' in app.test_client().get("/").data




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


def test_summary_models_match_ui_and_api_defaults():
    client = app.test_client()
    page = client.get("/").get_data(as_text=True)
    for select_id, model in (("summaryOllamaModel", DEFAULT_MODEL), ("finalOllamaModel", DEFAULT_FINAL_MODEL)):
        options = re.search(rf'<select id="{select_id}">(.*?)</select>', page, re.DOTALL).group(1)
        assert f'<option value="{model}" selected>' in options

    with TemporaryDirectory() as root:
        session = Path(root) / "summary-check"
        session.mkdir()
        (session / "paragraph_transcript.md").write_text("Source transcript", encoding="utf-8")
        with patch("live_caption_app.OUTPUT_ROOT", Path(root)), patch(
            "live_caption_app.generate_short_and_deep_summaries", return_value=("short", "deep", "short zh", "deep zh")
        ) as generate:
            for config, expected in (
                ({}, (DEFAULT_MODEL, DEFAULT_FINAL_MODEL)),
                ({"ollama_model": "qwen3:14b", "final_ollama_model": "gpt-oss:20b"}, ("qwen3:14b", "gpt-oss:20b")),
            ):
                response = client.post("/api/summarize_session", json={"session_id": session.name, **config})
                assert response.status_code == 200, response.get_json()
                assert generate.call_args.args[2:4] == expected


def test_stream_words_allow_corrections_and_keep_repeated_words():
    first = [{"word": " make", "start": 0., "end": 0.4}, {"word": " configure", "start": 0.4, "end": 1.}]
    corrected = [first[0], {"word": " configurable.", "start": 0.4, "end": 1.1}]
    confirmed, pending = confirm_stream_words(first, corrected, [], 3.)
    assert confirmed == first[:1] and pending == corrected[1:]
    yes = {"word": " Yes.", "start": 0., "end": 0.3}
    again = {"word": " Yes.", "start": 0.3, "end": 0.6}
    confirmed, pending = confirm_stream_words([again], [yes, again], [yes], 3.)
    assert confirmed == [again] and not pending
    # A repeated word at a different time is a new utterance, not agreement.
    later = {**again, "start": 4., "end": 4.3}
    assert confirm_stream_words([again], [later], [yes], 6.) == ([], [later])
    # A phrase anchor prevents an acronym from being appended twice when its timestamp moves.
    tail = [{"word": " from", "start": 3., "end": 3.2}, {"word": " the", "start": 3.2, "end": 3.3},
            {"word": " GECS.", "start": 3.3, "end": 4.}]
    following = {"word": " and", "start": 5., "end": 5.2}
    shifted = tail[:2] + [{"word": " GECS", "start": 4.1, "end": 5.}, following]
    assert confirm_stream_words([following], shifted, tail, 7.) == ([following], [])


def test_stream_api_commits_once_and_flushes_the_tail():
    def audio():
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(b"\0\0" * 48000)
        return io.BytesIO(output.getvalue())

    beginning = [{"word": " I", "start": 0., "end": 0.3}, {"word": " need", "start": 0.3, "end": 0.8}]
    first = beginning + [{"word": " configure", "start": 0.8, "end": 2.9}]
    corrected = beginning + [{"word": " configurable.", "start": 0.8, "end": 3.1}]
    client = app.test_client()
    with TemporaryDirectory() as root, patch("live_caption_app.OUTPUT_ROOT", Path(root)), patch.dict(stream_states, clear=True), patch(
        "live_caption_app.preload_asr_model", return_value=(None, "cpu", "int8", "faster_whisper")
    ), patch("live_caption_app.transcribe_audio_words", side_effect=[first, corrected]) as recognize, patch(
        "live_caption_app.queue_translation_buffer"
    ) as translate:
        def post(index, final=False, include_audio=True):
            data = {"session_id": "rolling-check", "chunk_index": str(index), "final": str(final).lower()}
            if include_audio:
                data["audio"] = (audio(), "chunk.wav")
            return client.post("/api/transcribe_stream", data=data)

        initial = post(0).get_json()
        assert initial["english"] == "" and initial["draft"] == "I need configure"
        assert not translate.called
        second = post(1).get_json()
        assert second["english"] == "I need" and second["draft"] == "configurable."
        assert post(1).get_json() == second  # A lost response can be retried without duplicate writes.
        assert recognize.call_count == 2 and translate.call_count == 1
        stream_states.clear()  # A server restart must restore the pending words and retry receipt from disk.
        assert post(1).get_json() == second
        final = post(2, final=True, include_audio=False).get_json()
        assert final["english"] == "configurable." and final["draft"] == ""
        assert post(2, final=True, include_audio=False).get_json() == final
        assert translate.call_count == 2
        saved = (Path(root) / "rolling-check/live_segments.jsonl").read_text(encoding="utf-8")
        assert saved.count('"english"') == 2 and "configure\"" not in saved
        assert len(list((Path(root) / "rolling-check").glob("audio_*.wav"))) == 2
        assert post(3, final=True, include_audio=False).status_code == 409
        invalid = client.post("/api/transcribe_stream", data={"session_id": "..", "chunk_index": "0", "final": "true"})
        assert invalid.status_code == 400
        missing = client.post("/api/transcribe_stream", data={"session_id": "new-session", "chunk_index": "1", "final": "true"})
        assert missing.status_code == 409


def test_failed_stream_inference_keeps_audio_and_can_retry():
    import numpy as np

    client = app.test_client()
    original_audio = io.BytesIO()
    with wave.open(original_audio, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 48000)
    original_audio = original_audio.getvalue()
    with TemporaryDirectory() as root, patch("live_caption_app.OUTPUT_ROOT", Path(root)), patch.dict(stream_states, clear=True), patch(
        "faster_whisper.audio.decode_audio", return_value=np.zeros(48000, np.float32)
    ), patch("live_caption_app.preload_asr_model", side_effect=[RuntimeError("temporary error"), (None, "cpu", "int8", "faster_whisper")]), patch(
        "live_caption_app.transcribe_audio_words", return_value=[]
    ) as recognize:
        for expected in (500, 200):
            response = client.post("/api/transcribe_stream", data={
                "session_id": "retry-check", "chunk_index": "0", "audio": (io.BytesIO(original_audio), "chunk.wav")
            })
            assert response.status_code == expected
            assert (Path(root) / "retry-check/audio_000000.wav").read_bytes() == original_audio
        assert len(recognize.call_args.args[3]) == 48000  # Failed attempts must not append the same audio twice.


if __name__ == "__main__":
    test_ollama_404_names_missing_model()
    test_empty_transcribe_metadata_is_not_content()
    test_selected_ollama_models_deduplicates_in_order()
    test_summary_models_match_ui_and_api_defaults()
    test_stream_words_allow_corrections_and_keep_repeated_words()
    test_stream_api_commits_once_and_flushes_the_tail()
    test_failed_stream_inference_keeps_audio_and_can_retry()
    print("ok")
