import json
import atexit
import faulthandler
import logging
import re
import sys
import traceback
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime
from pathlib import Path
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

import requests
from flask import Flask, jsonify, render_template, request
from summarize import (
    DEFAULT_FINAL_MODEL,
    DEFAULT_MODEL,
    DEFAULT_TERMS_FILE,
    EmptyTranscriptError,
    build_coverage_map,
    chunk_transcript_text,
    final_summary_deep,
    final_summary_deep_zh,
    final_summary_short,
    final_summary_short_zh,
    ensure_summary_grounded,
    prepare_partials_for_final_merge,
    read_prompt_terms,
    summarize_chunk_deep,
    validate_final_summary,
)


APP_ROOT = Path(__file__).parent.resolve()
OUTPUT_ROOT = APP_ROOT / "output" / "live"
LOG_ROOT = APP_ROOT / "logs"

app = Flask(__name__)
events = deque(maxlen=1000)
event_id = 0
events_lock = threading.Lock()
model_cache = {}
model_lock = threading.Lock()
cuda_disabled_reason = None
translation_buffers = {}
translation_lock = threading.Lock()
transcript_write_lock = threading.Lock()
LEGACY_SUMMARY_NAMES = ("meeting_notes.md", "meeting_review.md")


class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True


def setup_logging():
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / "live_caption_app.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    faulthandler.enable()
    logging.info("Logging to %s", log_path)


def log_exit():
    logging.info("Python process exiting. active_threads=%s", [t.name for t in threading.enumerate()])


def publish(kind: str, **payload):
    global event_id
    event = {"kind": kind, "time": datetime.now().strftime("%H:%M:%S"), **payload}
    with events_lock:
        event_id += 1
        event["id"] = event_id
        events.append(event)
    logging.info("event kind=%s payload_keys=%s", kind, sorted(payload.keys()))


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.\u4e00-\u9fff]+", "_", name).strip("_")[:100] or "session"


def list_audio_sources():
    logging.info("Listing browser capture sources")
    return [
        {
            "id": "browser_mic",
            "type": "browser",
            "name": "Browser microphone",
            "label": "Microphone (browser permission popup)",
        },
        {
            "id": "browser_system",
            "type": "browser",
            "name": "Browser screen/system audio",
            "label": "System audio via Chrome screen sharing",
        },
    ]


def load_whisper(model_name: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    global cuda_disabled_reason
    if device != "cpu" and cuda_disabled_reason:
        publish("status", message=f"CUDA disabled for this session, using CPU int8: {cuda_disabled_reason}")
        device = "cpu"
        compute_type = "int8"

    key = (model_name, device, compute_type)
    with model_lock:
        if key in model_cache:
            return model_cache[key], device, compute_type
        try:
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
            model_cache[key] = model
            return model, device, compute_type
        except Exception as exc:
            if device != "cpu":
                publish("status", message=f"GPU load failed, falling back to CPU int8: {exc}")
                fallback_key = (model_name, "cpu", "int8")
                if fallback_key not in model_cache:
                    model_cache[fallback_key] = WhisperModel(model_name, device="cpu", compute_type="int8")
                return model_cache[fallback_key], "cpu", "int8"
            raise


def is_cuda_runtime_error(exc: Exception) -> bool:
    text = str(exc).lower()
    needles = [
        "cublas",
        "cudnn",
        "cuda",
        "cufft",
        "curand",
        "dll",
        "not found",
        "cannot be loaded",
        "no kernel image",
    ]
    return any(needle in text for needle in needles)


def transcribe_file_to_text(model, chunk_path: Path) -> str:
    segments, _ = model.transcribe(
        str(chunk_path),
        language="en",
        task="transcribe",
        beam_size=3,
        vad_filter=False,
        condition_on_previous_text=True,
    )
    return " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()


MLX_WHISPER_REPOS = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base",
    "small": "mlx-community/whisper-small",
    "medium": "mlx-community/whisper-medium",
    "large-v3": "mlx-community/whisper-large-v3",
    "distil-large-v3": "mlx-community/distil-whisper-large-v3",
}


def mlx_whisper_repo(model_name: str) -> str:
    model_name = str(model_name or "").strip()
    return MLX_WHISPER_REPOS.get(model_name, model_name or MLX_WHISPER_REPOS["medium"])


def has_mlx_whisper() -> bool:
    return importlib.util.find_spec("mlx_whisper") is not None


def resolve_asr_backend(requested: str) -> str:
    requested = str(requested or "auto").strip().lower().replace("-", "_")
    if requested == "auto":
        if sys.platform == "darwin" and platform.machine() == "arm64" and has_mlx_whisper():
            return "mlx_whisper"
        return "faster_whisper"
    if requested in {"mlx", "mlx_whisper"}:
        return "mlx_whisper"
    if requested in {"faster", "faster_whisper"}:
        return "faster_whisper"
    raise ValueError(f"Unknown ASR backend: {requested}")


def preload_asr_model(asr_backend: str, model_name: str, device: str, compute_type: str):
    used_backend = resolve_asr_backend(asr_backend)
    if used_backend == "mlx_whisper":
        if not has_mlx_whisper():
            raise RuntimeError("mlx-whisper is not installed. On macOS run: .venv/bin/python -m pip install mlx-whisper")
        repo = mlx_whisper_repo(model_name)
        try:
            from huggingface_hub import snapshot_download
        except Exception:
            publish("status", message=f"mlx-whisper ready; model downloads on first use: {repo}")
        else:
            publish("status", message=f"Downloading/checking mlx-whisper model: {repo}")
            snapshot_download(repo_id=repo)
            publish("status", message=f"mlx-whisper model ready: {repo}")
        return None, "metal", "mlx", used_backend

    model, used_device, used_compute = load_whisper(model_name, device, compute_type)
    return model, used_device, used_compute, used_backend


def transcribe_asr_file(asr_backend: str, model, model_name: str, chunk_path: Path) -> str:
    if asr_backend == "mlx_whisper":
        import mlx_whisper

        result = mlx_whisper.transcribe(
            str(chunk_path),
            path_or_hf_repo=mlx_whisper_repo(model_name),
            language="en",
            task="transcribe",
        )
        return str(result.get("text", "")).strip()
    return transcribe_file_to_text(model, chunk_path)


def format_ollama_http_error(exc: requests.HTTPError, ollama_model: str) -> str:
    response = exc.response
    if response is None:
        return f"Ollama HTTP error: {exc}"
    detail = response.text.strip()
    hint = ""
    if response.status_code == 404:
        hint = f" Model may be missing: run `ollama pull {ollama_model}` or choose one from `ollama list`."
    return f"Ollama HTTP {response.status_code}: {detail or exc}.{hint}"


def ollama_chat(ollama_model: str, ollama_host: str, prompt: str, timeout: int = 600) -> str:
    response = requests.post(
        ollama_host.rstrip("/") + "/api/chat",
        json={
            "model": ollama_model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.1},
        },
        timeout=timeout,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(format_ollama_http_error(exc, ollama_model)) from exc
    return response.json()["message"]["content"].strip()


def format_bytes(size: int | float | None) -> str:
    if not size:
        return "0 B"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


def selected_ollama_models(*models: str) -> list[str]:
    selected = []
    for model in models:
        model = str(model or "").strip()
        if model and model not in selected:
            selected.append(model)
    return selected


def pull_ollama_model(ollama_model: str, ollama_host: str):
    publish("status", message=f"Ollama pull/check started: {ollama_model}")
    response = requests.post(
        ollama_host.rstrip("/") + "/api/pull",
        json={"name": ollama_model, "stream": True},
        stream=True,
        timeout=600,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(format_ollama_http_error(exc, ollama_model)) from exc

    last_bucket = None
    last_status = ""
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        status = str(item.get("status", "")).strip()
        completed = item.get("completed")
        total = item.get("total")
        if completed and total:
            pct = min(100, int(float(completed) * 100 / float(total)))
            bucket = pct // 5
            if bucket != last_bucket or status != last_status:
                publish(
                    "status",
                    message=(
                        f"Ollama {ollama_model}: {status} {pct}% "
                        f"({format_bytes(completed)}/{format_bytes(total)})"
                    ),
                )
                last_bucket = bucket
                last_status = status
        elif status and status != last_status:
            publish("status", message=f"Ollama {ollama_model}: {status}")
            last_status = status
    publish("status", message=f"Ollama ready on disk: {ollama_model}")


def warm_ollama_model(ollama_model: str, ollama_host: str):
    publish("status", message=f"Warming Ollama model: {ollama_model}")
    ollama_chat(ollama_model, ollama_host, "Reply with exactly OK.", timeout=180)
    publish("status", message=f"Ollama ready in memory: {ollama_model}")


def parse_json_object(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def build_polished_paragraph(
    buffer: dict,
    end_time: str,
    ollama_model: str,
    ollama_host: str,
    include_chinese: bool,
) -> dict:
    joined = " ".join(item["english"] for item in buffer["items"]).strip()
    start_time = buffer["started_at"]
    raw_lines = "\n".join(f"[{item['timestamp']}] {item['english']}" for item in buffer["items"])
    chinese_instruction = (
        "Then translate that repaired English paragraph into natural Simplified Chinese."
        if include_chinese
        else "Set the chinese field to an empty string."
    )
    prompt = f"""You are cleaning a live ASR meeting transcript.
The input is split into short chunks, so words and sentences may be cut at boundaries.
Create one natural English paragraph that preserves the full meaning, repairs obvious boundary errors,
removes duplicated fragments, and does not invent facts.
{chinese_instruction}

Return only valid JSON with this shape:
{{"english": "repaired English paragraph", "chinese": "Simplified Chinese translation"}}

Live chunks:

{raw_lines}
"""
    content = ollama_chat(ollama_model, ollama_host, prompt, timeout=240)
    try:
        refined = parse_json_object(content)
        repaired_english = str(refined.get("english", "")).strip() or joined
        repaired_chinese = str(refined.get("chinese", "")).strip() if include_chinese else ""
    except Exception:
        logging.exception("Could not parse polished transcript JSON; using raw model output")
        repaired_english = joined
        repaired_chinese = content.strip() if include_chinese else ""
    return {
        "start_time": start_time,
        "end_time": end_time,
        "raw_english": joined,
        "english": repaired_english,
        "chinese": repaired_chinese,
    }


def append_polished_markdown(path: Path, polished: dict):
    with transcript_write_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(
                f"\n## Polished paragraph {polished['start_time']} - {polished['end_time']}\n\n"
                f"**English:** {polished['english']}\n\n"
                f"**Chinese:** {polished['chinese']}\n\n"
            )


def process_translation_buffer(
    session_id: str,
    transcript_path: Path,
    buffer: dict,
    end_time: str,
    ollama_model: str,
    ollama_host: str,
    include_chinese: bool,
):
    try:
        polished = build_polished_paragraph(buffer, end_time, ollama_model, ollama_host, include_chinese)
        append_polished_markdown(transcript_path, polished)
        publish("polished", polished=polished, transcript=str(transcript_path))
        logging.info(
            "Polished paragraph session=%s chars=%s model=%s",
            session_id,
            len(polished.get("english", "")),
            ollama_model,
        )
    except Exception as exc:
        logging.exception("Paragraph repair failed")
        publish("status", message=f"Paragraph repair unavailable: {exc}")


def queue_translation_buffer(
    session_id: str,
    timestamp: str,
    english: str,
    transcript_path: Path,
    ollama_model: str,
    ollama_host: str,
    include_chinese: bool = True,
):
    if not english.strip():
        return

    with translation_lock:
        buffer = translation_buffers.setdefault(session_id, {"items": [], "started_at": timestamp})
        if not buffer.get("started_at"):
            buffer["started_at"] = timestamp
        buffer["items"].append({"timestamp": timestamp, "english": english})
        joined = " ".join(item["english"] for item in buffer["items"]).strip()
        should_flush = len(buffer["items"]) >= 4 or len(joined) >= 620
        if not should_flush:
            return
        ready_buffer = {"items": buffer["items"], "started_at": buffer["started_at"]}
        translation_buffers[session_id] = {"items": [], "started_at": None}

    threading.Thread(
        target=process_translation_buffer,
        args=(session_id, transcript_path, ready_buffer, timestamp, ollama_model, ollama_host, include_chinese),
        daemon=True,
        name=f"translate-{session_id[:24]}",
    ).start()


def flush_translation_buffer(session_id: str, transcript_path: Path, ollama_model: str, ollama_host: str, include_chinese: bool):
    with translation_lock:
        buffer = translation_buffers.get(session_id)
        if not buffer or not buffer.get("items"):
            return
        ready_buffer = {"items": buffer["items"], "started_at": buffer["started_at"]}
        translation_buffers[session_id] = {"items": [], "started_at": None}

    end_time = ready_buffer["items"][-1]["timestamp"]
    worker = threading.Thread(
        target=process_translation_buffer,
        args=(session_id, transcript_path, ready_buffer, end_time, ollama_model, ollama_host, include_chinese),
        daemon=True,
        name=f"translate-final-{session_id[:18]}",
    )
    worker.start()


def generate_short_and_deep_summaries(
    transcript_text: str,
    source_name: str,
    ollama_model: str,
    final_ollama_model: str,
    ollama_host: str,
):
    chunks = chunk_transcript_text(transcript_text, 8000, source_name)
    terms_path = APP_ROOT / DEFAULT_TERMS_FILE
    prompt_terms = read_prompt_terms(terms_path)
    logging.info(
        "Generating short/deep summaries from %d chunk(s) model=%s final_model=%s",
        len(chunks),
        ollama_model,
        final_ollama_model,
    )
    if prompt_terms:
        logging.info("Using prompt terms: %s", terms_path)
    partials = [
        summarize_chunk_deep(ollama_host, ollama_model, chunk, i + 1, len(chunks), prompt_terms)
        for i, chunk in enumerate(chunks)
    ]
    merge_partials = prepare_partials_for_final_merge(
        ollama_host,
        final_ollama_model,
        source_name,
        partials,
        prompt_terms,
    )
    coverage_map = build_coverage_map(ollama_host, final_ollama_model, source_name, merge_partials, prompt_terms)
    short_notes = final_summary_short(ollama_host, final_ollama_model, source_name, merge_partials, coverage_map, prompt_terms)
    validate_final_summary("short meeting summary", short_notes, coverage_map, is_deep=False)
    ensure_summary_grounded("short meeting summary", short_notes, transcript_text)
    deep_notes = final_summary_deep(ollama_host, final_ollama_model, source_name, merge_partials, coverage_map, prompt_terms)
    validate_final_summary("detailed meeting review", deep_notes, coverage_map, is_deep=True)
    ensure_summary_grounded("detailed meeting review", deep_notes, transcript_text)
    short_notes_zh = final_summary_short_zh(
        ollama_host,
        final_ollama_model,
        source_name,
        merge_partials,
        coverage_map,
        prompt_terms,
    )
    validate_final_summary("Chinese short meeting summary", short_notes_zh, coverage_map, is_deep=False)
    ensure_summary_grounded("Chinese short meeting summary", short_notes_zh, transcript_text)
    deep_notes_zh = final_summary_deep_zh(
        ollama_host,
        final_ollama_model,
        source_name,
        merge_partials,
        coverage_map,
        prompt_terms,
    )
    validate_final_summary("Chinese detailed meeting review", deep_notes_zh, coverage_map, is_deep=True)
    ensure_summary_grounded("Chinese detailed meeting review", deep_notes_zh, transcript_text)
    return short_notes, deep_notes, short_notes_zh, deep_notes_zh


def remove_legacy_summary_outputs(session_dir: Path):
    for name in LEGACY_SUMMARY_NAMES:
        legacy_path = session_dir / name
        if legacy_path.exists():
            legacy_path.unlink()
            logging.info("Deleted legacy summary output: %s", legacy_path)


def append_live_segment(session_dir: Path, timestamp: str, english: str):
    if not english.strip():
        return
    segments_path = session_dir / "live_segments.jsonl"
    with segments_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": timestamp, "english": english}, ensure_ascii=False) + "\n")


def load_live_segments(session_dir: Path):
    segments_path = session_dir / "live_segments.jsonl"
    if not segments_path.exists():
        return []
    segments = []
    for line in segments_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        english = str(item.get("english", "")).strip()
        timestamp = str(item.get("timestamp", "")).strip()
        if english:
            segments.append({"timestamp": timestamp, "english": english})
    return segments


def group_segments_for_refinement(segments, max_chars: int = 2600, max_items: int = 18):
    groups = []
    current = []
    current_len = 0
    for item in segments:
        item_len = len(item["english"]) + 16
        if current and (current_len + item_len > max_chars or len(current) >= max_items):
            groups.append(current)
            current = []
            current_len = 0
        current.append(item)
        current_len += item_len
    if current:
        groups.append(current)
    return groups


def refine_segment_group(group, ollama_model: str, ollama_host: str) -> str:
    start_time = group[0]["timestamp"]
    end_time = group[-1]["timestamp"]
    raw_lines = "\n".join(f"[{item['timestamp']}] {item['english']}" for item in group)
    prompt = f"""You are cleaning a live ASR meeting transcript.
The input is split into short live chunks, so words and sentences may be cut at boundaries.
Repair obvious boundary problems, remove duplicate fragments, and organize the content into readable paragraphs.
Do not invent details. Preserve names, numbers, dates, product names, and technical terms.

Return Markdown only using this exact structure for each paragraph:

### {start_time} - {end_time}

**English:** repaired English paragraph

**Chinese:** natural Simplified Chinese translation

Live chunks:

{raw_lines}
"""
    return ollama_chat(ollama_model, ollama_host, prompt, timeout=600)


def refine_session_transcript(session_dir: Path, session_id: str, ollama_model: str, ollama_host: str) -> Path:
    segments = load_live_segments(session_dir)
    if not segments:
        raise ValueError("No live segments found to refine yet.")

    output_path = session_dir / "paragraph_transcript.md"
    parts = [
        "# Paragraph Transcript",
        "",
        f"- Session: `{session_id}`",
        "- Source: `live_segments.jsonl`",
        "",
    ]
    for group in group_segments_for_refinement(segments):
        parts.append(refine_segment_group(group, ollama_model, ollama_host))
        parts.append("")
    output_path.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")
    return output_path


def append_markdown(path: Path, timestamp: str, english: str, chinese: str):
    with transcript_write_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(f"\n## {timestamp}\n\n")
            if english:
                f.write(f"**English:** {english}\n\n")
            if chinese:
                f.write(f"**Chinese:** {chinese}\n\n")


@app.get("/")
def index():
    logging.info("GET /")
    return render_template("index.html")


@app.get("/api/sources")
def api_sources():
    sources = list_audio_sources()
    logging.info("Returning %d audio source(s)", len(sources))
    return jsonify(sources)


@app.post("/api/preload")
def api_preload():
    started = time.time()
    config = request.get_json(force=True)
    asr_backend = config.get("asr_backend", "auto")
    whisper_model = config.get("whisper_model", "small")
    device = config.get("device", "cpu")
    compute_type = config.get("compute_type", "int8")
    ollama_host = config.get("ollama_host", "http://localhost:11434")
    ollama_models = selected_ollama_models(
        config.get("ollama_model", DEFAULT_MODEL),
        config.get("summary_ollama_model", DEFAULT_MODEL),
        config.get("final_ollama_model", DEFAULT_FINAL_MODEL),
    )
    logging.info(
        "POST /api/preload backend=%s model=%s device=%s compute=%s ollama_models=%s",
        asr_backend,
        whisper_model,
        device,
        compute_type,
        ollama_models,
    )
    try:
        publish("status", message=f"Loading ASR: {asr_backend} / {whisper_model} ({device}, {compute_type})")
        _, used_device, used_compute, used_backend = preload_asr_model(
            asr_backend,
            whisper_model,
            device,
            compute_type,
        )
        for model in ollama_models:
            pull_ollama_model(model, ollama_host)
            warm_ollama_model(model, ollama_host)
        elapsed = round(time.time() - started, 2)
        publish("status", message=f"ASR ready: {used_backend} / {whisper_model} ({used_device}, {used_compute})")
        publish("status", message=f"Preload complete in {elapsed}s")
        return jsonify(
            {
                "ok": True,
                "asr_backend": used_backend,
                "model": whisper_model,
                "device": used_device,
                "compute_type": used_compute,
                "ollama_models": ollama_models,
                "elapsed_seconds": elapsed,
            }
        )
    except Exception as exc:
        logging.exception("Preload failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/summarize_session")
def api_summarize_session():
    config = request.get_json(force=True)
    session_id = safe_name(config.get("session_id", ""))
    ollama_model = config.get("ollama_model", DEFAULT_MODEL)
    final_ollama_model = config.get("final_ollama_model", DEFAULT_FINAL_MODEL)
    ollama_host = config.get("ollama_host", "http://localhost:11434")
    if not session_id:
        return jsonify({"ok": False, "error": "Missing session_id"}), 400

    session_dir = OUTPUT_ROOT / session_id
    transcript_path = session_dir / "live_transcript.md"
    refined_path = session_dir / "paragraph_transcript.md"
    output_path = session_dir / "meeting_notes_short.md"
    detailed_output_path = session_dir / "meeting_notes_deep.md"
    output_zh_path = session_dir / "meeting_notes_short_zh.md"
    detailed_zh_output_path = session_dir / "meeting_notes_deep_zh.md"
    if not transcript_path.exists() and not refined_path.exists():
        return jsonify({"ok": False, "error": f"Transcript not found: {transcript_path}"}), 404

    try:
        if not refined_path.exists():
            try:
                refined_path = refine_session_transcript(session_dir, session_id, ollama_model, ollama_host)
            except ValueError as exc:
                logging.info("Refinement before summary skipped: %s", exc)
            except Exception:
                logging.exception("Refinement before summary failed; falling back to live transcript")
        source_path = refined_path if refined_path.exists() else transcript_path
        transcript_text = source_path.read_text(encoding="utf-8")
        notes, detailed, notes_zh, detailed_zh = generate_short_and_deep_summaries(
            transcript_text, source_path.name, ollama_model, final_ollama_model, ollama_host
        )
        output_path.write_text(notes + "\n", encoding="utf-8")
        detailed_output_path.write_text(detailed + "\n", encoding="utf-8")
        output_zh_path.write_text(notes_zh + "\n", encoding="utf-8")
        detailed_zh_output_path.write_text(detailed_zh + "\n", encoding="utf-8")
        remove_legacy_summary_outputs(session_dir)
        logging.info("Short summary saved: %s", output_path)
        logging.info("Deep review saved: %s", detailed_output_path)
        logging.info("Chinese short summary saved: %s", output_zh_path)
        logging.info("Chinese deep review saved: %s", detailed_zh_output_path)
        return jsonify(
            {
                "ok": True,
                "output": str(output_path),
                "detailed_output": str(detailed_output_path),
                "output_zh": str(output_zh_path),
                "detailed_output_zh": str(detailed_zh_output_path),
                "source": str(source_path),
                "ollama_model": ollama_model,
                "final_ollama_model": final_ollama_model,
            }
        )
    except EmptyTranscriptError as exc:
        logging.info("Summary skipped: %s", exc)
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        logging.exception("Summary failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/refine_session")
def api_refine_session():
    config = request.get_json(force=True)
    session_id = safe_name(config.get("session_id", ""))
    ollama_model = config.get("ollama_model", DEFAULT_MODEL)
    ollama_host = config.get("ollama_host", "http://localhost:11434")
    if not session_id:
        return jsonify({"ok": False, "error": "Missing session_id"}), 400

    session_dir = OUTPUT_ROOT / session_id
    if not session_dir.exists():
        return jsonify({"ok": False, "error": f"Session not found: {session_dir}"}), 404

    try:
        output_path = refine_session_transcript(session_dir, session_id, ollama_model, ollama_host)
        logging.info("Paragraph transcript saved: %s", output_path)
        return jsonify({"ok": True, "output": str(output_path)})
    except Exception as exc:
        logging.exception("Paragraph refinement failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/flush_translation")
def api_flush_translation():
    config = request.get_json(force=True)
    session_id = safe_name(config.get("session_id", ""))
    ollama_model = config.get("ollama_model", DEFAULT_MODEL)
    ollama_host = config.get("ollama_host", "http://localhost:11434")
    include_chinese = bool(config.get("translate", True))
    if not session_id:
        return jsonify({"ok": False, "error": "Missing session_id"}), 400

    transcript_path = OUTPUT_ROOT / session_id / "live_transcript.md"
    if not transcript_path.exists():
        return jsonify({"ok": False, "error": f"Transcript not found: {transcript_path}"}), 404

    flush_translation_buffer(session_id, transcript_path, ollama_model, ollama_host, include_chinese)
    return jsonify({"ok": True})


@app.post("/api/transcribe_chunk")
def api_transcribe_chunk():
    started = time.time()
    audio = request.files.get("audio")
    if audio is None:
        return jsonify({"ok": False, "error": "Missing audio chunk"}), 400

    asr_backend = request.form.get("asr_backend", "auto")
    whisper_model = request.form.get("whisper_model", "medium")
    device = request.form.get("device", "cpu")
    compute_type = request.form.get("compute_type", "int8")
    translate = request.form.get("translate", "true") == "true"
    ollama_model = request.form.get("ollama_model", DEFAULT_MODEL)
    ollama_host = request.form.get("ollama_host", "http://localhost:11434")
    session_id = request.form.get("session_id") or datetime.now().strftime("%Y%m%d_%H%M%S")

    session_id = safe_name(session_id)
    session_dir = OUTPUT_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(audio.filename or "chunk.webm").suffix or ".webm"
    chunk_path = session_dir / f"chunk_{int(time.time() * 1000)}{suffix}"
    transcript_path = session_dir / "live_transcript.md"
    if not transcript_path.exists():
        transcript_path.write_text(f"# Live Transcript\n\n- Session: `{session_id}`\n\n", encoding="utf-8")
    audio.save(chunk_path)
    logging.info("Saved browser audio chunk: %s size=%s", chunk_path, chunk_path.stat().st_size)

    try:
        global cuda_disabled_reason
        model, used_device, used_compute, used_backend = preload_asr_model(asr_backend, whisper_model, device, compute_type)
        try:
            english = transcribe_asr_file(used_backend, model, whisper_model, chunk_path)
        except Exception as exc:
            if used_backend == "faster_whisper" and used_device != "cpu" and is_cuda_runtime_error(exc):
                cuda_disabled_reason = str(exc)
                logging.exception("CUDA transcription failed; retrying chunk on CPU int8")
                publish("status", message=f"CUDA failed, retrying on CPU int8: {exc}")
                model, used_device, used_compute, used_backend = preload_asr_model("faster_whisper", whisper_model, "cpu", "int8")
                english = transcribe_asr_file(used_backend, model, whisper_model, chunk_path)
            else:
                raise
        timestamp = datetime.now().strftime("%H:%M:%S")
        chinese = ""
        append_live_segment(session_dir, timestamp, english)
        if english:
            queue_translation_buffer(
                session_id,
                timestamp,
                english,
                transcript_path,
                ollama_model,
                ollama_host,
                include_chinese=translate,
            )
        if english or chinese:
            append_markdown(transcript_path, timestamp, english, chinese)
        logging.info(
            "Transcribed chunk session=%s chars=%s elapsed=%.2fs device=%s compute=%s",
            session_id,
            len(english),
            time.time() - started,
            used_device,
            used_compute,
        )
        return jsonify(
            {
                "ok": True,
                "timestamp": timestamp,
                "english": english,
                "chinese": chinese,
                "polished": None,
                "transcript": str(transcript_path),
                "device": used_device,
                "compute_type": used_compute,
            }
        )
    except Exception as exc:
        logging.exception("Failed to transcribe browser chunk")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.get("/api/events")
def api_events():
    try:
        since = int(request.args.get("since", "0"))
    except ValueError:
        since = 0
    with events_lock:
        latest = event_id
        selected = [event for event in events if event["id"] > since]
    logging.info("GET /api/events since=%s latest=%s count=%s", since, latest, len(selected))
    return jsonify({"latest": latest, "events": selected})


if __name__ == "__main__":
    setup_logging()
    atexit.register(log_exit)
    try:
        class QuietHandler(WSGIRequestHandler):
            def log_message(self, format, *args):
                print(f"{self.address_string()} - - [{self.log_date_time_string()}] {format % args}")

        def open_browser():
            time.sleep(1.5)
            logging.info("Opening browser")
            webbrowser.open("http://127.0.0.1:7860")

        threading.Thread(target=open_browser, daemon=True).start()
        logging.info("Serving local UI on http://127.0.0.1:7860")
        logging.info("Press CTRL+C or close this window to stop.")
        with make_server(
            "127.0.0.1",
            7860,
            app,
            server_class=ThreadingWSGIServer,
            handler_class=QuietHandler,
        ) as server:
            while True:
                logging.info("Server loop waiting for request")
                server.handle_request()
                logging.info("Server loop handled request")
    except KeyboardInterrupt:
        logging.info("Stopped by user")
    except Exception:
        logging.exception("Fatal server error")
        traceback.print_exc()
        input("Press Enter to close...")
