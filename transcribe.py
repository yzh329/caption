import argparse
import importlib.util
import platform
import sys
from datetime import timedelta
from pathlib import Path

from faster_whisper import WhisperModel


def fmt_ts(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def load_model(model_name: str, device: str, compute_type: str, cpu_fallback: bool):
    try:
        return WhisperModel(model_name, device=device, compute_type=compute_type), device, compute_type
    except Exception as exc:
        if device != "cpu" and cpu_fallback:
            print(
                f"[warn] GPU load failed: {exc}\n"
                "[warn] Falling back to CPU with int8. This is slower but reliable.",
                file=sys.stderr,
            )
            return WhisperModel(model_name, device="cpu", compute_type="int8"), "cpu", "int8"
        raise


MLX_WHISPER_REPOS = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base",
    "small": "mlx-community/whisper-small",
    "medium": "mlx-community/whisper-medium",
    "large-v3": "mlx-community/whisper-large-v3",
    "distil-large-v3": "mlx-community/whisper-turbo",
}
MLX_FALLBACK_REPO = "mlx-community/whisper-turbo"


def mlx_whisper_repo(model_name: str, mlx_model: str = "") -> str:
    mlx_model = str(mlx_model or "").strip()
    if mlx_model:
        return mlx_model
    model_name = str(model_name or "").strip()
    return MLX_WHISPER_REPOS.get(model_name, model_name or MLX_WHISPER_REPOS["medium"])


def has_mlx_whisper() -> bool:
    return importlib.util.find_spec("mlx_whisper") is not None


def resolve_backend(requested: str) -> str:
    requested = str(requested or "auto").strip().lower().replace("-", "_")
    if requested == "auto":
        if sys.platform == "darwin" and platform.machine() == "arm64" and has_mlx_whisper():
            return "mlx_whisper"
        return "faster_whisper"
    if requested in {"mlx", "mlx_whisper"}:
        return "mlx_whisper"
    if requested in {"faster", "faster_whisper"}:
        return "faster_whisper"
    raise ValueError(f"Unknown backend: {requested}")


def transcribe_with_mlx(input_path: Path, model_name: str, language: str | None, mlx_model: str = ""):
    if not has_mlx_whisper():
        raise RuntimeError("mlx-whisper is not installed. On macOS run: .venv/bin/python -m pip install mlx-whisper")
    import mlx_whisper

    try:
        result = mlx_whisper.transcribe(
            str(input_path),
            path_or_hf_repo=mlx_whisper_repo(model_name, mlx_model),
            language=language,
            task="transcribe",
        )
    except Exception:
        if mlx_model:
            raise
        result = mlx_whisper.transcribe(
            str(input_path),
            path_or_hf_repo=MLX_FALLBACK_REPO,
            language=language,
            task="transcribe",
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe a meeting audio/video file to transcript.md."
    )
    parser.add_argument("input", help="Audio/video file, e.g. meeting.m4a, meeting.wav, meeting.mp4")
    parser.add_argument("-o", "--output", default="transcript.md", help="Markdown transcript output path")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "faster-whisper", "mlx-whisper"],
        help="ASR backend: auto uses mlx-whisper on Apple Silicon when installed, otherwise faster-whisper.",
    )
    parser.add_argument(
        "--mlx-model",
        default="",
        help="Optional MLX Whisper HF repo or local path. Empty means match --model when possible.",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model: tiny, base, small, medium, large-v3, distil-large-v3",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        choices=["gpu", "cuda", "cpu", "auto"],
        help="Use gpu for Mac Metal/NVIDIA CUDA intent, or cpu for fallback.",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="auto picks float16 on GPU/CUDA and int8 on CPU.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language hint: zh, en, etc. Leave empty for mixed Chinese/English auto-detect.",
    )
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--vad-filter", action="store_true", help="Filter long silence with Silero VAD.")
    parser.add_argument("--no-cpu-fallback", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.exists():
        print(f"[error] Input file not found: {input_path}", file=sys.stderr)
        return 2

    backend = resolve_backend(args.backend)
    device = "cuda" if args.device in {"gpu", "auto", "cuda"} else "cpu"
    compute_type = args.compute_type
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    elif device == "cpu" and compute_type == "float16":
        compute_type = "int8"

    print(f"[info] Backend={backend} model={args.model} device={device} compute_type={compute_type}")
    print(f"[info] Transcribing: {input_path}")
    rows = []
    full_text = []
    detected = "unknown"
    probability = 0.0
    duration_seconds = 0

    if backend == "mlx_whisper":
        result = transcribe_with_mlx(input_path, args.model, args.language, args.mlx_model)
        detected = result.get("language") or "unknown"
        for seg in result.get("segments") or []:
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            start = float(seg.get("start", 0) or 0)
            end = float(seg.get("end", start) or start)
            duration_seconds = max(duration_seconds, int(end))
            rows.append(f"- [{fmt_ts(start)} - {fmt_ts(end)}] {text}")
            full_text.append(text)
        if not full_text and str(result.get("text", "")).strip():
            full_text.append(str(result.get("text", "")).strip())
        used_device = "metal"
        used_compute_type = "mlx"
    else:
        model, used_device, used_compute_type = load_model(
            args.model, device, compute_type, cpu_fallback=not args.no_cpu_fallback
        )
        segments, info = model.transcribe(
            str(input_path),
            language=args.language,
            task="transcribe",
            beam_size=args.beam_size,
            vad_filter=args.vad_filter,
            condition_on_previous_text=True,
            word_timestamps=False,
        )
        detected = info.language or "unknown"
        probability = getattr(info, "language_probability", 0.0)
        duration_seconds = int(info.duration or 0)
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            rows.append(f"- [{fmt_ts(seg.start)} - {fmt_ts(seg.end)}] {text}")
            full_text.append(text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                "# Transcript",
                "",
                f"- Source: `{input_path.name}`",
                f"- Model: `{args.model}`",
                f"- Device: `{used_device}`",
                f"- Compute type: `{used_compute_type}`",
                f"- Detected language: `{detected}` ({probability:.2f})",
                f"- Duration: `{timedelta(seconds=duration_seconds)}`",
                "",
                "## Segments",
                "",
                *rows,
                "",
                "## Plain Text",
                "",
                "\n".join(full_text),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[ok] Wrote transcript: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
