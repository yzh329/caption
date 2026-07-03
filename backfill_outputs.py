import argparse
import json
import re
import sys
from pathlib import Path

import requests

from live_caption_app import (
    OUTPUT_ROOT,
    refine_session_transcript,
)
from summarize import (
    DEFAULT_FINAL_MODEL,
    DEFAULT_MODEL,
    DEFAULT_TERMS_FILE,
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


LEGACY_OUTPUT_NAMES = ("meeting_notes.md", "meeting_review.md")


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def parse_live_transcript(transcript_path: Path):
    segments = []
    current_time = ""
    for raw_line in transcript_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        match = re.match(r"^##\s+(\d{2}:\d{2}:\d{2})\s*$", line)
        if match:
            current_time = match.group(1)
            continue
        if line.startswith("**English:**"):
            english = line.replace("**English:**", "", 1).strip()
            if english:
                segments.append({"timestamp": current_time, "english": english})
    return segments


def ensure_segments(session_dir: Path):
    segments_path = session_dir / "live_segments.jsonl"
    if segments_path.exists() and segments_path.stat().st_size > 0:
        return segments_path

    transcript_path = session_dir / "live_transcript.md"
    if not transcript_path.exists():
        raise FileNotFoundError(f"No live_segments.jsonl or live_transcript.md in {session_dir}")

    segments = parse_live_transcript(transcript_path)
    if not segments:
        raise ValueError(f"No English segments found in {transcript_path}")

    with segments_path.open("w", encoding="utf-8") as f:
        for item in segments:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return segments_path


def read_source_text(source_path: Path) -> str:
    if source_path.suffix.lower() != ".jsonl":
        return source_path.read_text(encoding="utf-8", errors="replace")

    parts = ["# Live Transcript", ""]
    count = 0
    for line in source_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        timestamp = str(item.get("timestamp", "")).strip()
        english = str(item.get("english", "")).strip()
        if not english:
            continue
        if timestamp:
            parts.extend([f"## {timestamp}", ""])
        parts.extend([f"**English:** {english}", ""])
        count += 1
    if count == 0:
        raise ValueError(f"No English segments found in {source_path}")
    return "\n".join(parts).strip() + "\n"


def remove_legacy_outputs(output_dir: Path) -> None:
    for name in LEGACY_OUTPUT_NAMES:
        legacy_path = output_dir / name
        if legacy_path.exists():
            legacy_path.unlink()
            print(f"[delete] {legacy_path.name}")


def format_ollama_http_error(exc: requests.HTTPError) -> str:
    response = exc.response
    if response is None:
        return f"[error] Ollama HTTP error: {exc}"
    detail = response.text.strip()
    hint = ""
    if response.status_code == 404:
        hint = " Hint: Ollama returned 404, which usually means one of the selected models is not installed. Run `ollama list`, or pull the missing model."
    if detail:
        return f"[error] Ollama HTTP error {response.status_code}: {detail}{hint}"
    return f"[error] Ollama HTTP error {response.status_code}: {exc}{hint}"


def extract_partials(
    transcript_text: str,
    source_name: str,
    model: str,
    host: str,
    max_chars: int,
    prompt_terms: str,
) -> list[str]:
    chunks = chunk_transcript_text(transcript_text, max_chars, source_name)
    print(f"[info] Extracting {len(chunks)} chunk(s) with Ollama model={model}")
    return [
        summarize_chunk_deep(host, model, chunk, i + 1, len(chunks), prompt_terms)
        for i, chunk in enumerate(chunks)
    ]


def write_summary_outputs(
    source_path: Path,
    output_dir: Path,
    model: str,
    final_model: str,
    host: str,
    force: bool,
    max_chars: int,
    prompt_terms: str,
    include_zh: bool,
):
    transcript_text = read_source_text(source_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    short_path = output_dir / "meeting_notes_short.md"
    deep_path = output_dir / "meeting_notes_deep.md"
    short_zh_path = output_dir / "meeting_notes_short_zh.md"
    deep_zh_path = output_dir / "meeting_notes_deep_zh.md"
    needs_short = force or not short_path.exists()
    needs_deep = force or not deep_path.exists()
    needs_short_zh = include_zh and (force or not short_zh_path.exists())
    needs_deep_zh = include_zh and (force or not deep_zh_path.exists())

    partials: list[str] | None = None
    merge_partials: list[str] | None = None
    coverage_map: str | None = None
    if needs_short or needs_deep or needs_short_zh or needs_deep_zh:
        partials = extract_partials(transcript_text, source_path.name, model, host, max_chars, prompt_terms)
        merge_partials = prepare_partials_for_final_merge(
            host,
            final_model,
            source_path.name,
            partials,
            prompt_terms,
        )
        print(f"[info] Building coverage map with Ollama model={final_model}")
        coverage_map = build_coverage_map(host, final_model, source_path.name, merge_partials, prompt_terms)

    if needs_short:
        print(f"[write] {short_path.name}")
        assert merge_partials is not None
        assert coverage_map is not None
        short_notes = final_summary_short(host, final_model, source_path.name, merge_partials, coverage_map, prompt_terms)
        validate_final_summary("short meeting summary", short_notes, coverage_map, is_deep=False)
        ensure_summary_grounded("short meeting summary", short_notes, transcript_text)
        short_path.write_text(short_notes + "\n", encoding="utf-8")
    else:
        print(f"[skip] {short_path.name} exists")

    if needs_deep:
        print(f"[write] {deep_path.name}")
        assert merge_partials is not None
        assert coverage_map is not None
        deep_notes = final_summary_deep(host, final_model, source_path.name, merge_partials, coverage_map, prompt_terms)
        validate_final_summary("detailed meeting review", deep_notes, coverage_map, is_deep=True)
        ensure_summary_grounded("detailed meeting review", deep_notes, transcript_text)
        deep_path.write_text(deep_notes + "\n", encoding="utf-8")
    else:
        print(f"[skip] {deep_path.name} exists")

    if include_zh:
        if needs_short_zh:
            print(f"[write] {short_zh_path.name}")
            assert merge_partials is not None
            assert coverage_map is not None
            short_notes_zh = final_summary_short_zh(host, final_model, source_path.name, merge_partials, coverage_map, prompt_terms)
            validate_final_summary("Chinese short meeting summary", short_notes_zh, coverage_map, is_deep=False)
            ensure_summary_grounded("Chinese short meeting summary", short_notes_zh, transcript_text)
            short_zh_path.write_text(short_notes_zh + "\n", encoding="utf-8")
        else:
            print(f"[skip] {short_zh_path.name} exists")

        if needs_deep_zh:
            print(f"[write] {deep_zh_path.name}")
            assert merge_partials is not None
            assert coverage_map is not None
            deep_notes_zh = final_summary_deep_zh(host, final_model, source_path.name, merge_partials, coverage_map, prompt_terms)
            validate_final_summary("Chinese detailed meeting review", deep_notes_zh, coverage_map, is_deep=True)
            ensure_summary_grounded("Chinese detailed meeting review", deep_notes_zh, transcript_text)
            deep_zh_path.write_text(deep_notes_zh + "\n", encoding="utf-8")
        else:
            print(f"[skip] {deep_zh_path.name} exists")

    remove_legacy_outputs(output_dir)


def process_source_file(
    source_file: Path,
    output_dir: Path | None,
    model: str,
    final_model: str,
    host: str,
    force_summary: bool,
    max_chars: int,
    prompt_terms: str,
    include_zh: bool,
):
    if not source_file.exists() or not source_file.is_file():
        raise FileNotFoundError(f"Source file not found: {source_file}")

    target_dir = output_dir or source_file.parent
    print(f"\n[file] {source_file}")
    print(f"[output] {target_dir}")
    print(f"[models] chunks={model} final={final_model}")
    write_summary_outputs(source_file, target_dir, model, final_model, host, force_summary, max_chars, prompt_terms, include_zh)


def process_session(
    session_dir: Path,
    model: str,
    final_model: str,
    host: str,
    force_refine: bool,
    force_summary: bool,
    max_chars: int,
    prompt_terms: str,
    include_zh: bool,
):
    session_id = session_dir.name
    print(f"\n[session] {session_id}")
    print(f"[models] chunks={model} final={final_model}")
    ensure_segments(session_dir)

    paragraph_path = session_dir / "paragraph_transcript.md"
    if force_refine or not paragraph_path.exists():
        print("[write] paragraph_transcript.md")
        paragraph_path = refine_session_transcript(session_dir, session_id, model, host)
    else:
        print("[skip] paragraph_transcript.md exists")

    source_path = paragraph_path if paragraph_path.exists() else session_dir / "live_transcript.md"
    print(f"[source] {source_path.name}")
    write_summary_outputs(source_path, session_dir, model, final_model, host, force_summary, max_chars, prompt_terms, include_zh)


def main():
    parser = argparse.ArgumentParser(description="Backfill paragraph transcript, short notes, and deep review.")
    parser.add_argument("--session", help="Specific session folder name under output/live")
    parser.add_argument("--all", action="store_true", help="Process all sessions under output/live")
    parser.add_argument("--file", help="Specific transcript/review markdown, text, or live_segments.jsonl file")
    parser.add_argument("--output-dir", help="Directory for short/deep meeting notes when using --file")
    parser.add_argument("--force", action="store_true", help="Regenerate paragraph transcript and short/deep summary files")
    parser.add_argument(
        "--redo-summary",
        action="store_true",
        help="Regenerate short/deep and Chinese short/deep notes without forcing paragraph transcript refinement",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model for paragraph refinement and chunk extraction")
    parser.add_argument(
        "--final-model",
        default=DEFAULT_FINAL_MODEL,
        help="Ollama model for final short/deep summaries, e.g. gpt-oss:20b, qwen3:30b, qwen3:14b",
    )
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host")
    parser.add_argument("--max-chars", type=int, default=8000, help="Chunk size for long meetings")
    parser.add_argument(
        "--terms-file",
        default=DEFAULT_TERMS_FILE,
        help="Markdown/text glossary with known names, product terms, spelling corrections, and translation preferences.",
    )
    parser.add_argument("--no-zh", action="store_true", help="Do not generate additional Chinese short/deep files")
    args = parser.parse_args()

    modes = [bool(args.all), bool(args.session), bool(args.file)]
    if sum(modes) != 1:
        parser.error("Use exactly one of --all, --session <folder-name>, or --file <path>")
    if args.output_dir and not args.file:
        parser.error("--output-dir can only be used with --file")

    force_refine = args.force
    force_summary = args.force or args.redo_summary or bool(args.file)
    terms_path = resolve_path(args.terms_file) if args.terms_file else None
    prompt_terms = read_prompt_terms(terms_path)
    include_zh = not args.no_zh
    if prompt_terms:
        print(f"[info] Prompt terms: {terms_path}")

    if args.file:
        source_file = resolve_path(args.file)
        output_dir = resolve_path(args.output_dir) if args.output_dir else None
        try:
            process_source_file(
                source_file,
                output_dir,
                args.model,
                args.final_model,
                args.host,
                force_summary,
                args.max_chars,
                prompt_terms,
                include_zh,
            )
        except requests.HTTPError as exc:
            print(format_ollama_http_error(exc), file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 1
        print("\n[done] Backfill complete.")
        return 0

    if args.session:
        sessions = [OUTPUT_ROOT / args.session]
    else:
        sessions = sorted([p for p in OUTPUT_ROOT.iterdir() if p.is_dir()])

    failed = 0
    for session_dir in sessions:
        try:
            process_session(
                session_dir,
                args.model,
                args.final_model,
                args.host,
                force_refine,
                force_summary,
                args.max_chars,
                prompt_terms,
                include_zh,
            )
        except requests.HTTPError as exc:
            failed += 1
            print(f"[error] {session_dir.name}: {format_ollama_http_error(exc)}", file=sys.stderr)
        except Exception as exc:
            failed += 1
            print(f"[error] {session_dir.name}: {exc}", file=sys.stderr)

    if failed:
        print(f"\n[done] Completed with {failed} failed session(s).", file=sys.stderr)
        return 1
    print("\n[done] Backfill complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
