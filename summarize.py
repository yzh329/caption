import argparse
import os
import re
import sys
from pathlib import Path

import requests


DEFAULT_TERMS_FILE = "summary_terms.md"
DEFAULT_MODEL = "gpt-oss:20b"
DEFAULT_FINAL_MODEL = "gpt-oss:20b"
MIN_TAIL_CHARS = 1200
FINAL_PARTIALS_CHAR_BUDGET = 30000
PARTIAL_REDUCTION_GROUP_CHARS = 18000
MAX_PARTIAL_REDUCTION_ROUNDS = 4


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


DEFAULT_OLLAMA_NUM_CTX = env_int("OLLAMA_NUM_CTX", 32768)

GROUNDING_STOPWORDS = {
    "about",
    "above",
    "action",
    "across",
    "again",
    "all",
    "almost",
    "also",
    "and",
    "another",
    "any",
    "are",
    "around",
    "ask",
    "asked",
    "available",
    "back",
    "been",
    "being",
    "best",
    "better",
    "because",
    "before",
    "but",
    "bye",
    "called",
    "can",
    "cannot",
    "check",
    "chunk",
    "chinese",
    "concise",
    "come",
    "comes",
    "comfortable",
    "completed",
    "confidence",
    "context",
    "could",
    "current",
    "currently",
    "data",
    "date",
    "decision",
    "decisions",
    "detail",
    "details",
    "did",
    "didn",
    "different",
    "discuss",
    "discussed",
    "direction",
    "does",
    "doesn",
    "doing",
    "done",
    "due",
    "english",
    "evidence",
    "every",
    "everyone",
    "exact",
    "excerpt",
    "excerpts",
    "final",
    "first",
    "for",
    "format",
    "from",
    "generate",
    "generated",
    "get",
    "going",
    "good",
    "got",
    "guess",
    "had",
    "has",
    "have",
    "haven",
    "having",
    "help",
    "helpful",
    "her",
    "here",
    "high",
    "him",
    "his",
    "how",
    "important",
    "include",
    "included",
    "includes",
    "including",
    "into",
    "isn",
    "items",
    "key",
    "know",
    "mentioned",
    "meeting",
    "more",
    "most",
    "need",
    "new",
    "next",
    "not",
    "notes",
    "now",
    "only",
    "once",
    "open",
    "other",
    "our",
    "output",
    "owner",
    "paragraph",
    "plan",
    "please",
    "pretty",
    "probably",
    "provide",
    "provided",
    "question",
    "questions",
    "quite",
    "ready",
    "review",
    "right",
    "risk",
    "risks",
    "said",
    "same",
    "section",
    "session",
    "share",
    "shared",
    "sharing",
    "she",
    "should",
    "since",
    "source",
    "structure",
    "summary",
    "summarize",
    "summarized",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "thing",
    "things",
    "think",
    "this",
    "timestamp",
    "topic",
    "transcript",
    "try",
    "unclear",
    "use",
    "used",
    "using",
    "want",
    "was",
    "way",
    "we",
    "well",
    "were",
    "what",
    "when",
    "where",
    "whether",
    "who",
    "why",
    "with",
    "without",
    "work",
    "working",
    "would",
    "yeah",
    "you",
}


SYSTEM_PROMPT = """You are a senior bilingual technical meeting analyst.

You receive a noisy bilingual transcript or extracted bilingual notes.
The source may contain English, Chinese translation, or mixed Chinese-English.

Your job is to create structured meeting notes, not a cleaned transcript.

Core rules:
- Merge English and Chinese translation into one understanding. Do not summarize English and Chinese separately.
- If English and Chinese conflict, prefer the original English unless the Chinese translation clearly fixes obvious ASR noise.
- Remove filler, repeated speech, greetings, small talk, false starts, and unclear fragments unless they carry business meaning.
- Preserve names, product names, company/customer names, numbers, dates, technical terms, APIs, tools, repositories, services, credentials, access steps, workflows, and concrete commitments.
- Treat unfamiliar company context as important, especially for a new employee.
- Flag uncertain ASR terms, especially product names and people's names.
- Suggest likely spellings only when supported by context.
- Do not invent missing information.
- If information is missing, write "Not mentioned".
- Follow the language requested by the specific output prompt. Preserve important English technical terms, names, product names, APIs, tools, and acronyms.
- If a name, tool, ID, system, project term, or ASR phrase is uncertain, keep it as [unclear: original phrase] until verified.
"""


def read_prompt_terms(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    lines = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def prompt_terms_block(prompt_terms: str) -> str:
    if not prompt_terms.strip():
        return ""
    return f"""Known names, product terms, spelling corrections, and translation preferences:

{prompt_terms.strip()}

Use this glossary/context to interpret ASR noise and translate names/terms consistently.
Do not mention this glossary as a separate output section unless it directly affects a decision, action item, risk, or discussion detail.
The glossary can confirm a term only when the transcript phrase reasonably matches it. If the transcript still sounds uncertain, keep [unclear: original phrase].
"""


class EmptyTranscriptError(ValueError):
    pass


MISSING_SOURCE_RESPONSE_RE = re.compile(
    r"\b(?:"
    r"i(?:'|\u2019)?m\s+(?:ready|happy)\s+to\s+help|"
    r"could\s+you\s+(?:please\s+)?(?:share|provide|send|paste)|"
    r"please\s+(?:share|provide|send|paste)|"
    r"once\s+i\s+have\s+(?:that|it|the)|"
    r"without\s+(?:the\s+)?(?:transcript|meeting\s+notes|coverage\s+map)|"
    r"need\s+(?:the\s+)?(?:transcript|meeting\s+notes|coverage\s+map)"
    r")\b",
    re.IGNORECASE,
)
TRANSCRIPT_LABEL_RE = re.compile(r"^\*\*(?:English(?: combined)?|Chinese):\*\*\s*", re.IGNORECASE)
TRANSCRIPT_HEADING_RE = re.compile(r"^#{1,6}\s*(?:Live Transcript|Paragraph Transcript)\s*$", re.IGNORECASE)
TRANSCRIPT_TIME_HEADING_RE = re.compile(
    r"^#{1,6}\s*(?:Polished paragraph\s+)?\d{1,2}:\d{2}:\d{2}(?:\s*-\s*\d{1,2}:\d{2}:\d{2})?\s*$",
    re.IGNORECASE,
)
TRANSCRIPT_METADATA_RE = re.compile(
    r"^-\s*(?:Session|Source|Model|Device|Compute type|Detected language|Duration):\s*`?.*?`?\s*$",
    re.IGNORECASE,
)


def transcript_signal_text(text: str) -> str:
    signal_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if TRANSCRIPT_HEADING_RE.match(line):
            continue
        if TRANSCRIPT_TIME_HEADING_RE.match(line):
            continue
        if TRANSCRIPT_METADATA_RE.match(line):
            continue
        if re.match(r"^#{1,6}\s+", line):
            continue
        line = TRANSCRIPT_LABEL_RE.sub("", line).strip()
        if not line or re.fullmatch(r"[-|:\s]+", line):
            continue
        if any(char.isalnum() for char in line):
            signal_lines.append(line)
    return "\n".join(signal_lines).strip()


def ensure_transcript_has_content(text: str, source_name: str) -> None:
    if not transcript_signal_text(text):
        raise EmptyTranscriptError(
            f"No transcribed speech found in {source_name}; skipping summary to avoid invented meeting notes."
        )


def ensure_summary_is_not_missing_source_request(label: str, summary: str) -> None:
    compact = re.sub(r"\s+", " ", summary).strip()
    if len(compact) <= 1200 and MISSING_SOURCE_RESPONSE_RE.search(compact):
        raise ValueError(
            f"{label} asks for the transcript or notes instead of summarizing the supplied source."
        )


def _is_markdown_section_heading(line: str) -> bool:
    return bool(re.match(r"^#{2,6}\s+\S", line.strip()))


def _clean_block(lines: list[str]) -> str:
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(line.rstrip() for line in lines).strip()


def _transcript_blocks(text: str) -> list[str]:
    lines = text.splitlines()
    blocks: list[str] = []
    current: list[str] = []

    for raw_line in lines:
        line = raw_line.rstrip()
        if _is_markdown_section_heading(line) and current:
            block = _clean_block(current)
            if block:
                blocks.append(block)
            current = [line]
        else:
            current.append(line)

    block = _clean_block(current)
    if block:
        blocks.append(block)

    if len(blocks) <= 1:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return paragraphs or ([text.strip()] if text.strip() else [])
    return blocks


def _split_oversized_block(block: str, max_chars: int) -> list[str]:
    if len(block) <= max_chars:
        return [block]

    units = [line.strip() for line in block.splitlines() if line.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for unit in units:
        if current and current_len + len(unit) + 1 > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(unit)
        current_len += len(unit) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def _merge_small_tail_chunks(chunks: list[str], max_chars: int) -> list[str]:
    if len(chunks) < 2:
        return chunks

    merged = chunks[:]
    tail_threshold = max(MIN_TAIL_CHARS, min(max_chars // 4, 2400))
    overflow_allowance = tail_threshold
    while len(merged) >= 2:
        tail = merged[-1]
        previous = merged[-2]
        tail_signal_len = len(transcript_signal_text(tail))
        combined = f"{previous}\n\n{tail}"
        if tail_signal_len >= tail_threshold:
            break
        if len(combined) > max_chars + overflow_allowance:
            break
        merged[-2:] = [combined]
    return merged


def chunk_text(text: str, max_chars: int) -> list[str]:
    blocks: list[str] = []
    for block in _transcript_blocks(text):
        blocks.extend(_split_oversized_block(block, max_chars))

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for block in blocks:
        separator_len = 2 if current else 0
        if current and current_len + len(block) + separator_len > max_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(block)
        current_len += len(block) + separator_len
    if current:
        chunks.append("\n\n".join(current))
    return _merge_small_tail_chunks(chunks, max_chars)


def chunk_transcript_text(text: str, max_chars: int, source_name: str) -> list[str]:
    ensure_transcript_has_content(text, source_name)
    chunks = [chunk for chunk in chunk_text(text, max_chars) if transcript_signal_text(chunk)]
    if not chunks:
        raise EmptyTranscriptError(
            f"No transcribed speech found in {source_name}; skipping summary to avoid invented meeting notes."
        )
    return chunks


def ollama_chat(host: str, model: str, messages: list[dict], temperature: float) -> str:
    url = host.rstrip("/") + "/api/chat"
    options = {"temperature": temperature}
    if DEFAULT_OLLAMA_NUM_CTX > 0:
        options["num_ctx"] = DEFAULT_OLLAMA_NUM_CTX
    response = requests.post(
        url,
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
        },
        timeout=600,
    )
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"].strip()


def _joined_partials(partials: list[str]) -> str:
    return "\n\n---\n\n".join(part.strip() for part in partials if part.strip())


def _grounding_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for term in re.findall(r"[A-Za-z][A-Za-z0-9_'-]{2,}", text.lower()):
        cleaned = term.strip("_'-")
        if len(cleaned) < 3 or cleaned in GROUNDING_STOPWORDS:
            continue
        if cleaned.isdigit():
            continue
        terms.add(cleaned)
    return terms


def ensure_summary_grounded(label: str, summary: str, source_text: str) -> None:
    ensure_summary_is_not_missing_source_request(label, summary)
    source_terms = _grounding_terms(transcript_signal_text(source_text))
    if len(source_terms) < 8:
        return

    summary_terms = _grounding_terms(summary)
    shared_terms = source_terms & summary_terms
    if len(shared_terms) >= 3:
        return

    sample_source_terms = ", ".join(sorted(source_terms)[:20])
    sample_summary_terms = ", ".join(sorted(summary_terms - source_terms)[:20])
    raise ValueError(
        f"{label} appears unrelated to the source transcript; shared source terms={len(shared_terms)}. "
        f"Source terms include: {sample_source_terms}. "
        f"Summary-only terms include: {sample_summary_terms}."
    )


def _pack_text_items(items: list[str], max_chars: int) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for item in [item.strip() for item in items if item.strip()]:
        separator_len = 7 if current else 0
        if current and current_len + len(item) + separator_len > max_chars:
            groups.append(current)
            current = []
            current_len = 0
            separator_len = 0
        current.append(item)
        current_len += len(item) + separator_len
    if current:
        groups.append(current)
    return groups


def summarize_partial_group(
    host: str,
    model: str,
    source_name: str,
    partial_group: list[str],
    index: int,
    total: int,
    prompt_terms: str = "",
) -> str:
    terms_block = prompt_terms_block(prompt_terms)
    joined = _joined_partials(partial_group)
    user_prompt = f"""Source transcript: {source_name}

You are given detailed extracted notes from one group of transcript chunks: group {index}/{total}.

{terms_block}

Partial notes in this group:

{joined}

Condense these partial notes for a later final merge.
This is not the final meeting summary. Preserve coverage and uncertainty while removing repeated rows.

Use this exact Markdown structure:

## Condensed Coverage Metadata
| Field | Value |
|---|---|
| Source | {source_name} |
| Group | {index}/{total} |
| Timestamp Range | First and last timestamp covered by this group, otherwise "Not mentioned". |
| Main Topics In This Group | List the main topics. |

## Confirmed Facts Only
- Preserve directly supported facts.
- If a fact depends on unclear ASR, mark the unclear part as [unclear: original phrase].
- If none, write "Not mentioned".

## Main Topic Notes
For each major topic:
### Topic: <topic name>
- What was discussed:
- Important details:
- Technical / business context:
- Evidence anchors: short transcript phrases with timestamps if available.

## Final Decisions
| Timestamp | Decision | Evidence Phrase | Confidence |
|---|---|---|---|

## Working Directions
| Timestamp | Direction / Tentative Plan | Evidence Phrase | Confidence |
|---|---|---|---|

## Action Items
| Timestamp | Owner | Action | Due Date | Evidence Phrase | Confidence |
|---|---|---|---|---|---|

## Open Questions / Blockers
- Preserve unresolved questions, dependencies, missing access, blockers, or unclear requirements.
- If none, write "Not mentioned".

## Concrete Risks
- Preserve concrete risks only when tied to source details.
- If none, write "Not mentioned".

## New Employee Context
- Preserve project-specific workflow, ownership, tools, and background.
- If none, write "Not mentioned".

## Important Technical Details
- Preserve APIs, services, database, auth, deployment, models, infra, codebase, configs, numbers, dates, or commands.

## Unclear ASR Terms
| Transcript Phrase | Possible Meaning | Why It Is Unclear | Suggested Follow-up |
|---|---|---|---|

Rules:
- Do not add facts that are not in the partial notes.
- Do not write polished final notes.
- Do not drop early or middle topics just because a later topic has clearer decisions.
- Keep decisions, working directions, action items, open questions, risks, technical details, timestamps, and unclear terms if present.
- Deduplicate repeated information, but preserve all distinct topics.
"""
    return ollama_chat(
        host,
        model,
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
        temperature=0.1,
    )


def prepare_partials_for_final_merge(
    host: str,
    model: str,
    source_name: str,
    partials: list[str],
    prompt_terms: str = "",
    max_chars: int = FINAL_PARTIALS_CHAR_BUDGET,
) -> list[str]:
    current = [partial.strip() for partial in partials if partial.strip()]
    rounds = 0
    while len(current) > 1 and len(_joined_partials(current)) > max_chars:
        rounds += 1
        if rounds > MAX_PARTIAL_REDUCTION_ROUNDS:
            break
        groups = _pack_text_items(current, PARTIAL_REDUCTION_GROUP_CHARS)
        if len(groups) >= len(current):
            break
        current = [
            summarize_partial_group(host, model, source_name, group, i + 1, len(groups), prompt_terms)
            for i, group in enumerate(groups)
        ]
    return current


def format_partials_debug(partials: list[str]) -> str:
    sections = []
    for index, partial in enumerate(partials, start=1):
        sections.append(f"# Chunk {index} Partial Notes\n\n{partial.strip()}")
    return "\n\n---\n\n".join(sections).strip()


def _normalize_coverage_and_terms(coverage_map: str, prompt_terms: str) -> tuple[str, str]:
    """Allow old callers that passed prompt_terms as the fifth positional argument."""
    if prompt_terms == "" and coverage_map and not coverage_map.lstrip().startswith("# Coverage Map"):
        return "", coverage_map
    return coverage_map, prompt_terms


def summarize_chunk_deep(
    host: str,
    model: str,
    chunk: str,
    index: int,
    total: int,
    prompt_terms: str = "",
) -> str:
    ensure_transcript_has_content(chunk, f"chunk {index}/{total}")
    terms_block = prompt_terms_block(prompt_terms)
    user_prompt = f"""Transcript chunk {index}/{total}:

{terms_block}

Raw transcript chunk:

{chunk}

Extract detailed useful information from this chunk only.
Do not write a polished final summary yet.
The output is evidence for a later merge, so preserve coverage and uncertainty.

Use this exact Markdown structure:

## Chunk Coverage Metadata
| Field | Value |
|---|---|
| Chunk | {index}/{total} |
| Timestamp Range | Write the first and last timestamp in this chunk if available; otherwise "Not mentioned". |
| Main Topics In This Chunk | List the main topics in this chunk only. |

## Confirmed Facts Only
- List facts that are directly supported by this chunk.
- If a fact depends on unclear ASR, mark the unclear part as [unclear: original phrase].
- If none, write "Not mentioned".

## Detailed Topic Notes
For each topic:
### Topic: <topic name>
- What was discussed:
- Important details:
- Technical / business context:
- Names, systems, tools, APIs, customers, repositories, or workflows mentioned:
- Why this may matter later:
- Evidence anchors: short transcript phrases with timestamps if available.

## Final Decisions
| Timestamp | Decision | Evidence Phrase | Confidence |
|---|---|---|---|

Rules:
- A final decision requires explicit agreement, approval, rejection, or instruction.
- Confidence must be High / Medium / Low.
- If the transcript was unclear or ASR may be wrong, use Low.
- If no final decisions appear in this chunk, write "Not mentioned".

## Working Directions
| Timestamp | Direction / Tentative Plan | Evidence Phrase | Confidence |
|---|---|---|---|

Rules:
- Use this for tentative direction, exploration, "try this", "prepare a draft", or proposed next steps.
- Do not promote a working direction into a final decision.
- Confidence must be High / Medium / Low.
- If none, write "Not mentioned".

## Action Items
| Timestamp | Owner | Action | Due Date | Evidence Phrase | Confidence |
|---|---|---|---|---|---|

Rules:
- Only include an action item when a concrete commitment appears in the transcript.
- Do not invent owner, due date, tool, system, authority, or scope.
- If owner is unclear, write "Unclear".
- If due date is missing, write "Not mentioned".
- Confidence must be High / Medium / Low.
- If an action uses [unclear: original phrase], confidence must be Low.

## Open Questions / Blockers
- Include explicit unresolved questions, dependencies, missing access, blockers, or unclear requirements.
- If none, write "Not mentioned".

## Concrete Risks
- Include only risks tied to a concrete transcript detail.
- Each risk must name the concrete cause and consequence.
- Generic risks like "data integrity is important" are forbidden.
- If none, write "Not mentioned".

## New Employee Context
Capture information that a new employee may need later:
- access process
- team workflow
- project background
- company-specific tools
- customer/product context
- who owns what
- where to ask for help

If none, write "Not mentioned".

## Important Technical Details
- APIs, services, database, auth, deployment, models, infra, codebase, configs, numbers, dates, or commands.
- Preserve only details that appear in this chunk.

## Unclear ASR Terms
| Transcript Phrase | Possible Meaning | Why It Is Unclear | Suggested Follow-up |
|---|---|---|---|

Rules:
- Include uncertain names, IDs, tools, systems, project terms, and ASR artifacts.
- If a name/tool/ID sounds uncertain, write [unclear: original phrase].
- Do not normalize uncertain words into confirmed terms.
- If no uncertainty, write "Not mentioned".

## Evidence Anchors
| Timestamp | Topic | Short Evidence Phrase |
|---|---|---|

Rules:
- Use short phrases copied from the transcript, not long quotes.
- Include timestamps when available.
- Do not convert speculation into fact.
- Do not invent authority, systems, final decisions, risks, or action owners.
- Preserve detail, but remove filler and repeated speech.
"""
    return ollama_chat(
        host,
        model,
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
        temperature=0.2,
    )


def build_coverage_map(
    host: str,
    model: str,
    source_name: str,
    partials: list[str],
    prompt_terms: str = "",
) -> str:
    if not partials:
        return """# Coverage Map

| Chunk | Timestamp Range | Main Topics | Must Appear in Final Notes? | Reason |
|---|---|---|---|---|

## Cross-Chunk Duplicates
- Not mentioned

## Potential Coverage Risks
- Not mentioned
"""
    terms_block = prompt_terms_block(prompt_terms)
    joined = _joined_partials(partials)
    user_prompt = f"""Source transcript: {source_name}

You are given detailed extracted notes from all transcript chunks.

{terms_block}

Chunk partial notes:

{joined}

Build an intermediate coverage map. This is not the final summary.
Its job is to help the final merge avoid missing early or middle chunks and avoid repeating the same action item under many topics.

Use this exact Markdown structure:

# Coverage Map

| Chunk | Timestamp Range | Main Topics | Must Appear in Final Notes? | Reason |
|---|---|---|---|---|

## Cross-Chunk Duplicates
- List duplicated topics, decisions, working directions, action items, risks, or unclear terms that should be merged.
- If none, write "Not mentioned".

## Potential Coverage Risks
- List topics that may be lost if the final summary over-compresses.
- Include topics that appeared in only one chunk but are still important because they contain a concrete decision, working direction, action item, blocker, risk, technical detail, access process, customer/product context, or new-employee context.
- If none, write "Not mentioned".

Rules:
- Do not write final meeting notes.
- Do not add facts that are not in the chunk partial notes.
- Deduplicate repeated action items and repeated topics by describing how they should be merged.
- Mark "Must Appear in Final Notes?" as Yes / Maybe / No.
- Use [unclear: original phrase] for uncertain names, tools, IDs, systems, or project terms.
"""
    return ollama_chat(
        host,
        model,
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
        temperature=0.1,
    )


def final_summary_short(
    host: str,
    model: str,
    source_name: str,
    partials: list[str],
    coverage_map: str = "",
    prompt_terms: str = "",
) -> str:
    coverage_map, prompt_terms = _normalize_coverage_and_terms(coverage_map, prompt_terms)
    joined = _joined_partials(partials)
    terms_block = prompt_terms_block(prompt_terms)
    coverage_text = coverage_map.strip() or "Not generated."
    user_prompt = f"""Source transcript: {source_name}

You are given an intermediate coverage map and detailed extracted notes from transcript chunks.

{terms_block}

Coverage map:

{coverage_text}

Partial notes:

{joined}

Merge and deduplicate them into a SHORT meeting summary for quick review.

Use this exact Markdown structure:

# Short Meeting Summary

## 1. High-Level Summary
Write 3-5 bullets total.
Each bullet should be one sentence.
Only include the most important discussion topics, outcomes, blockers, or concerns.

## 2. Key Decisions
| Timestamp | Decision | Evidence / Context | Confidence |
|---|---|---|---|

Rules:
- Include final decisions only when there is explicit agreement, approval, rejection, or instruction.
- If no final decisions were made, write "Not mentioned".
- Confidence must be High / Medium / Low.

## 3. Working Direction
| Timestamp | Direction / Tentative Plan | Evidence / Context | Confidence |
|---|---|---|---|

Rules:
- Include tentative plans separately from final decisions.
- If none, write "Not mentioned".

## 4. Action Items
| Timestamp | Owner | Action | Due Date | Evidence / Context | Confidence |
|---|---|---|---|---|---|

Rules for Action Items:
- Only include concrete commitments.
- Do not invent owner or due date.
- If owner is unclear, write "Unclear".
- If due date is not mentioned, write "Not mentioned".
- If an action uses [unclear: original phrase], confidence must be Low.

## 5. Open Questions / Risks
- List unresolved issues, blockers, dependencies, unclear requirements, or concrete risks.
- Every risk must name the concrete cause and consequence.
- Generic risks are forbidden.
- If none, write "Not mentioned".

Merge rules:
- Use the coverage map to ensure all major transcript sections are represented.
- Do not summarize only the final chunk.
- Deduplicate repeated items across chunks.
- If a topic is excluded from this short summary, exclude it because it is low importance, not because it was forgotten.
- Do not use unclear terms as confirmed terminology. Keep [unclear: original phrase] until verified.

Length constraint:
The entire output should be skimmable in under one minute.

Do not include:
- detailed timeline
- long topic-by-topic notes
- full background context
- repeated explanations
- cleaned transcript text
- a standalone terms-to-verify section
"""
    return ollama_chat(
        host,
        model,
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
        temperature=0.1,
    )


def final_summary_deep(
    host: str,
    model: str,
    source_name: str,
    partials: list[str],
    coverage_map: str = "",
    prompt_terms: str = "",
) -> str:
    coverage_map, prompt_terms = _normalize_coverage_and_terms(coverage_map, prompt_terms)
    joined = _joined_partials(partials)
    terms_block = prompt_terms_block(prompt_terms)
    coverage_text = coverage_map.strip() or "Not generated."
    user_prompt = f"""Source transcript: {source_name}

You are given an intermediate coverage map and detailed extracted notes from multiple transcript chunks.

{terms_block}

Coverage map:

{coverage_text}

Partial notes:

{joined}

Merge, deduplicate, and reorganize them into a detailed meeting review document.

Use this exact Markdown structure:

# Detailed Meeting Review

## 1. Executive Overview
Write 5-8 bullets summarizing the whole meeting.
Focus on the major topics, direction, outcomes, and unresolved issues.

## 2. Main Discussion Topics

For each major topic from the coverage map, use this format:

### Topic: <topic name>

#### Summary
Explain the topic in 2-4 sentences.

#### Details Discussed
- Preserve specific points, examples, constraints, numbers, tools, APIs, systems, customers, or workflows.
- Include relevant background if it helps future review.
- Do not use unclear terms as confirmed terminology.

#### Final Decisions
- List final decisions related to this topic only when there is explicit agreement, approval, rejection, or instruction.
- If none, write "Not mentioned".

#### Working Direction
- List tentative plans, exploration paths, drafts to prepare, things to try, or non-final direction.
- Do not promote working direction into a final decision.
- If none, write "Not mentioned".

#### Concerns / Trade-offs / Risks
- List concerns, alternatives, risks, blockers, or limitations.
- Every risk must name the concrete cause and consequence.
- Generic risks are forbidden.
- If none, write "Not mentioned".

#### Follow-up Needed
- List follow-up questions, validation tasks, or things to check.
- If none, write "Not mentioned".

## 3. Decisions
| Timestamp | Decision | Evidence / Context | Confidence |
|---|---|---|---|

Rules:
- Decisions and Working Direction are separate concepts.
- A final decision requires explicit agreement, approval, rejection, or instruction.
- Confidence must be High / Medium / Low.
- Use Low if the transcript was unclear, ASR may be wrong, or the decision uses [unclear: original phrase].
- If no final decisions were made, write "Not mentioned".

## 4. Working Direction
| Timestamp | Direction / Tentative Plan | Evidence / Context | Confidence |
|---|---|---|---|

Rules:
- A working direction can include tentative plans like "try this today", "prepare a draft", or "look into this".
- Do not represent working direction as final approval.
- Confidence must be High / Medium / Low.
- If none, write "Not mentioned".

## 5. Action Items
| Timestamp | Owner | Action | Due Date | Evidence / Context | Confidence |
|---|---|---|---|---|---|

Rules:
- Only include concrete commitments.
- Every action item must have evidence/context.
- Do not invent owners, dates, authority, systems, databases, tools, or scope.
- If owner is unclear, write "Unclear".
- If due date is missing, write "Not mentioned".
- If the action uses [unclear: original phrase], confidence must be Low.

## 6. Open Questions
- List unresolved questions.
- Include questions the team explicitly raised and questions implied by missing information.
- Mark inferred questions as "(inferred)".
- If none, write "Not mentioned".

## 7. Risks / Blockers / Dependencies
- Include technical risks, process blockers, access dependencies, unclear ownership, missing requirements, or external dependencies only when tied to concrete transcript detail.
- Every risk must name the concrete cause and consequence.
- Generic risks are forbidden.
- If none, write "Not mentioned".

## 8. Technical / Operational Details
Group details by category if possible:
- APIs / Services
- Data / Database
- Authentication / Access
- Infrastructure / Deployment
- Tools / Workflow
- Models / Algorithms
- Repositories / Codebase
- Metrics / Numbers / Dates

Only include categories that appear in the notes.

## 9. Context for New Employee
Explain company-specific or project-specific context that may help a new employee:
- who to ask
- what tools are used
- how workflow seems to operate
- important background
- terminology
- dependencies

Do not invent. If not mentioned, write "Not mentioned".

## 10. Chronological Timeline
If timestamps exist, summarize the meeting flow chronologically.
If timestamps do not exist, reconstruct only a rough topic order.
Use concise bullets.

## Unclear Terms to Verify
| Transcript Phrase | Possible Meaning | Why It Is Unclear | Suggested Follow-up |
|---|---|---|---|

Rules:
- Include uncertain names, IDs, tools, systems, project terms, and ASR artifacts.
- Do not use unclear terms as confirmed terminology elsewhere.
- If an unclear term is used in a decision/action item, confidence must be Low.
- If no unclear terms appear, write "Not mentioned".

## 12. What I Should Review Next
List concrete things the user should check after the meeting:
- documents
- tools
- access
- people to ask
- technical concepts
- unclear terms
- project background

Merge rules:
- Use the coverage map to ensure all major transcript sections are represented.
- Do not summarize only the final chunk.
- Deep summary should cover all major topics from the coverage map.
- Deduplicate repeated items across chunks.
- Preserve useful details even if they seem minor.
- Do not over-compress.
- Do not convert speculation into fact.
"""
    return ollama_chat(
        host,
        model,
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
        temperature=0.15,
    )


def final_summary_short_zh(
    host: str,
    model: str,
    source_name: str,
    partials: list[str],
    coverage_map: str = "",
    prompt_terms: str = "",
) -> str:
    coverage_map, prompt_terms = _normalize_coverage_and_terms(coverage_map, prompt_terms)
    joined = _joined_partials(partials)
    terms_block = prompt_terms_block(prompt_terms)
    coverage_text = coverage_map.strip() or "Not generated."
    user_prompt = f"""Source transcript: {source_name}

You are given an intermediate coverage map and detailed extracted notes from transcript chunks.
The source data is provided below. Do not ask the user to provide a transcript.

{terms_block}

Task:
Merge and deduplicate the source data into a short Simplified Chinese meeting summary for quick review.
Write the full response in Simplified Chinese, while preserving important English names, product names,
APIs, tools, repositories, services, database fields, and acronyms exactly as English.

Use this Markdown structure, translated naturally into Simplified Chinese:

# Chinese Short Meeting Summary

## 1. High-Level Summary
- Write 3-5 concise bullets.

## 2. Key Decisions
| Timestamp | Decision | Evidence / Context | Confidence |
|---|---|---|---|

## 3. Working Direction
| Timestamp | Direction / Tentative Plan | Evidence / Context | Confidence |
|---|---|---|---|

## 4. Action Items
| Timestamp | Owner | Action | Due Date | Evidence / Context | Confidence |
|---|---|---|---|---|---|

## 5. Open Questions / Risks
- List unresolved questions, blockers, dependencies, unclear requirements, or concrete risks.

Rules:
- Include final decisions only when there is explicit agreement, approval, rejection, or instruction.
- Keep tentative plans separate from final decisions.
- Include only concrete commitments as action items.
- Do not invent owner, due date, authority, tools, systems, databases, or scope.
- If information is missing, write "Not mentioned".
- Every risk must be tied to a concrete cause and consequence in the source data.
- Use the coverage map to avoid losing early or middle topics.
- Keep [unclear: original phrase] for uncertain ASR terms.
- Do not include a cleaned transcript or a standalone glossary section.

Coverage map:

{coverage_text}

Partial notes:

{joined}

Now write the short Simplified Chinese meeting summary using only the source data above.
"""
    return ollama_chat(
        host,
        model,
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
        temperature=0.1,
    )


def final_summary_deep_zh(
    host: str,
    model: str,
    source_name: str,
    partials: list[str],
    coverage_map: str = "",
    prompt_terms: str = "",
) -> str:
    coverage_map, prompt_terms = _normalize_coverage_and_terms(coverage_map, prompt_terms)
    joined = _joined_partials(partials)
    terms_block = prompt_terms_block(prompt_terms)
    coverage_text = coverage_map.strip() or "Not generated."
    user_prompt = f"""Source transcript: {source_name}

You are given an intermediate coverage map and detailed extracted notes from transcript chunks.
The source data is provided below. Do not ask the user to provide a transcript or coverage map.

{terms_block}

Task:
Merge, deduplicate, and reorganize the source data into a detailed Simplified Chinese meeting review.
Write the full response in Simplified Chinese, while preserving important English names, product names,
APIs, tools, repositories, services, database fields, and acronyms exactly as English.

Use this Markdown structure, translated naturally into Simplified Chinese:

# Chinese Detailed Meeting Review

## 1. Executive Overview
- Write 5-8 bullets covering the whole meeting.

## 2. Main Discussion Topics
For each major topic from the coverage map, include:
### Topic: <topic name>
#### Summary
#### Details Discussed
#### Final Decisions
#### Working Direction
#### Concerns / Trade-offs / Risks
#### Follow-up Needed

## 3. Decisions
| Timestamp | Decision | Evidence / Context | Confidence |
|---|---|---|---|

## 4. Working Direction
| Timestamp | Direction / Tentative Plan | Evidence / Context | Confidence |
|---|---|---|---|

## 5. Action Items
| Timestamp | Owner | Action | Due Date | Evidence / Context | Confidence |
|---|---|---|---|---|---|

## 6. Open Questions

## 7. Risks / Blockers / Dependencies

## 8. Technical / Operational Details

## 9. Context for New Employee

## 10. Chronological Timeline

## 11. Unclear Terms to Verify
| Transcript Phrase | Possible Meaning | Why It Is Unclear | Suggested Follow-up |
|---|---|---|---|

## 12. What To Review Next

Rules:
- Do not invent facts, owners, dates, systems, authority, or scope.
- Keep final decisions and tentative working directions separate.
- Every action item must have evidence/context from the source data.
- Every risk must name the concrete cause and consequence.
- Use the coverage map to ensure all major transcript sections are represented.
- Do not summarize only the final chunk.
- Preserve useful technical and operational details.
- Keep [unclear: original phrase] for uncertain ASR terms.
- If information is missing, write "Not mentioned".

Coverage map:

{coverage_text}

Partial notes:

{joined}

Now write the detailed Simplified Chinese meeting review using only the source data above.
"""
    return ollama_chat(
        host,
        model,
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
        temperature=0.15,
    )


def _section_by_heading(text: str, heading_terms: list[str]) -> str:
    lines = text.splitlines()
    captured: list[str] = []
    in_section = False
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if stripped.startswith("#"):
            if in_section and stripped.startswith("##"):
                break
            if any(term.lower() in lower for term in heading_terms):
                in_section = True
                captured.append(line)
                continue
        elif in_section:
            captured.append(line)
    return "\n".join(captured).strip()


def _extract_table_rows(section: str) -> list[str]:
    rows: list[str] = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or stripped.count("|") < 2:
            continue
        compact = re.sub(r"\s+", " ", stripped.strip("|")).strip().lower()
        divider = compact.replace("|", "").replace("-", "").replace(":", "").strip()
        if not divider:
            continue
        if "owner" in compact and "action" in compact:
            continue
        if "decision" in compact and "confidence" in compact:
            continue
        if "timestamp" in compact and "confidence" in compact:
            continue
        if "时间戳" in compact and "置信度" in compact:
            continue
        if "not mentioned" in compact and len(compact) < 40:
            continue
        rows.append(compact)
    return rows


def _extract_unclear_phrases(*texts: str) -> list[str]:
    phrases: list[str] = []
    for text in texts:
        for phrase in re.findall(r"\[unclear:\s*([^\]]+?)\]", text, flags=re.IGNORECASE):
            cleaned = re.sub(r"\s+", " ", phrase).strip(" .,:;，。；：")
            if len(cleaned) >= 3 and cleaned.lower() not in {p.lower() for p in phrases}:
                phrases.append(cleaned)
    return phrases


def _extract_coverage_topics(coverage_map: str) -> list[str]:
    topics: list[str] = []
    for line in coverage_map.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or stripped.count("|") < 5:
            continue
        if "Main Topics" in stripped or re.fullmatch(r"\|?\s*[-:\s|]+\s*\|?", stripped):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 3:
            continue
        for topic in re.split(r",|;|/|、|；|\band\b", cells[2]):
            cleaned = re.sub(r"\s+", " ", topic).strip(" .,:;，。；：")
            if cleaned and cleaned.lower() not in {"not mentioned", "none", "n/a"}:
                if cleaned.lower() not in {item.lower() for item in topics}:
                    topics.append(cleaned)
    return topics[:12]


def validate_final_summary(label: str, summary: str, coverage_map: str, is_deep: bool) -> None:
    warnings: list[str] = []

    action_section = _section_by_heading(summary, ["action items", "行动项"])
    action_rows = _extract_table_rows(action_section)
    seen_rows: set[str] = set()
    duplicates: set[str] = set()
    for row in action_rows:
        if row in seen_rows:
            duplicates.add(row)
        seen_rows.add(row)
    if duplicates:
        warnings.append(f"repeated identical action item row(s): {len(duplicates)}")

    decision_section = _section_by_heading(summary, ["decisions", "关键决策", "决策"])
    if "not mentioned" in decision_section.lower():
        coverage_lower = coverage_map.lower()
        direction_terms = ["working direction", "tentative", "try ", "prepare a draft", "direction / tentative"]
        if any(term in coverage_lower for term in direction_terms):
            warnings.append(
                'decisions say "Not mentioned" while the coverage map shows possible working directions; verify the separation is intentional'
            )

    unclear_phrases = _extract_unclear_phrases(coverage_map, summary)
    summary_without_markers = re.sub(r"\[unclear:\s*[^\]]+?\]", "", summary, flags=re.IGNORECASE)
    for phrase in unclear_phrases:
        if re.search(re.escape(phrase), summary_without_markers, flags=re.IGNORECASE):
            warnings.append(f"unclear term may be used as confirmed terminology: {phrase}")
            break

    if is_deep and len(summary.strip()) < 800:
        warnings.append("deep summary is empty or very short")

    coverage_topics = _extract_coverage_topics(coverage_map)
    if len(coverage_topics) > 1:
        summary_lower = summary.lower()
        mentioned = [topic for topic in coverage_topics if topic.lower() in summary_lower]
        if len(mentioned) <= 1:
            warnings.append(
                f"summary may mention only one coverage-map topic despite {len(coverage_topics)} major topics"
            )

    for warning in warnings:
        print(f"[warn] {label}: {warning}", file=sys.stderr)


def resolve_output_path(path_text: str) -> Path:
    return Path(path_text).expanduser().resolve()


def write_notes(path: Path, content: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    print(f"[ok] Wrote {label}: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize transcript.md into short and/or deep meeting notes using local Ollama."
    )
    parser.add_argument("transcript", help="Transcript markdown file")
    parser.add_argument("--mode", choices=["short", "deep", "both"], default="both")
    parser.add_argument("--output-short", default="meeting_notes_short.md")
    parser.add_argument("--output-deep", default="meeting_notes_deep.md")
    parser.add_argument("--output-short-zh", default="meeting_notes_short_zh.md")
    parser.add_argument("--output-deep-zh", default="meeting_notes_deep_zh.md")
    parser.add_argument(
        "--output-partials",
        default=None,
        help="Optional debug path for chunk partial extraction notes.",
    )
    parser.add_argument(
        "--output-coverage",
        default=None,
        help="Optional debug path for the intermediate coverage map.",
    )
    parser.add_argument(
        "--no-zh",
        action="store_true",
        help="Do not generate additional Chinese short/deep files.",
    )
    parser.add_argument(
        "--terms-file",
        default=DEFAULT_TERMS_FILE,
        help="Markdown/text glossary with known names, product terms, spelling corrections, and translation preferences.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Backward-compatible output path for --mode short or --mode deep only.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model for chunk extraction/refinement")
    parser.add_argument(
        "--final-model",
        default=DEFAULT_FINAL_MODEL,
        help="Ollama model for final merged summaries, e.g. gpt-oss:20b, qwen3:30b, qwen3:14b",
    )
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--max-chars", type=int, default=8000, help="Chunk size for long meetings")
    args = parser.parse_args()

    if args.output and args.mode == "both":
        parser.error("--output can only be used with --mode short or --mode deep")

    transcript_path = Path(args.transcript).expanduser().resolve()
    if not transcript_path.exists():
        print(f"[error] Transcript not found: {transcript_path}", file=sys.stderr)
        return 2

    short_output_path = resolve_output_path(args.output if args.output and args.mode == "short" else args.output_short)
    deep_output_path = resolve_output_path(args.output if args.output and args.mode == "deep" else args.output_deep)
    short_zh_output_path = resolve_output_path(args.output_short_zh)
    deep_zh_output_path = resolve_output_path(args.output_deep_zh)
    partials_output_path = resolve_output_path(args.output_partials) if args.output_partials else None
    coverage_output_path = resolve_output_path(args.output_coverage) if args.output_coverage else None
    terms_path = Path(args.terms_file).expanduser().resolve() if args.terms_file else None
    prompt_terms = read_prompt_terms(terms_path)

    text = transcript_path.read_text(encoding="utf-8")
    try:
        chunks = chunk_transcript_text(text, args.max_chars, transcript_path.name)
    except EmptyTranscriptError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    print(
        f"[info] Summarizing {len(chunks)} chunk(s) with Ollama model={args.model} "
        f"final_model={args.final_model} mode={args.mode}"
    )
    if prompt_terms:
        print(f"[info] Prompt terms: {terms_path}")
    if partials_output_path:
        print(f"[info] Partial debug output: {partials_output_path}")
    if coverage_output_path:
        print(f"[info] Coverage debug output: {coverage_output_path}")
    if args.mode in ("short", "both"):
        print(f"[info] Short output: {short_output_path}")
        if not args.no_zh:
            print(f"[info] Short Chinese output: {short_zh_output_path}")
    if args.mode in ("deep", "both"):
        print(f"[info] Deep output: {deep_output_path}")
        if not args.no_zh:
            print(f"[info] Deep Chinese output: {deep_zh_output_path}")

    try:
        partials = [
            summarize_chunk_deep(args.host, args.model, chunk, i + 1, len(chunks), prompt_terms)
            for i, chunk in enumerate(chunks)
        ]
        if partials_output_path:
            write_notes(partials_output_path, format_partials_debug(partials), "chunk partial notes")

        merge_partials = prepare_partials_for_final_merge(
            args.host,
            args.final_model,
            transcript_path.name,
            partials,
            prompt_terms,
        )
        coverage_map = build_coverage_map(args.host, args.final_model, transcript_path.name, merge_partials, prompt_terms)
        if coverage_output_path:
            write_notes(coverage_output_path, coverage_map, "coverage map")

        if args.mode in ("short", "both"):
            short_notes = final_summary_short(
                args.host,
                args.final_model,
                transcript_path.name,
                merge_partials,
                coverage_map,
                prompt_terms,
            )
            validate_final_summary("short meeting summary", short_notes, coverage_map, is_deep=False)
            ensure_summary_grounded("short meeting summary", short_notes, text)
            write_notes(short_output_path, short_notes, "short meeting summary")
            if not args.no_zh:
                short_notes_zh = final_summary_short_zh(
                    args.host,
                    args.final_model,
                    transcript_path.name,
                    merge_partials,
                    coverage_map,
                    prompt_terms,
                )
                validate_final_summary(
                    "Chinese short meeting summary",
                    short_notes_zh,
                    coverage_map,
                    is_deep=False,
                )
                ensure_summary_grounded("Chinese short meeting summary", short_notes_zh, text)
                write_notes(short_zh_output_path, short_notes_zh, "Chinese short meeting summary")
        if args.mode in ("deep", "both"):
            deep_notes = final_summary_deep(
                args.host,
                args.final_model,
                transcript_path.name,
                merge_partials,
                coverage_map,
                prompt_terms,
            )
            validate_final_summary("detailed meeting review", deep_notes, coverage_map, is_deep=True)
            ensure_summary_grounded("detailed meeting review", deep_notes, text)
            write_notes(deep_output_path, deep_notes, "detailed meeting review")
            if not args.no_zh:
                deep_notes_zh = final_summary_deep_zh(
                    args.host,
                    args.final_model,
                    transcript_path.name,
                    merge_partials,
                    coverage_map,
                    prompt_terms,
                )
                validate_final_summary(
                    "Chinese detailed meeting review",
                    deep_notes_zh,
                    coverage_map,
                    is_deep=True,
                )
                ensure_summary_grounded("Chinese detailed meeting review", deep_notes_zh, text)
                write_notes(deep_zh_output_path, deep_notes_zh, "Chinese detailed meeting review")
    except requests.exceptions.ConnectionError:
        print("[error] Cannot connect to Ollama. Start it with: ollama serve", file=sys.stderr)
        return 3
    except requests.HTTPError as exc:
        body = exc.response.text if exc.response is not None else ""
        print(f"[error] Ollama HTTP error: {exc}\n{body}", file=sys.stderr)
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
