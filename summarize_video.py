#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["anthropic>=1.0", "openai>=1.0"]
# ///
"""
Fetch a video's transcript, format it, summarize it with Claude, and save
everything to one folder per video.

Usage:
    ./summarize_video.py <video-url> [--model MODEL] [--out DIR] [--no-summary]

Requires:
    - yt-dlp on PATH (brew install yt-dlp)
    - ANTHROPIC_API_KEY set (unless --no-summary)

Output (one subfolder per video under the output dir):
    <date>_<slug>/
        metadata.json      video details + run info
        transcript.md      formatted transcript with chapters and paragraphs
        transcript.txt     plain transcript, timestamp every 30s (fed to Claude)
        transcript.raw.vtt original subtitle file from yt-dlp
        summary.md         the Claude summary
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

# Default model. Override per-run with --model or the VIDEO_SUMMARY_MODEL env var.
# Sonnet is the default: cheaper and plenty for summarization. Use
# --model claude-opus-5 when a video needs sharper reasoning.
DEFAULT_MODEL = os.environ.get("VIDEO_SUMMARY_MODEL", "claude-sonnet-5")
DEFAULT_OUT = os.environ.get(
    "VIDEO_SUMMARY_DIR", str(Path.home() / "Documents" / "video-summaries")
)

# Emit a timestamp marker in the formatted transcript at least this often.
TIMESTAMP_EVERY_SECONDS = 30


def resolve_provider(model: str, provider_arg: str, base_url: str | None) -> str:
    """Pick the summarizer provider. Explicit flag wins; a base URL forces the
    OpenAI-compatible path; otherwise infer from the model name."""
    if provider_arg and provider_arg != "auto":
        return provider_arg
    if base_url or os.environ.get("OPENAI_BASE_URL"):
        return "openai"
    return "anthropic" if model.startswith("claude") else "openai"


def die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def check_yt_dlp() -> None:
    from shutil import which

    if which("yt-dlp") is None:
        die("yt-dlp not found on PATH. Install it with: brew install yt-dlp")


def fetch_metadata(url: str) -> dict:
    """Get video metadata as a dict via yt-dlp -J (no download)."""
    try:
        out = subprocess.run(
            ["yt-dlp", "-J", "--skip-download", "--no-warnings", url],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError as e:
        die(f"yt-dlp could not read the video:\n{e.stderr.strip()}")
    return json.loads(out)


def download_subs(url: str, workdir: Path) -> Path | None:
    """
    Download subtitles as VTT. Prefers manual English subs; falls back to
    auto-generated captions. Returns the path to the .vtt file, or None if the
    video has no captions in any English variant.
    """
    subprocess.run(
        [
            "yt-dlp",
            "--skip-download",
            "--write-subs",
            "--write-auto-subs",
            # en.* also matches en-US, en-GB, and YouTube's "en-orig" auto track.
            "--sub-langs",
            "en.*",
            "--sub-format",
            "vtt",
            "--convert-subs",
            "vtt",
            "--no-warnings",
            "-o",
            str(workdir / "%(id)s.%(ext)s"),
            url,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    vtts = sorted(workdir.glob("*.vtt"))
    if not vtts:
        return None
    # Prefer a plain "en" track over region/auto variants when several exist.
    for v in vtts:
        if re.search(r"\.en\.vtt$", v.name):
            return v
    return vtts[0]


_TAG_RE = re.compile(r"<[^>]+>")
_TS_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})")


def _ts_to_seconds(ts: str) -> float:
    m = _TS_RE.search(ts)
    if not m:
        return 0.0
    h, mnt, s, ms = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + int(s) + int(ms) / 1000.0


def parse_vtt(path: Path) -> list[tuple[float, str]]:
    """
    Parse a WebVTT file into (start_seconds, line) pairs.

    YouTube auto-captions repeat the previous line in each new cue (a rolling
    window), so consecutive duplicate lines are dropped. This keeps the reading
    order and the start time of the cue where each line first appeared.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    # Cues are separated by blank lines.
    blocks = re.split(r"\n\s*\n", raw)
    kept: list[tuple[float, str]] = []
    last_line: str | None = None

    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        header = lines[0].strip()
        if header.startswith("WEBVTT") or header.startswith("NOTE") or header == "":
            # Skip file header and NOTE metadata blocks, but a cue may still
            # follow inside the same block if timing is on a later line.
            timing_idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
            if timing_idx is None:
                continue
        else:
            timing_idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
            if timing_idx is None:
                continue

        start = _ts_to_seconds(lines[timing_idx].split("-->")[0])
        for text_line in lines[timing_idx + 1 :]:
            cleaned = _TAG_RE.sub("", text_line).strip()
            # Collapse repeated inner whitespace.
            cleaned = re.sub(r"\s+", " ", cleaned)
            if not cleaned:
                continue
            if cleaned == last_line:
                continue
            kept.append((start, cleaned))
            last_line = cleaned

    return kept


def _fmt_clock(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_transcript(segments: list[tuple[float, str]]) -> str:
    """
    Turn (start, line) pairs into readable text with a [HH:MM:SS] marker at
    least every TIMESTAMP_EVERY_SECONDS, each followed by a paragraph.
    """
    if not segments:
        return "(no transcript text)"
    out: list[str] = []
    para: list[str] = []
    last_marker = -TIMESTAMP_EVERY_SECONDS

    def flush() -> None:
        if para:
            out.append(" ".join(para))
            para.clear()

    for start, line in segments:
        if start - last_marker >= TIMESTAMP_EVERY_SECONDS:
            flush()
            out.append("")
            out.append(f"[{_fmt_clock(start)}]")
            last_marker = start
        para.append(line)
    flush()
    return "\n".join(out).strip() + "\n"


_SENTENCE_END_RE = re.compile(r"""[.!?]["'”’)\]]?$""")


def group_paragraphs(
    segments: list[tuple[float, str]],
    boundaries: list[float] | None = None,
    soft_words: int = 70,
    hard_words: int = 110,
) -> list[tuple[float, str]]:
    """
    Merge caption lines into readable paragraphs. Once a paragraph passes
    soft_words it ends at the next line that finishes a sentence; hard_words is
    a ceiling for captions with no punctuation (auto-generated ones). A chapter
    boundary always forces a break so no paragraph straddles a chapter.
    Caption start-to-start gaps are not a reliable pause signal, so they are not
    used. Returns (start_seconds, text) pairs.
    """
    bounds = sorted(boundaries or [])
    paras: list[tuple[float, str]] = []
    cur: list[tuple[float, str]] = []
    bi = 0
    prev_text: str | None = None

    def flush() -> None:
        if cur:
            paras.append((cur[0][0], " ".join(t for _, t in cur)))
            cur.clear()

    for start, text in segments:
        forced = False
        while bi < len(bounds) and start >= bounds[bi]:
            forced = True
            bi += 1
        words = sum(len(t.split()) for _, t in cur)
        sentence_end = prev_text is not None and _SENTENCE_END_RE.search(prev_text)
        if cur and (
            forced or words >= hard_words or (words >= soft_words and sentence_end)
        ):
            flush()
        cur.append((start, text))
        prev_text = text
    flush()
    return paras


def build_transcript_md(meta: dict, segments: list[tuple[float, str]]) -> str:
    """
    Readable Markdown transcript: video header, chapter headings when the video
    has them, and pause/length-based paragraphs. Each paragraph is prefixed with
    its start timestamp so a reader can jump to that spot in the video.
    """
    title = meta.get("title") or meta.get("id") or "Video"
    channel = meta.get("uploader") or meta.get("channel") or ""
    url = meta.get("webpage_url") or ""
    dur = _fmt_clock(meta.get("duration") or 0)

    out = [f"# {title}", "", f"[Watch]({url}) · {channel} · {dur}", ""]

    if not segments:
        out.append("_(no transcript text)_")
        return "\n".join(out) + "\n"

    chapters = [
        c for c in (meta.get("chapters") or []) if c.get("start_time") is not None
    ]

    def emit(paras: list[tuple[float, str]]) -> None:
        for pstart, ptext in paras:
            out.append(f"**[{_fmt_clock(pstart)}]** {ptext}")
            out.append("")

    if chapters:
        paras = group_paragraphs(
            segments, boundaries=[c["start_time"] for c in chapters]
        )
        for c in chapters:
            cs = c["start_time"]
            ce = c.get("end_time")
            out.append(f"## [{_fmt_clock(cs)}] {c.get('title') or 'Chapter'}")
            out.append("")
            emit([p for p in paras if p[0] >= cs and (ce is None or p[0] < ce)])
    else:
        emit(group_paragraphs(segments))

    return "\n".join(out).strip() + "\n"


def slugify(title: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug[:60] or "video"


SUMMARY_SYSTEM = (
    "You summarize video transcripts so a reader gets the high points without "
    "watching. The transcript is machine-generated and may have small errors "
    "or missing punctuation; read past those. When a claim maps to a moment in "
    "the video, cite the nearest [HH:MM:SS] marker from the transcript. Be "
    "accurate to what was actually said; do not invent details."
)

SUMMARY_INSTRUCTIONS = """\
Write a summary of this video transcript as Markdown with these sections:

## TL;DR
Two or three sentences capturing the whole video.

## Key points
The main points as bullets, in the order they come up. Add a [HH:MM:SS]
marker to a bullet when it clearly maps to one spot.

## Section walkthrough
The video broken into its natural segments. For each: a [HH:MM:SS] range or
start marker, a short heading, and one or two sentences on what it covers.

## Takeaways
What a viewer should remember or act on. Skip this section if the video has no
clear takeaways.

## Notable quotes
Up to five short, verbatim quotes worth keeping, each with its [HH:MM:SS]
marker. Skip this section if nothing stands out.
"""


def build_user_content(meta: dict, transcript: str) -> str:
    header = (
        f"Title: {meta.get('title')}\n"
        f"Channel: {meta.get('uploader') or meta.get('channel')}\n"
        f"Duration: {_fmt_clock(meta.get('duration') or 0)}\n"
        f"URL: {meta.get('webpage_url')}\n"
    )
    return (
        f"{SUMMARY_INSTRUCTIONS}\n"
        f"---\nVideo details:\n{header}\n"
        f"---\nTranscript:\n\n{transcript}"
    )


def summarize_anthropic(model: str, meta: dict, transcript: str) -> tuple[str, object]:
    import anthropic

    client = anthropic.Anthropic()
    user_content = build_user_content(meta, transcript)

    # Stream because a long transcript is a long request; get_final_message()
    # returns the assembled response. Adaptive thinking improves structure.
    with client.messages.stream(
        model=model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=SUMMARY_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    ) as stream:
        response = stream.get_final_message()

    if response.stop_reason == "refusal":
        detail = getattr(response, "stop_details", None)
        cat = getattr(detail, "category", None) if detail else None
        die(f"Claude declined to summarize this video (category: {cat}).")

    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return text, response.usage


def summarize_openai(
    model: str, meta: dict, transcript: str, base_url: str | None
) -> tuple[str, object]:
    from openai import OpenAI

    resolved_base = base_url or os.environ.get("OPENAI_BASE_URL") or None
    client = OpenAI(base_url=resolved_base)
    user_content = build_user_content(meta, transcript)

    stream = client.chat.completions.create(
        model=model,
        max_tokens=16000,  # see plan: newer OpenAI reasoning models want max_completion_tokens
        stream=True,
        stream_options={"include_usage": True},
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": user_content},
        ],
    )

    parts: list[str] = []
    usage = None
    finish = None
    refused = False
    for event in stream:
        if event.choices:
            choice = event.choices[0]
            delta = getattr(choice, "delta", None)
            if delta is not None:
                if getattr(delta, "content", None):
                    parts.append(delta.content)
                if getattr(delta, "refusal", None):
                    refused = True
            if choice.finish_reason:
                finish = choice.finish_reason
        if getattr(event, "usage", None):
            usage = event.usage

    if refused or finish == "content_filter":
        die(f"The model declined to summarize this video (finish_reason: {finish}).")

    text = "".join(parts).strip()
    return text, usage


def normalize_usage(provider: str, usage) -> tuple[int, int] | None:
    """Collapse the two SDKs' usage objects into one (input, output) tuple."""
    if usage is None:
        return None
    if provider == "anthropic":
        return (usage.input_tokens, usage.output_tokens)
    return (usage.prompt_tokens, usage.completion_tokens)


# input/output $ per 1M tokens for common models. Unknown models -> no estimate.
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4.1-mini": (0.4, 1.6),
}


def estimate_cost(
    model: str,
    usage: tuple[int, int] | None,
    price_override: tuple[float, float] | None,
) -> str | None:
    if usage is None:
        return None
    price = price_override or PRICES.get(model)
    if price is None:
        return None
    tin, tout = usage
    pin, pout = price
    cost = (tin * pin + tout * pout) / 1_000_000
    return f"${cost:.3f} ({tin} in / {tout} out)"


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize a video transcript.")
    ap.add_argument("url", help="Video URL (YouTube or any yt-dlp-supported site)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"default: {DEFAULT_MODEL}")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"default: {DEFAULT_OUT}")
    ap.add_argument(
        "--provider",
        choices=["auto", "anthropic", "openai"],
        default="auto",
        help="Model provider. 'auto' infers from --model/--base-url (default: auto).",
    )
    ap.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible endpoint (e.g. http://localhost:11434/v1 for Ollama, "
        "https://openrouter.ai/api/v1 for OpenRouter). Forces the openai provider.",
    )
    ap.add_argument(
        "--price-in",
        type=float,
        default=None,
        help="$ per 1M input tokens (cost override)",
    )
    ap.add_argument(
        "--price-out",
        type=float,
        default=None,
        help="$ per 1M output tokens (cost override)",
    )
    ap.add_argument(
        "--no-summary",
        action="store_true",
        help="Fetch and format the transcript only; skip the Claude summary.",
    )
    args = ap.parse_args()

    check_yt_dlp()

    print("Fetching video details...")
    meta = fetch_metadata(args.url)
    title = meta.get("title") or meta.get("id") or "video"
    print(f"  {title}")

    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td)
        print("Downloading transcript...")
        vtt = download_subs(args.url, workdir)
        if vtt is None:
            die(
                "No captions found for this video. This tool reads existing "
                "subtitles; it does not transcribe audio. Try a video that has "
                "captions, or ask to add local Whisper transcription."
            )
        segments = parse_vtt(vtt)
        transcript = format_transcript(segments)
        raw_vtt_text = vtt.read_text(encoding="utf-8", errors="replace")

    # Build the output folder.
    upload = meta.get("upload_date")  # YYYYMMDD
    date_prefix = (
        f"{upload[:4]}-{upload[4:6]}-{upload[6:8]}"
        if upload and len(upload) == 8
        else datetime.now().strftime("%Y-%m-%d")
    )
    folder = Path(args.out) / f"{date_prefix}_{slugify(title)}"
    folder.mkdir(parents=True, exist_ok=True)

    chapters = [
        c for c in (meta.get("chapters") or []) if c.get("start_time") is not None
    ]
    (folder / "transcript.txt").write_text(transcript, encoding="utf-8")
    (folder / "transcript.raw.vtt").write_text(raw_vtt_text, encoding="utf-8")
    (folder / "transcript.md").write_text(
        build_transcript_md(meta, segments), encoding="utf-8"
    )

    run_info = {
        "source_url": args.url,
        "title": title,
        "channel": meta.get("uploader") or meta.get("channel"),
        "duration_seconds": meta.get("duration"),
        "duration": _fmt_clock(meta.get("duration") or 0),
        "video_id": meta.get("id"),
        "upload_date": date_prefix,
        "view_count": meta.get("view_count"),
        "webpage_url": meta.get("webpage_url"),
        "fetched_at": datetime.now(UTC).isoformat(),
        "transcript_segments": len(segments),
        "chapters": len(chapters),
        "summary_model": None if args.no_summary else args.model,
    }
    (folder / "metadata.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"Transcript saved: {len(segments)} segments"
        + (f", {len(chapters)} chapters" if chapters else ", no chapters")
    )

    if not args.no_summary:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print(
                "warning: ANTHROPIC_API_KEY not set; skipping summary. "
                "Transcript files were still saved.",
                file=sys.stderr,
            )
        else:
            print(f"Summarizing with {args.model}...")
            summary, usage = summarize(args.model, meta, transcript)
            (folder / "summary.md").write_text(
                f"# {title}\n\n"
                f"[Watch]({meta.get('webpage_url')}) · "
                f"{run_info['channel']} · {run_info['duration']}\n\n"
                f"{summary}\n",
                encoding="utf-8",
            )
            cost = estimate_cost(args.model, usage)
            print("Summary saved" + (f" (~{cost})" if cost else ""))

    print(f"\nDone: {folder}")


if __name__ == "__main__":
    main()
