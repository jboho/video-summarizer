<!-- markdownlint-disable MD033 MD041 -->

<h1 align="center">video-summarizer</h1>

<p align="center">
  <strong>Fetch a video's transcript, timestamp it, and summarize it with Claude or any OpenAI-compatible model</strong>
</p>

<p align="center">
  <a href="https://github.com/jboho/video-summarizer/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs welcome" /></a>
  <a href="https://docs.astral.sh/uv/"><img src="https://img.shields.io/badge/uv-script-DE5FE9.svg" alt="uv script" /></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white" alt="Python 3.11+" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-yellow.svg" alt="License: MIT" /></a>
</p>

<p align="center">
  <sub>CLI · single-file uv script · skim a long video instead of watching it</sub>
</p>

---

## Overview

**video-summarizer** reads the captions that already exist on a video (via [yt-dlp](https://github.com/yt-dlp/yt-dlp)), cleans them into readable timestamped text, sends them to a model, and saves everything to one folder per video. It does **not** transcribe audio, so a video with no captions won't work (see [Videos without captions](#videos-without-captions)).

| Topic          | Links                                                                         |
| -------------- | ----------------------------------------------------------------------------- |
| **License**    | [MIT](LICENSE)                                                                |
| **Source**     | [`summarize_video.py`](summarize_video.py) · tests in [`test_summarize_video.py`](test_summarize_video.py) |
| **Default model** | `claude-sonnet-5` (override with `--model` or `VIDEO_SUMMARY_MODEL`)       |

## Features

- **Timestamped transcript** — Downloads manual English subtitles first, else auto-captions, and drops the rolling-caption duplicates YouTube produces. A `[HH:MM:SS]` marker lands every 30 seconds.
- **Formatted Markdown transcript** — The video's chapters become headings and the text is arranged into paragraphs, each prefixed with its start time.
- **Structured summary** — TL;DR, key points, section walkthrough, takeaways, and notable quotes, with `[HH:MM:SS]` markers back into the video.
- **Multi-provider** — Claude through the Anthropic SDK by default, or any endpoint that speaks the OpenAI Chat Completions API: OpenAI, OpenRouter, or local Ollama and LM Studio.
- **Cost estimate** — Printed after every run for models in the built-in price table; pass `--price-in` and `--price-out` for any other model.
- **Free mode** — `--no-summary` fetches and formats the transcript only, with no API call.
- **Other sites** — Anything yt-dlp supports (Vimeo, conference platforms, podcasts) works as long as the video carries subtitles. YouTube is the tested path.

## Requirements

- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)** on your PATH — `brew install yt-dlp`
- **[uv](https://docs.astral.sh/uv/)** — runs the script and installs its Python dependencies (`anthropic`, `openai`) from the inline PEP 723 header
- **An API key** for the summary step: `ANTHROPIC_API_KEY` for Claude (the default), or `OPENAI_API_KEY` for OpenAI-compatible endpoints. Not needed with `--no-summary`, and local endpoints (Ollama, LM Studio) need no key.

## Quick start

```bash
git clone https://github.com/jboho/video-summarizer.git
cd video-summarizer
./summarize_video.py "https://www.youtube.com/watch?v=..."
```

To run it from anywhere, add an alias:

```bash
# in ~/.zshrc
alias vidsum="/path/to/video-summarizer/summarize_video.py"
```

Then: `vidsum "https://youtu.be/..."`

## Options

| Flag                               | Meaning                                                                                                                                     |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `--model MODEL`                    | Model to use. Default `claude-sonnet-5`. Use `claude-opus-5` for sharper reasoning, or any OpenAI-compatible model name (e.g. `gpt-4o-mini`). |
| `--provider auto\|anthropic\|openai` | Which provider path to use. `auto` (default) infers from the model name and `--base-url`.                                                  |
| `--base-url URL`                   | An OpenAI-compatible endpoint (e.g. `https://openrouter.ai/api/v1`, or `http://localhost:11434/v1` for Ollama). Forces the OpenAI path.     |
| `--price-in N` / `--price-out N`   | Cost-estimate override in dollars per 1M input / output tokens, for models not in the built-in price table. Both are required; one alone is ignored. |
| `--out DIR`                        | Output directory. Default `~/Documents/video-summaries`.                                                                                    |
| `--no-summary`                     | Fetch and format the transcript only; skip the model call (no cost).                                                                        |

Environment overrides: `VIDEO_SUMMARY_MODEL`, `VIDEO_SUMMARY_DIR`.

## Using other models

Claude stays the default. The provider is chosen automatically: a `claude-*` model uses the Anthropic SDK; any other model, or setting `--base-url`, uses the OpenAI-compatible path. Force it with `--provider anthropic|openai`.

```bash
# Claude (default) — needs ANTHROPIC_API_KEY
vidsum "https://youtu.be/..."

# OpenAI — needs OPENAI_API_KEY
vidsum "https://youtu.be/..." --model gpt-4o-mini

# OpenRouter — needs OPENAI_API_KEY set to your OpenRouter key
vidsum "https://youtu.be/..." --model anthropic/claude-sonnet-5 \
  --base-url https://openrouter.ai/api/v1

# Local Ollama — no key needed
vidsum "https://youtu.be/..." --model llama3.1 \
  --base-url http://localhost:11434/v1
```

Adaptive thinking is a Claude-only feature and is skipped on the OpenAI path.

## Output

Each run creates one folder, e.g. `~/Documents/video-summaries/2020-09-08_react-in-100-seconds/`:

| File                  | Contents                                                                                                                   |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `summary.md`          | The model's summary (TL;DR, key points, section walkthrough, takeaways, notable quotes) with `[HH:MM:SS]` markers.         |
| `transcript.md`       | Formatted transcript: chapters as headings (when the video has them), paragraphs prefixed with their `[HH:MM:SS]` start time. |
| `transcript.txt`      | Plain transcript with a timestamp marker every 30 seconds. This is the exact text sent to the model.                       |
| `transcript.raw.vtt`  | The original subtitle file, kept as-is.                                                                                    |
| `metadata.json`       | Video details, chapter count, and which provider and model summarized it.                                                  |

## Cost

A summary costs a few cents; a 1-hour talk is roughly 15k input tokens. The tool prints the estimate after each run. It defaults to `claude-sonnet-5` to keep cost low, and `--no-summary` costs nothing.

## Development

```bash
uv run --with pytest pytest -q test_summarize_video.py
```

Lint and format config for [Ruff](https://docs.astral.sh/ruff/) lives in [`pyproject.toml`](pyproject.toml). The tool is a single-file uv script and is not packaged.

## Videos without captions

This tool only reads captions that already exist. Videos with no captions could be handled by downloading the audio and transcribing it locally with Whisper (`ffmpeg` is needed); that is not built yet.

## Contributing

Issues and pull requests are welcome. Run the tests and `ruff check` before opening a PR.

## License

Published under the [MIT License](LICENSE).
