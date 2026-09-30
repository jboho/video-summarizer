# video-summarizer

Fetch a video's transcript, clean and timestamp it, summarize it with Claude,
and save everything to one folder per video. Built for skimming a long video
instead of watching it.

## What it does

Given a video URL, it:

1. Reads the video's details with `yt-dlp` (title, channel, duration, date).
2. Downloads the existing subtitles (manual English first, else auto-captions).
3. Cleans them into readable, timestamped text (drops the rolling-caption
   duplicates YouTube produces) and a formatted Markdown version with the
   video's chapters as headings and text arranged into paragraphs.
4. Sends the transcript to Claude and gets a structured summary.
5. Writes it all to `~/Documents/video-summaries/<date>_<slug>/`.

It reads captions that already exist on the video. It does **not** transcribe
audio, so a video with no captions won't work (see "Videos without captions").

## Requirements

- `yt-dlp` on your PATH — `brew install yt-dlp`
- `uv` — already installed; it handles the Python dependency automatically
- An API key for the summary step: `ANTHROPIC_API_KEY` for Claude (the default), or `OPENAI_API_KEY` for OpenAI-compatible endpoints. Not needed with `--no-summary`, and local endpoints (Ollama/LM Studio) need no key.

## Usage

```bash
./summarize_video.py "https://www.youtube.com/watch?v=..."
```

Options:

| Flag | Meaning |
|------|---------|
| `--model MODEL` | Model to use. Default `claude-sonnet-5`. Use `claude-opus-5` for sharper reasoning, or any OpenAI-compatible model name (e.g. `gpt-4o-mini`). |
| `--provider auto\|anthropic\|openai` | Which provider path to use. `auto` (default) infers from the model name and `--base-url`. |
| `--base-url URL` | An OpenAI-compatible endpoint (e.g. `https://openrouter.ai/api/v1`, or `http://localhost:11434/v1` for Ollama). Forces the OpenAI path. |
| `--price-in N` / `--price-out N` | Cost-estimate override in dollars per 1M input / output tokens, for models not in the built-in price table. |
| `--out DIR` | Output directory. Default `~/Documents/video-summaries`. |
| `--no-summary` | Fetch and format the transcript only; skip the Claude call (no cost). |

Environment overrides: `VIDEO_SUMMARY_MODEL`, `VIDEO_SUMMARY_DIR`.

### Make it a command

Add an alias so you can run it from anywhere:

```bash
# in ~/.zshrc
alias vidsum="/path/to/video-summarizer/summarize_video.py"
```

Then: `vidsum "https://youtu.be/..."`

## Using other models

This tool was built for Claude, and Claude stays the default. It also works with
any endpoint that speaks the OpenAI Chat Completions API — hosted or local.

The provider is chosen automatically: a `claude-*` model uses the Anthropic SDK; any
other model, or setting `--base-url`, uses the OpenAI-compatible path. Force it with
`--provider anthropic|openai`.

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

Cost is estimated only for models in the built-in price table; for anything else,
pass both `--price-in` and `--price-out` (dollars per 1M tokens) to get an estimate.
Passing only one is ignored.

Note: adaptive thinking is a Claude-only feature and is skipped on the OpenAI path.

## Output

Each run creates one folder, e.g.
`~/Documents/video-summaries/2020-09-08_react-in-100-seconds/`:

- `summary.md` — the Claude summary (TL;DR, key points, section walkthrough,
  takeaways, notable quotes), with `[HH:MM:SS]` markers back into the video.
- `transcript.md` — formatted, readable transcript: the video's chapters as
  headings (when it has them) and the text in paragraphs, each paragraph
  prefixed with its `[HH:MM:SS]` start time.
- `transcript.txt` — plain transcript with a timestamp marker every 30 seconds
  (this is the exact text sent to Claude).
- `transcript.raw.vtt` — the original subtitle file, kept as-is.
- `metadata.json` — video details, chapter count, and which provider and model
  summarized it.

## Cost

A summary costs a few cents. The tool prints the estimate after each run
(a 1-hour talk is roughly 15k input tokens). It defaults to `claude-sonnet-5`
to keep cost low; use `--no-summary` for no cost at all.

## Videos without captions

This tool only reads captions that already exist. If you want it to handle
videos with no captions, it can download the audio and transcribe it locally
with Whisper (`ffmpeg` is already installed) — ask and it can be added.

## Sources beyond YouTube

`yt-dlp` supports many sites (Vimeo, conference platforms, podcasts). Those
work as long as the video carries subtitles. YouTube is the tested path.

## License

MIT — see [LICENSE](./LICENSE).
