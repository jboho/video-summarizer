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
- `ANTHROPIC_API_KEY` set in your environment (only needed for the summary)

## Usage

```bash
./summarize_video.py "https://www.youtube.com/watch?v=..."
```

Options:

| Flag | Meaning |
|------|---------|
| `--model MODEL` | Which Claude model to use. Default `claude-sonnet-5` (cheaper, plenty for summaries). Use `claude-opus-5` for sharper reasoning. |
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
- `metadata.json` — video details, chapter count, and which model summarized it.

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
