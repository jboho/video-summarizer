# Evals

`faithfulness.py` checks summaries against their transcripts. Run it with
`uv run evals/faithfulness.py COMMAND` from the repo root; `--help` on any
command lists its flags. The data lives outside this repo:

```
VIDEO_SUMMARY_DIR=~/Code/video-summaries     # transcripts, one folder per video
$VIDEO_SUMMARY_DIR/_eval/sets/phase2.json    # 20 videos: 12 dev, 8 held-out
$VIDEO_SUMMARY_DIR/_eval/runs/gen/VARIANT/   # summaries made by regen
$VIDEO_SUMMARY_DIR/_eval/runs/system/VARIANT/  # judge results, labels files
```

Set `VIDEO_SUMMARY_DIR` first; the default is `~/Documents/video-summaries`.

Commands marked paid call the Anthropic API and need `ANTHROPIC_API_KEY`. Every
paid command takes `--dry-run`, which prints the job list and a rough token
count and calls nothing.

## The prompt-edit loop

The goal is to change the summarizer prompt and know whether it got better,
without fitting the prompt to the videos that grade it. The set is split: you
iterate on **dev** as often as you like, and score **held-out** once, at the
end, for the variant you picked.

### 1. Edit the prompt

Add an entry to `PROMPTS` in `evals/prompts.py` under a new name. It overrides
the system prompt and/or individual section blocks; anything left out comes
from production. `baseline` is the production prompt in `summarize_video.py`,
and stays that way: the loop never edits the production prompt. Ship a winner
by copying it into `summarize_video.py` in its own PR.

Don't edit an entry once it has summaries on disk. `regen` stamps each variant
folder with a fingerprint of the prompt and refuses to add summaries made with
a different one; make a new name instead (`grounded-v2`).

### 2. Run it on dev

```
uv run evals/faithfulness.py regen --prompt grounded-v1 --variant grounded-v1 --dry-run
uv run evals/faithfulness.py regen --prompt grounded-v1 --variant grounded-v1 --reps 2
uv run evals/faithfulness.py system-eval --variant grounded-v1 --dry-run
uv run evals/faithfulness.py system-eval --variant grounded-v1
```

`--split` defaults to `dev` for both. `regen` writes summaries (paid); a
re-run skips the ones already on disk. `system-eval` runs the free
deterministic checks and the LLM judge (paid) and ends with the dev report.
Use `--reps 2` to match the baseline, so rep-to-rep noise is comparable.

### 3. Compare

```
uv run evals/faithfulness.py system-report --variant baseline
uv run evals/faithfulness.py system-report --variant grounded-v1
```

The report gives judge flags per summary and per 100 claims, the share of
summaries with no major flag (with a 95% CI), a breakdown by category and
video type, the rep0-vs-rep1 gap, and the deterministic check errors.

**Dev scoring needs labels.** A judge flag is only a candidate error. Each
report writes `runs/system/VARIANT/labels-dev.json` with one entry per flag;
set `verdict` to `"real"` or `"false_alarm"` (and add a `note` if useful).
Re-running the report keeps verdicts already filled in. Until the flags are
labeled, flag counts mix real errors with judge mistakes, so a drop in flags
can mean the prompt got better or the judge got quieter. Compare variants on
labeled real errors, and read the judge precision line before trusting raw
counts. The baseline's file is `runs/system/baseline/labels-dev.json`.

The variants are summarized with the same model and judged by the same judge,
so a difference is the prompt's. Twelve dev videos is a small sample: treat a
gap inside the CIs as no difference.

### 4. Repeat

Go back to step 1 with a new name. Keep the losers' folders; they are the record
of what was tried.

### 5. Held-out, once

When you have picked one variant, score it on held-out:

```
uv run evals/faithfulness.py regen --prompt grounded-v1 --variant grounded-v1 --split held-out --reps 2
uv run evals/faithfulness.py system-eval --variant grounded-v1 --split held-out
uv run evals/faithfulness.py system-report --variant grounded-v1 --split held-out
uv run evals/faithfulness.py system-report --variant baseline --split held-out
```

`system-eval` records each held-out run in
`runs/system/VARIANT/held-out-runs.json` and refuses a second held-out run of
the same variant (`--split held-out` or `--split all`). `--force-held-out`
overrides that, and the forced run is recorded too. Re-printing the report is
free and allowed. Label `labels-held-out.json` the same way before comparing.
If the held-out result disagrees with dev, report that; don't go back to step 1
and tune against held-out.

## Other commands

- `check VIDEO_DIR` runs the deterministic checks on one summary (free).
- `selfcheck` proves the checks and scorer on the gold cases (free).
- `judge-eval` and `judge-report` measure the judge itself on gold cases with
  planted errors (paid / free).
