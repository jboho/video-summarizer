# Evals

`faithfulness.py` checks summaries against their transcripts. Run it with
`uv run evals/faithfulness.py COMMAND` from the repo root; `--help` on any
command lists its flags. The data lives outside this repo:

```
VIDEO_SUMMARY_DIR=~/Code/video-summaries     # transcripts, one folder per video
$VIDEO_SUMMARY_DIR/_eval/sets/phase2.json    # 20 videos: 12 dev, 8 held-out
$VIDEO_SUMMARY_DIR/_eval/runs/gen/VARIANT/   # summaries made by regen
$VIDEO_SUMMARY_DIR/_eval/runs/system/VARIANT/  # judge results, label tasks
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
uv run evals/faithfulness.py compare baseline grounded-v1
```

The report gives judge flags per summary and per 100 claims, the share of
summaries with no major flag (with a 95% CI), a breakdown by category and
video type, the rep0-vs-rep1 gap, and the deterministic check errors.

**Dev scoring needs labels.** A judge flag is only a candidate error. Until
the flags are labeled, flag counts mix real errors with judge mistakes, so a
drop in flags can mean the prompt got better or the judge got quieter. Label
them in [arvio](https://github.com/jboho/arvio), a small local labeling page.
It shows one flag at a time with the rules, the summary section, and the
transcript a minute either side of the times the summary cites and of the time
the judge pointed to. It keeps the judge's answer hidden until you finish a
session.

```
uv tool install git+https://github.com/jboho/arvio@v0.1.0   # once
uv run evals/faithfulness.py label-export --variant grounded-v1
arvio add $VIDEO_SUMMARY_DIR/_eval/runs/system/grounded-v1/label-dev  # once
arvio
```

`label-export` writes the flags into `runs/system/VARIANT/label-dev/`, an arvio
task folder, and arvio writes your labels there. Run it again after
`system-eval` adds results. Stop arvio first (Ctrl-C in its terminal): while it
runs it locks every task folder it knows, and `label-export` refuses a locked
folder. The same quote on the same summary line, flagged again in another rep
or variant, keeps its id (made from the video, the line and the quoted text),
so you label it once: the export leaves out flags already labeled in another
label folder. That works only for labels that exist when you export, so
export, label and Finish one variant before you export the next. Only labels
from a finished session count (Finish locks them).

The report then shows how many flagged quotes are labeled (the same quote on
the same line, flagged in several summaries, counts once), the judge's
precision on them with a 95% CI, and, once every flag is labeled, real errors
per summary and the share of summaries with no real major error (before that,
an estimate from a sample once all of it is labeled; see Fewer labels). Compare
variants on labeled real errors, and read the precision line before trusting
raw counts. A flag with an empty `summary_quote` has nothing to show, so it
can't be labeled; while any summary has one, the report leaves out real errors
per summary, counted or estimated, and says why. If one quote has a different
answer or severity in two label folders, the report says how many and uses a
finished label first, then the latest.

Any change to the task card, even to its title, marks the older label folders
stale in the report until you run `label-export` for them again. Do that
before you export a new variant: the export can't see a stale folder's labels,
so it would ask for those flags again. After that, their labels count again
unless the judge prompt, the labeling rules, the question, the choices or the
follow-ups changed: the card's pin is a hash of the judge prompt and the
rules. Labels made under an older card stay in `labels.jsonl` but no longer
count. A known limit: `results.jsonl` rows don't record which judge prompt
produced them, so after a judge prompt change, rows judged before and after it
sit in one variant's results unmarked.

**Fewer labels.** `label-export --sample 10` draws 10 of the variant's flagged
quotes in a seeded random order (`--seed`, default 1) and saves the draw in
`sample.json` in the label folder. Only quotes flagged today take part. A drawn
quote already labeled in another label folder isn't exported again, since that
label counts here too; items you labeled in this folder before stay, in the
draw or not. The same options on the same flags give the same draw, and a
larger N with the same seed adds to it. An export without `--sample` exports
every flag and leaves `sample.json` as it was.

While some flags are unlabeled, the report estimates real errors per summary
from the labels on the variant's drawn quotes only, wherever they were made,
once every drawn quote has a finished label. The whole draw is a random sample
of the variant's flags; part of it is not, since a drawn quote labeled in
another variant's folder counts at once, and those are the quotes both variants
flag. Until then the report says how many drawn quotes are labeled, and how
many wait in a session not finished yet. Labels from a full pass of another
variant on quotes outside the draw don't count toward the estimate. The
estimate is flagged quotes per summary times the judge's precision on the drawn
quotes, each quote weighted by the number of summaries that flag it, with can't
tell counted as not real, as in the counted figure. Once every flag is labeled,
the report shows the counted figure instead. There is no estimate for a
variant without a draw, for one whose flags changed since its draw (export
again), or for `--split all` when both splits have flags (a draw is per split).

`compare A B` prints both variants side by side: the number of summaries, flags
per summary, the share with no major flag, how many flagged quotes are labeled,
the judge's precision, and real errors per summary (counted, or marked
"(est.)"). Each rate has a 95% range; the two counts have none. Where the two
ranges overlap it says "ranges overlap"; otherwise it names the variant with
fewer flags, more summaries with no major flag, or fewer real errors. Precision
gets no verdict: the two variants share labels on the quotes they share, so
their ranges aren't independent. It warns when the variants cover different
videos or different reps of a video, and if the labels can't be read it still
prints the flag rows.

The ranges are approximate. A mean per summary uses a t multiplier, or an exact
Poisson range on the total when every summary has the same count. The
estimate's range multiplies the low ends, and the high ends, of the flags per
summary range and the precision's 95% CI, so it is wide. The variants are
summarized with the same model and judged by the same judge, so a difference
is the prompt's. Twelve dev videos is a small sample, and two reps of one video
are not independent, so the ranges are narrower than they should be: treat a
gap inside them as no difference.

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
free and allowed. Label the held-out flags the same way (`label-export --split
held-out`) before comparing.
If the held-out result disagrees with dev, report that; don't go back to step 1
and tune against held-out.

## Other commands

- `check VIDEO_DIR` runs the deterministic checks on one summary (free).
- `selfcheck` proves the checks and scorer on the gold cases (free).
- `judge-eval` and `judge-report` measure the judge itself on gold cases with
  planted errors (paid / free).
