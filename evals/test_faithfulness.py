import tomllib

import faithfulness as f
import pytest

TRANSCRIPT = """\
[00:00:02]
&gt;&gt; Welcome back. Set the layers as low as you can.

[00:00:32]
Sonnet took about 45 seconds while the local model took about 2 and 1/2 minutes.
"""

SUMMARY = """\
# Title

[Watch](https://example.com) · Channel · 00:01:00

## TL;DR
Short.

## Actionable summary (How-to: tune it)
1. Start low on the layers [00:00:02].
2. Sonnet took ~45 seconds vs. ~2.5 minutes [00:00:32].

## Key points
- Point [00:00:32].

## Section walkthrough
- [00:00:02] Intro.
"""

SECTIONS = ["tldr", "actionable", "key-points", "walkthrough"]


def test_norm_strips_entities_markdown_and_case():
    assert f.norm("&gt;&gt; **Keep  GPU** `low`") == ">> keep gpu low"


def test_clean_summary_has_no_findings():
    assert f.run_checks(SUMMARY, TRANSCRIPT, 60, SECTIONS) == []


def test_missing_section_is_flagged():
    text = SUMMARY.replace("## Key points", "## Points")
    found = f.check_sections(text, SECTIONS)
    assert [x.detail for x in found] == ["missing section header '## key points'"]


def test_citation_not_a_marker_and_past_end():
    text = SUMMARY.replace("layers [00:00:02]", "layers [00:00:10]").replace(
        "Point [00:00:32]", "Point [00:05:00]"
    )
    found = f.check_citations(text, f.transcript_markers(TRANSCRIPT), 60)
    assert [(x.detail, x.advisory) for x in found] == [
        ("[00:00:10] is not a transcript marker", True),
        ("[00:05:00] is past the video's end", False),
    ]


def test_number_check_flags_only_unsaid_numbers():
    text = SUMMARY.replace("~45 seconds", "~54 seconds")
    found = f.check_numbers(text, TRANSCRIPT)
    assert [(x.detail, x.advisory) for x in found] == [
        ("54 not found in transcript", True)
    ]


CASE = {
    "video": "2026-01-01_x",
    "known_errors": [
        {
            "id": "e1",
            "match": ["start high"],
            "fix": {"find": "Start low", "replace": "Start low"},
        }
    ],
    "perturbations": [
        {
            "id": "p01",
            "category": "contradicted",
            "find": "~45 seconds",
            "replace": "~54 seconds",
            "match": ["~54 seconds"],
        },
        {
            "id": "p02",
            "category": "wrong_citation",
            "checker": "deterministic",
            "find": "layers [00:00:02]",
            "replace": "layers [00:02:00]",
            "match": ["00:02:00"],
        },
    ],
}


def test_apply_edit_needs_exactly_one_target():
    with pytest.raises(ValueError):
        f.apply_edit("a a", "a", "b")
    with pytest.raises(ValueError):
        f.apply_edit("a", "z", "b")
    assert f.apply_edit("a b", "b", "c") == "a c"


def test_build_variants_plants_one_change_each():
    vs = f.build_variants(CASE, SUMMARY)
    assert [v.kind for v in vs] == ["real", "control", "planted", "planted"]
    control = vs[1].text
    for v in vs[2:]:
        pairs = zip(control.splitlines(), v.text.splitlines(), strict=True)
        diff = [a for a, b in pairs if a != b]
        assert len(diff) == 1
    assert vs[3].deterministic_only


def test_scorer_oracle_partial_null_and_wrong_line():
    v = f.build_variants(CASE, SUMMARY)[2]
    exp = v.expected[0]
    line = f.target_lines(v.text, exp)[0]
    assert f.is_detected(v.text, exp, [{"summary_quote": line}])
    # The judge may quote only part of the claim, e.g. the marker.
    assert f.is_detected(
        v.text, exp, [{"summary_quote": "vs. ~2.5 minutes [00:00:32]"}]
    )
    assert not f.is_detected(v.text, exp, [])
    assert not f.is_detected(
        v.text, exp, [{"summary_quote": "Start low on the layers"}]
    )


def test_flag_verdict_uses_only_decided_labels():
    labels = [
        {"id": "f1", "match": ["load faster"], "verdict": "false_alarm"},
        {"id": "f2", "match": ["maxOutputTokens"], "verdict": None},
    ]
    assert f.flag_verdict({"summary_quote": "Models **load faster**"}, labels) == (
        "false_alarm"
    )
    assert f.flag_verdict({"summary_quote": "`maxOutputTokens` copied"}, labels) is None


def test_case_prefix_prefers_explicit_id():
    assert f.case_prefix({"video": "2026-01-01_x"}) == "2026-01-01"
    assert f.case_prefix({"video": "2026-01-01_x", "id": "x"}) == "x"


def test_wilson_matches_known_interval():
    lo, hi = f.wilson(4, 4)
    assert round(lo, 2) == 0.51 and hi == pytest.approx(1.0)


def test_price_key_accepts_snapshots_and_rejects_unknown():
    assert f.price_key("claude-opus-5-5-20260901") == "claude-opus-5-5"
    with pytest.raises(KeyError):
        f.price_key("some-other-model")


SAVED_META = {
    "title": "A video",
    "channel": "A channel",
    "duration_seconds": 2697,
    "duration": "00:44:57",
    "webpage_url": "https://example.com/v",
}


def test_prompt_meta_feeds_the_production_header():
    content = f.sv.build_user_content(f.prompt_meta(SAVED_META), "T", SECTIONS)
    assert "Channel: A channel\nDuration: 00:44:57\n" in content


def test_prompt_sha_tracks_the_sections():
    assert f.prompt_sha(SECTIONS) == f.prompt_sha(list(SECTIONS))
    assert f.prompt_sha(SECTIONS) != f.prompt_sha(["tldr"])


def test_check_stamp_refuses_a_changed_prompt(tmp_path):
    path = tmp_path / "v" / "variant.json"
    stamp = {"prompt_sha": "abc", "model": "m", "sections": SECTIONS}
    f.check_stamp(path, stamp)
    f.check_stamp(path, dict(stamp))
    with pytest.raises(SystemExit, match="prompt_sha"):
        f.check_stamp(path, stamp | {"prompt_sha": "def"})


def test_load_set_filters_by_split(tmp_path, monkeypatch):
    monkeypatch.setattr(f, "EVAL_DIR", tmp_path)
    (tmp_path / "sets").mkdir()
    rows = [{"video": "a", "split": "dev"}, {"video": "b", "split": "held-out"}]
    (tmp_path / "sets" / "s.json").write_text(f.json.dumps(rows))
    assert [r["video"] for r in f.load_set("s", "dev")] == ["a"]
    assert len(f.load_set("s", "all")) == 2
    (tmp_path / "sets" / "e.json").write_text(f.json.dumps(rows[:1]))
    with pytest.raises(SystemExit):
        f.load_set("e", "held-out")


QUOTE = "Sonnet took ~45 seconds vs. ~2.5 minutes"


def _issue(quote: str = QUOTE, **kw) -> dict:
    return {
        "summary_quote": quote,
        "category": "unsupported",
        "severity": "minor",
        "transcript_ts": "00:00:32",
        "transcript_quote": "t",
        "explanation": "e",
    } | kw


def _row(rep: int, issues: list[dict], video: str = "2026-01-01_x") -> dict:
    return {
        "video": video,
        "rep": rep,
        "type": "how-to",
        "split": "dev",
        "status": "ok",
        "issues": issues,
    }


def test_flag_key_ignores_rep_markdown_and_case():
    key = f.flag_key("2026-01-01_x", "**Start** low")
    assert key == f.flag_key("2026-01-01_x", "start  LOW")
    assert key != f.flag_key("2026-01-02_y", "start low")
    assert key.startswith("vs1-") and len(key) == 20


def test_label_card_parses_and_follows_the_judge_prompt():
    card = tomllib.loads(f.label_card())
    assert [c["key"] for c in card["choices"]] == ["real", "false_alarm", "cant_tell"]
    hotkeys = [c["hotkey"] for c in card["choices"]] + [
        o["hotkey"] for o in card["followups"][0]["options"]
    ]
    assert len(set(hotkeys)) == len(hotkeys) and "n" not in hotkeys
    assert card["pin"] == f.judge_pin()
    rules = " ".join(card["rules"])
    issue = f.ISSUE_SCHEMA["properties"]["issues"]["items"]["properties"]
    for category in issue["category"]["enum"]:
        assert category.replace("_", " ") in rules
    assert [o["key"] for o in card["followups"][0]["options"]] == issue["severity"][
        "enum"
    ]


def test_transcript_windows_merge_neighbours_and_split_far_ones():
    text = "".join(
        f"[00:{m:02d}:{s:02d}]\npart {i}\n\n"
        for i, (m, s) in enumerate([(0, 0), (0, 30), (3, 20), (3, 50), (6, 40)])
    )
    chunks = f.transcript_chunks(text)
    assert [start for start, _ in chunks] == [0, 30, 200, 230, 400]
    assert chunks[1][1] == "[00:00:30]\npart 1"
    assert f.transcript_windows(chunks, [30, 400]) == [[0, 1], [3, 4]]
    assert f.transcript_windows(chunks, []) == []


def test_flag_item_shows_the_section_and_hides_the_judge():
    saved = {"title": "T", "duration": "00:01:00"}
    item = f.flag_item(_row(0, []), _issue(), SUMMARY, TRANSCRIPT, saved, "base")
    shown = item["display"][1]
    assert shown["text"].startswith("## Actionable summary")
    assert shown["text"].endswith("~2.5 minutes [00:00:32].")
    assert shown["highlight"] == [QUOTE]
    window = item["display"][2]
    assert window["title"] == "Transcript [00:00:02] to [00:00:32]"
    assert "45 seconds" in window["text"]
    assert [b.get("collapsed") for b in item["display"][3:]] == [True, True]
    assert item["judge"]["verdict"] == "error"
    assert "explanation" not in str(item["display"])

    loose = f.flag_item(
        _row(0, []), _issue("start LOW on the layers"), SUMMARY, TRANSCRIPT, saved, "b"
    )
    assert loose["display"][1]["highlight"] == [
        "1. Start low on the layers [00:00:02]."
    ]
    gone = f.flag_item(
        _row(0, []), _issue("Not in it."), SUMMARY, TRANSCRIPT, saved, "b"
    )
    assert gone["display"][1]["text"] == "Not in it."
    assert gone["display"][1]["title"].endswith("(not found in it word for word)")


def _export_setup(tmp_path, monkeypatch, rows: list[dict], variant: str = "base"):
    monkeypatch.setattr(f, "EVAL_DIR", tmp_path / "_eval")
    monkeypatch.setattr(f, "CORPUS", tmp_path)
    for r in rows:
        vdir = tmp_path / r["video"]
        vdir.mkdir(exist_ok=True)
        (vdir / "transcript.txt").write_text(TRANSCRIPT)
        (vdir / "metadata.json").write_text('{"title": "T", "duration": "00:01:00"}')
        rdir = f.gen_dir(variant) / r["video"] / f"rep{r['rep']}"
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "summary.md").write_text(SUMMARY)
    f.system_dir(variant).mkdir(parents=True, exist_ok=True)
    (f.system_dir(variant) / "results.jsonl").write_text(
        "".join(f.json.dumps(r) + "\n" for r in rows)
    )
    return f.argparse.Namespace(variant=variant, split="dev")


def test_label_export_writes_one_item_per_flagged_sentence(
    tmp_path, monkeypatch, capsys
):
    other = _issue("Point [00:00:32].", transcript_ts="[00:00:32]")
    rows = [_row(0, [_issue(), other]), _row(1, [_issue()]), _row(2, [])]
    args = _export_setup(tmp_path, monkeypatch, rows)
    assert f.cmd_label_export(args) == 0
    task = f.label_task("base", "dev")
    assert (task / "task.toml").read_text() == f.label_card()
    first = (task / "items.jsonl").read_bytes()
    ids = [i["id"] for i in f.read_jsonl(task / "items.jsonl")]
    assert ids == [f.flag_key("2026-01-01_x", q) for q in (QUOTE, "Point [00:00:32].")]
    assert (
        "3 judge flags in 3 summaries -> 2 items (1 repeats" in capsys.readouterr().out
    )
    f.cmd_label_export(args)
    assert (task / "items.jsonl").read_bytes() == first


def test_label_export_keeps_labeled_items_and_skips_labels_elsewhere(
    tmp_path, monkeypatch, capsys
):
    other = _issue("Point [00:00:32].")
    args = _export_setup(tmp_path, monkeypatch, [_row(0, [_issue(), other])])
    f.cmd_label_export(args)
    task = f.label_task("base", "dev")
    items = f.read_jsonl(task / "items.jsonl")
    items[0]["display"] = [{"kind": "text", "text": "as the labeler saw it"}]
    (task / "items.jsonl").write_text("".join(f.json.dumps(i) + "\n" for i in items))
    (task / "labels.jsonl").write_text(
        f.json.dumps({"kind": "label", "item_id": items[0]["id"]}) + "\n{torn"
    )
    f.cmd_label_export(args)
    again = f.read_jsonl(task / "items.jsonl")
    assert again[0] == items[0]
    assert again[1]["display"][1]["highlight"] == ["Point [00:00:32]."]

    labeled = {"item_id": items[1]["id"], "labeled_at": "2026-10-18T10:00:00Z"}
    monkeypatch.setattr(f, "read_labels", lambda t: [labeled] if t == task else [])
    v2 = _export_setup(tmp_path, monkeypatch, [_row(0, [_issue(), other])], "v2")
    capsys.readouterr()
    f.cmd_label_export(v2)
    assert [
        i["id"] for i in f.read_jsonl(f.label_task("v2", "dev") / "items.jsonl")
    ] == [items[0]["id"]]
    assert "1 already labeled in another variant" in capsys.readouterr().out


def _label(key: str, answer: str, locked: bool = True, severity: str = "") -> dict:
    return {
        "item_id": key,
        "answer": answer,
        "followups": {"severity": severity} if severity else {},
        "labeled_at": "2026-10-18T10:00:00Z",
        "locked": locked,
    }


def test_report_labels_counts_finished_labels_and_precision(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(f, "EVAL_DIR", tmp_path)
    quotes = ["a", "b", "c", "d", "e"]
    part = [
        _row(0, [_issue(q) for q in quotes[:3]]),
        _row(1, [_issue(q) for q in quotes[3:]]),
    ]
    keys = [f.flag_key("2026-01-01_x", q) for q in quotes]
    task = f.label_task("base", "dev")
    task.mkdir(parents=True)
    (task / "task.toml").write_text(f.label_card())
    stale = f.label_task("old", "dev")
    stale.mkdir(parents=True)
    (stale / "task.toml").write_text("format = 1\n")
    labels = [
        _label(keys[0], "real", severity="major"),
        _label(keys[1], "false_alarm"),
        _label(keys[2], "real", severity="minor"),
        _label(keys[3], "real", locked=False, severity="minor"),
    ]

    def fake(t):
        assert t == task, "a folder with an older card must not be read"
        return labels

    monkeypatch.setattr(f, "read_labels", fake)
    f.report_labels(part, "base", "dev")
    out = capsys.readouterr().out
    assert "has an older card" in out
    assert "flags labeled: 3/5 (real 2, false alarm 1, can't tell 0)" in out
    assert "1 more labeled in an arvio session not finished yet" in out
    lo, hi = f.wilson(2, 3)
    assert f"judge precision on labeled flags: 2/3 (95% CI {lo:.2f}-{hi:.2f}" in out
    assert "real errors per summary" not in out

    labels[3]["locked"] = True
    labels.append(_label(keys[4], "cant_tell"))
    f.report_labels(part, "base", "dev")
    out = capsys.readouterr().out
    assert "real errors per summary: mean 1.50" in out
    assert "no real major error: 1/2" in out
    assert "can't tell counted as no error" in out


def test_report_labels_says_how_to_start(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(f, "EVAL_DIR", tmp_path)
    f.report_labels([_row(0, [_issue()])], "base", "all")
    out = capsys.readouterr().out
    assert "flags labeled: 0/1" in out
    assert "label-export --variant base --split dev" in out


def test_read_labels_fails_closed(tmp_path, monkeypatch):
    def missing(*a, **kw):
        raise FileNotFoundError("arvio")

    monkeypatch.setattr(f.subprocess, "run", missing)
    with pytest.raises(SystemExit, match="uv tool install"):
        f.read_labels(tmp_path)
    failed = f.subprocess.CompletedProcess([], 1, "", "task.toml: bad")
    monkeypatch.setattr(f.subprocess, "run", lambda *a, **kw: failed)
    with pytest.raises(SystemExit, match="task.toml: bad"):
        f.read_labels(tmp_path)


def test_run_parallel_records_rows_and_failures(tmp_path):
    def one(name: str) -> dict:
        if name == "boom":
            raise RuntimeError("bad")
        if name == "refused":
            raise SystemExit(1)
        return {"name": name}

    results = tmp_path / "results.jsonl"
    order: list[str] = []
    f.run_parallel(
        [("a",), ("boom",), ("refused",)],
        lambda n: order.append(n) or one(n),
        ident=lambda n: {"job": n},
        show=lambda row: "",
        warm_key=lambda n: "same",
        results_path=results,
        workers=2,
    )
    assert order[0] == "a"  # the warm job runs alone first
    assert [f.json.loads(x)["name"] for x in results.read_text().splitlines()] == ["a"]
    errors = [
        f.json.loads(x) for x in (tmp_path / "errors.jsonl").read_text().splitlines()
    ]
    assert sorted((e["job"], e["class"]) for e in errors) == [
        ("boom", "RuntimeError"),
        ("refused", "SystemExit"),
    ]


def test_prompt_sha_tracks_the_provider():
    assert f.prompt_sha(SECTIONS) == f.prompt_sha(SECTIONS, "anthropic")
    assert f.prompt_sha(SECTIONS) != f.prompt_sha(SECTIONS, "openai")


def test_regen_with_base_url_uses_the_openai_path(tmp_path, monkeypatch):
    corpus = tmp_path
    monkeypatch.setattr(f, "CORPUS", corpus)
    monkeypatch.setattr(f, "EVAL_DIR", corpus / "_eval")
    (corpus / "_eval" / "sets").mkdir(parents=True)
    rows = [{"video": "v", "type": "list", "split": "dev"}]
    (corpus / "_eval" / "sets" / "s.json").write_text(f.json.dumps(rows))
    (corpus / "v").mkdir()
    (corpus / "v" / "metadata.json").write_text(
        f.json.dumps(SAVED_META | {"title": "T", "channel": "C"})
    )
    (corpus / "v" / "transcript.txt").write_text("[00:00:01] hi")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    calls = []

    def fake_summarize(provider, model, meta, transcript, base_url, sections):
        calls.append((provider, model, base_url))

        class U:
            prompt_tokens, completion_tokens = 10, 5

        return "## TL;DR\nok", U()

    monkeypatch.setattr(f.sv, "summarize", fake_summarize)
    base = "http://localhost:11434/v1"
    args = f.argparse.Namespace(
        set="s",
        split="all",
        variant="local",
        prompt="baseline",
        model="llama3.1:8b",
        sections=",".join(SECTIONS),
        base_url=base,
        reps=2,
        workers=1,
        dry_run=False,
    )
    assert f.cmd_regen(args) == 0
    assert calls == [("openai", "llama3.1:8b", base)] * 2
    out = corpus / "_eval" / "runs" / "gen" / "local"
    stamp = f.json.loads((out / "variant.json").read_text())
    assert stamp["base_url"] == base
    assert stamp["prompt_sha"] == f.prompt_sha(SECTIONS, "openai")
    assert (out / "v" / "rep1" / "summary.md").read_text().endswith("ok\n")
    lines = (out / "results.jsonl").read_text().splitlines()
    assert [f.json.loads(x)["usage"]["output_tokens"] for x in lines] == [5, 5]
    assert all(f.json.loads(x)["cost"] is None for x in lines)


def test_baseline_prompt_is_production_and_variants_change_the_sha(monkeypatch):
    monkeypatch.setattr(f.sv, "SUMMARY_SYSTEM", f.sv.SUMMARY_SYSTEM)
    monkeypatch.setattr(f.sv, "SECTION_PROMPTS", f.sv.SECTION_PROMPTS)
    before = f.prompt_sha(SECTIONS)
    f.prompts.apply_prompt("baseline")
    assert f.prompt_sha(SECTIONS) == before
    f.prompts.apply_prompt("grounded-v1")
    assert f.prompt_sha(SECTIONS) != before
    f.prompts.apply_prompt("baseline")
    assert f.prompt_sha(SECTIONS) == before
    with pytest.raises(SystemExit, match="unknown --prompt"):
        f.prompts.apply_prompt("nope")


def test_variant_sections_keep_the_production_headers():
    for name, entry in f.prompts.PROMPTS.items():
        for key, text in entry.get("sections", {}).items():
            base = f.prompts.BASE_SECTIONS[key].splitlines()[0]
            assert text.splitlines()[0] == base, (name, key)


def _held_out_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(f, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(f, "CORPUS", tmp_path)
    (tmp_path / "sets").mkdir()
    rows = [{"video": "a", "type": "t", "split": "held-out"}]
    (tmp_path / "sets" / "s.json").write_text(f.json.dumps(rows))
    rdir = tmp_path / "runs" / "gen" / "v" / "a" / "rep0"
    rdir.mkdir(parents=True)
    (rdir / "summary.md").write_text("s")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "transcript.txt").write_text("t")
    (rdir.parent.parent / "variant.json").write_text(f.json.dumps({"sections": []}))
    return f.argparse.Namespace(
        set="s",
        split="held-out",
        variant="v",
        model="m",
        dry_run=True,
        force_held_out=False,
    )


def test_system_eval_refuses_a_second_held_out_run(tmp_path, monkeypatch, capsys):
    args = _held_out_setup(tmp_path, monkeypatch)
    assert f.cmd_system_eval(args) == 0
    run_dir = f.system_dir("v")
    run_dir.mkdir(parents=True)
    f.record_held_out(run_dir, args, 1)
    with pytest.raises(SystemExit, match="already scored"):
        f.cmd_system_eval(args)
    args.force_held_out = True
    assert f.cmd_system_eval(args) == 0
    f.record_held_out(run_dir, args, 1)
    runs = f.json.loads((run_dir / f.HELD_OUT_MARKER).read_text())
    assert [r["forced"] for r in runs] == [False, True]


def test_system_eval_dev_ignores_the_held_out_guard(tmp_path, monkeypatch):
    args = _held_out_setup(tmp_path, monkeypatch)
    run_dir = f.system_dir("v")
    run_dir.mkdir(parents=True)
    f.record_held_out(run_dir, args, 1)
    rows = [{"video": "a", "type": "t", "split": "dev"}]
    (tmp_path / "sets" / "s.json").write_text(f.json.dumps(rows))
    args.split = "dev"
    assert f.cmd_system_eval(args) == 0
