import os
import shutil
import tomllib

import faithfulness as f
import pytest

# CI sets REQUIRE_ARVIO, so a missing arvio fails these tests there instead of
# skipping them.
needs_arvio = pytest.mark.skipif(
    shutil.which("arvio") is None and not os.environ.get("REQUIRE_ARVIO"),
    reason="arvio is not installed",
)

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
        "model": "claude-opus-5-5",
        "usage": dict.fromkeys(
            [
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            ],
            0,
        ),
        "claims_checked": 10,
        "checks": {"errors": [], "advisory": 0},
        "issues": issues,
    }


def _key(quote: str, summary: str = SUMMARY, video: str = "2026-01-01_x") -> str:
    return f.issue_key(video, summary.splitlines(), quote)


def test_flag_key_matches_reps_and_keeps_lines_apart():
    line = "- Point [00:00:32]."
    assert (
        f.flag_key("2026-01-01_x", line, "Point [00:00:32].") == "vs1-8a00801493cbbfae"
    )
    key = f.flag_key("2026-01-01_x", "1. **Start** low", "**Start** low")
    assert key == f.flag_key("2026-01-01_x", "1. start  LOW", "start  LOW")
    assert key != f.flag_key("2026-01-02_y", "1. start low", "start low")
    assert f.flag_key("v", "a [00:00:32]", "[00:00:32]") != f.flag_key(
        "v", "b [00:00:32]", "[00:00:32]"
    )
    assert _key("") is None and _key("  ") is None
    assert _key("Not in it.") == f.flag_key("2026-01-01_x", "", "Not in it.")


def test_label_rules_copy_the_judge_wording():
    rules = " ".join(f.LABEL_RULES)
    names = []
    for line in f.JUDGE_INSTRUCTIONS.splitlines():
        if line.startswith("- "):
            name, _, meaning = line[2:].partition(": ")
            names.append(name)
            assert f"{name.replace('_', ' ')} ({meaning.rstrip('.')})" in rules
        elif line.startswith("Not errors:"):
            assert line in rules
    issue = f.ISSUE_SCHEMA["properties"]["issues"]["items"]["properties"]
    assert names == issue["category"]["enum"]


def test_label_card_parses_and_pins_the_prompt_and_rules(monkeypatch):
    card = tomllib.loads(f.label_card())
    assert [c["key"] for c in card["choices"]] == ["real", "false_alarm", "cant_tell"]
    hotkeys = [c["hotkey"] for c in card["choices"]] + [
        o["hotkey"] for o in card["followups"][0]["options"]
    ]
    assert len(set(hotkeys)) == len(hotkeys) and "n" not in hotkeys
    assert card["rules"] == f.LABEL_RULES
    issue = f.ISSUE_SCHEMA["properties"]["issues"]["items"]["properties"]
    assert [o["key"] for o in card["followups"][0]["options"]] == issue["severity"][
        "enum"
    ]
    pin = f.label_pin()
    assert card["pin"] == pin
    monkeypatch.setattr(f, "LABEL_RULES", [*f.LABEL_RULES, "One more rule."])
    assert f.label_pin() != pin
    monkeypatch.setattr(f, "JUDGE_INSTRUCTIONS", f.JUDGE_INSTRUCTIONS + "x")
    assert f.label_pin() != pin
    monkeypatch.setattr(f, "LABEL_RULES", ["Emoji \U0001f600 rule."])
    assert tomllib.loads(f.label_card())["rules"] == ["Emoji \U0001f600 rule."]


def test_transcript_windows_merge_neighbours_and_split_far_ones():
    text = "".join(
        f"[00:{m:02d}:{s:02d}]\npart {i}\n\n"
        for i, (m, s) in enumerate([(0, 10), (0, 30), (3, 20), (3, 50), (6, 40)])
    )
    chunks = f.transcript_chunks(text)
    assert [start for start, _ in chunks] == [10, 30, 200, 230, 400]
    assert chunks[1][1] == "[00:00:30]\npart 1"
    assert f.transcript_windows(chunks, [30, 400]) == [[0, 1], [3, 4]]
    assert f.transcript_windows(chunks, []) == []
    assert f.transcript_windows(chunks, [0]) == [[0, 1]]
    assert f.transcript_windows(chunks, [9000]) == [[4]]
    assert f.transcript_windows(f.transcript_chunks("no markers\n"), [30]) == []


def test_flag_item_shows_the_section_and_hides_the_judge():
    saved = {"title": "T", "duration": "00:01:00"}
    secret = _issue(
        explanation="SENTINEL-EXPLANATION", transcript_quote="SENTINEL-TQUOTE"
    )
    item = f.flag_item(_row(0, []), secret, SUMMARY, TRANSCRIPT, saved, "base")
    assert item["id"] == _key(QUOTE)
    shown = item["display"][1]
    assert shown["title"] == "Flagged in the summary"
    assert shown["text"].startswith("## Actionable summary")
    assert shown["text"].endswith("~2.5 minutes [00:00:32].")
    assert shown["highlight"] == [QUOTE]
    window = item["display"][2]
    assert window["title"] == "Transcript [00:00:02] to [00:00:32]"
    assert "45 seconds" in window["text"]
    assert [b.get("collapsed") for b in item["display"][3:]] == [True, True]
    assert item["judge"]["verdict"] == "error"
    display = f.json.dumps(item["display"])
    assert "SENTINEL" not in display and "how-to" not in display

    loose = f.flag_item(
        _row(0, []), _issue("start LOW on the layers"), SUMMARY, TRANSCRIPT, saved, "b"
    )
    assert loose["display"][1]["highlight"] == ["Start low on the layers"]
    entity = f.flag_item(
        _row(0, []), _issue("2. Sonnet took &#126;45"), SUMMARY, TRANSCRIPT, saved, "b"
    )
    assert entity["display"][1]["highlight"] == ["2. Sonnet took ~45"]
    gone = f.flag_item(
        _row(0, []), _issue("Not in it."), SUMMARY, TRANSCRIPT, saved, "b"
    )
    assert gone["display"][1]["text"] == "Not in it."
    assert gone["display"][1]["title"].endswith("(not found in it word for word)")
    twice = f.flag_item(
        _row(0, []), _issue("[00:00:32]"), SUMMARY, TRANSCRIPT, saved, "b"
    )
    assert twice["display"][1]["title"].endswith("(on 2 lines; the first shown)")


def test_flag_item_windows_follow_the_judge_time():
    transcript = "".join(f"[00:{m:02d}:00]\npart {m}\n\n" for m in range(0, 20, 2))
    saved = {"title": "T", "duration": "00:20:00"}

    def titles(quote: str, ts: str) -> list[str]:
        item = f.flag_item(
            _row(0, []),
            _issue(quote, transcript_ts=ts),
            SUMMARY,
            transcript,
            saved,
            "b",
        )
        return [b["title"] for b in item["display"][2:-2]]

    assert titles("Short.", "") == []
    assert titles("Short.", "[00:10:00]") == ["Transcript [00:08:00] to [00:10:00]"]
    assert titles("Short.", "00:04:00-00:16:00") == [
        "Transcript [00:02:00] to [00:04:00]",
        "Transcript [00:14:00] to [00:16:00]",
    ]
    assert titles("Point [00:00:32].", "00:16:00") == [
        "Transcript [00:00:00]",
        "Transcript [00:14:00] to [00:16:00]",
    ]


def test_marked_span_keeps_the_markdown_the_judge_dropped():
    line = "1. **Start** low on the layers [00:00:02]. Then add `X`."
    assert f.marked_span(line, "Start low on the layers") == (
        "**Start** low on the layers"
    )
    assert f.marked_span(line, "start  LOW") == "**Start** low"
    assert f.marked_span(line, "then add x.") == "Then add `X`."
    assert f.marked_span(line, "Not in it") is None
    assert f.marked_span(line, "**") is None
    # An entity in the summary line itself: flag_item marks the whole line.
    assert f.marked_span("x &gt; y", "x > y") is None


def test_every_highlight_is_an_exact_part_of_its_block():
    saved = {"title": "T", "duration": "00:01:00"}
    quotes = [
        QUOTE,
        "start LOW on the layers",
        "Start low on the layers [00:00:02].\n2. Sonnet took",
        "  Point [00:00:32].  ",
        "Not in it.",
        "[00:00:32]",
        "Short",
        "**Short.**",
    ]
    for quote in quotes:
        item = f.flag_item(_row(0, []), _issue(quote), SUMMARY, TRANSCRIPT, saved, "b")
        for block in item["display"]:
            for h in block.get("highlight", []):
                assert h and h in block["text"], quote


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


def test_label_export_writes_one_item_per_flagged_line(tmp_path, monkeypatch, capsys):
    other = _issue("Point [00:00:32].", transcript_ts="[00:00:32]")
    rows = [_row(0, [_issue(), other]), _row(1, [_issue()]), _row(2, [])]
    args = _export_setup(tmp_path, monkeypatch, rows)
    assert f.cmd_label_export(args) == 0
    task = f.label_task("base", "dev")
    assert (task / "task.toml").read_text() == f.label_card()
    first = (task / "items.jsonl").read_bytes()
    ids = [i["id"] for i in f.read_jsonl(task / "items.jsonl")]
    assert ids == [_key(QUOTE), _key("Point [00:00:32].")]
    assert (
        "3 judge flags in 3 summaries -> 2 items (1 repeats" in capsys.readouterr().out
    )
    f.cmd_label_export(args)
    assert (task / "items.jsonl").read_bytes() == first
    assert {p.name for p in task.iterdir()} == {"items.jsonl", "task.toml"}


def test_label_export_keeps_a_quote_on_two_lines_apart(tmp_path, monkeypatch, capsys):
    rows = [_row(0, [_issue("[00:00:32]")]), _row(1, [_issue("[00:00:32]")])]
    args = _export_setup(tmp_path, monkeypatch, rows)
    moved = SUMMARY.replace("~2.5 minutes [00:00:32]", "~2.5 minutes [00:00:40]")
    (f.gen_dir("base") / "2026-01-01_x" / "rep1" / "summary.md").write_text(moved)
    f.cmd_label_export(args)
    ids = [i["id"] for i in f.read_jsonl(f.label_task("base", "dev") / "items.jsonl")]
    assert ids == [_key("[00:00:32]"), _key("[00:00:32]", moved)]
    assert ids[0] != ids[1]


def test_label_export_leaves_out_empty_quotes(tmp_path, monkeypatch, capsys):
    rows = [_row(0, [_issue(""), _issue("   "), _issue()])]
    f.cmd_label_export(_export_setup(tmp_path, monkeypatch, rows))
    out = capsys.readouterr().out
    assert "-> 1 items" in out
    assert "left out 2 flags with an empty summary_quote" in out
    items = f.read_jsonl(f.label_task("base", "dev") / "items.jsonl")
    assert [i["id"] for i in items] == [_key(QUOTE)]


def test_label_export_keeps_labeled_items_and_skips_labels_elsewhere(
    tmp_path, monkeypatch, capsys
):
    other = _issue("Point [00:00:32].")
    args = _export_setup(tmp_path, monkeypatch, [_row(0, [_issue(), other])])
    f.cmd_label_export(args)
    task = f.label_task("base", "dev")
    items = f.read_jsonl(task / "items.jsonl")
    items[0]["display"] = [{"kind": "text", "text": "as the labeler saw it "}]
    (task / "items.jsonl").write_text(
        "".join(f.json.dumps(i, ensure_ascii=False) + "\n" for i in items)
    )
    (task / "labels.jsonl").write_text(
        f.json.dumps({"kind": "label", "item_id": items[0]["id"]}) + "\n{torn"
    )
    counts = [_label(items[0]["id"], "real")]
    monkeypatch.setattr(f, "read_labels", lambda t, quiet=False: counts)
    f.cmd_label_export(args)
    again = f.read_jsonl(task / "items.jsonl")
    assert again[0] == items[0]
    assert again[1]["display"][1]["highlight"] == ["Point [00:00:32]."]

    labeled = _label(items[1]["id"], "real")
    monkeypatch.setattr(f, "read_labels", lambda t: [labeled] if t == task else [])
    v2 = _export_setup(tmp_path, monkeypatch, [_row(0, [_issue(), other])], "v2")
    capsys.readouterr()
    f.cmd_label_export(v2)
    assert [
        i["id"] for i in f.read_jsonl(f.label_task("v2", "dev") / "items.jsonl")
    ] == [items[0]["id"]]
    assert "1 already labeled in another label folder" in capsys.readouterr().out


def test_label_export_rewrites_items_whose_label_no_longer_counts(
    tmp_path, monkeypatch, capsys
):
    other = _issue("Point [00:00:32].")
    args = _export_setup(tmp_path, monkeypatch, [_row(0, [_issue(), other])])
    f.cmd_label_export(args)
    task = f.label_task("base", "dev")
    items = f.read_jsonl(task / "items.jsonl")
    for item in items:
        item["display"] = [{"kind": "text", "text": "an older display"}]
    (task / "items.jsonl").write_text(f.jsonl(items))
    (task / "labels.jsonl").write_text(
        "".join(
            f.json.dumps({"kind": "label", "item_id": i["id"]}) + "\n" for i in items
        )
    )
    v2 = f.label_task("v2", "dev")
    v2.mkdir(parents=True)
    (v2 / "task.toml").write_text(f.label_card())
    later = _label(items[1]["id"], "real")

    def arvio(t, quiet=False):
        # The labels in base were made under an older card: none count.
        return [later] if t == v2 else []

    monkeypatch.setattr(f, "read_labels", arvio)
    capsys.readouterr()
    f.cmd_label_export(args)
    again = f.read_jsonl(task / "items.jsonl")
    assert [i["id"] for i in again] == [items[0]["id"]]
    assert again[0]["display"][1]["highlight"] == [QUOTE]
    assert "1 already labeled in another label folder" in capsys.readouterr().out


def test_label_export_refuses_when_labels_change_meanwhile(
    tmp_path, monkeypatch, capsys
):
    args = _export_setup(tmp_path, monkeypatch, [_row(0, [_issue()])])
    f.cmd_label_export(args)
    task = f.label_task("base", "dev")
    key = _key(QUOTE)
    (task / "labels.jsonl").write_text(
        f.json.dumps({"kind": "label", "item_id": key}) + "\n"
    )

    def arvio_meanwhile(t, quiet=False):
        with (task / "labels.jsonl").open("a") as fh:
            fh.write(f.json.dumps({"kind": "label", "item_id": key}) + "\n")
        return [_label(key, "real")]

    monkeypatch.setattr(f, "read_labels", arvio_meanwhile)
    (task / "task.toml").write_text("format = 1\n")
    before = {p.name: p.read_bytes() for p in task.iterdir()}
    with pytest.raises(SystemExit, match="changed while exporting"):
        f.cmd_label_export(args)
    after = {p.name: p.read_bytes() for p in task.iterdir()}
    assert after["items.jsonl"] == before["items.jsonl"]
    assert after["task.toml"] == before["task.toml"]


def test_label_export_without_arvio_gives_only_the_install_advice(
    tmp_path, monkeypatch, capsys
):
    args = _export_setup(tmp_path, monkeypatch, [_row(0, [_issue()])])
    f.cmd_label_export(args)
    task = f.label_task("base", "dev")
    (task / "labels.jsonl").write_text(
        f.json.dumps({"kind": "label", "item_id": _key(QUOTE)}) + "\n"
    )

    def missing(*a, **kw):
        raise FileNotFoundError("arvio")

    monkeypatch.setattr(f.subprocess, "run", missing)
    with pytest.raises(SystemExit) as done:
        f.cmd_label_export(args)
    message = str(done.value)
    assert "not on PATH" in message and "Nothing was written" in message
    assert "restore items.jsonl" not in message


def test_label_export_keeps_a_gone_flag_only_while_its_label_counts(
    tmp_path, monkeypatch, capsys
):
    other = _issue("Point [00:00:32].")
    args = _export_setup(tmp_path, monkeypatch, [_row(0, [_issue(), other])])
    f.cmd_label_export(args)
    task = f.label_task("base", "dev")
    gone = _key("Point [00:00:32].")
    (task / "labels.jsonl").write_text(
        f.json.dumps({"kind": "label", "item_id": gone}) + "\n"
    )
    (task / "task.toml").write_text(f.label_card().replace("Video summary", "Old"))
    _export_setup(tmp_path, monkeypatch, [_row(0, [_issue()])])
    counted = [_label(gone, "real", severity="minor")]

    def arvio(copy, quiet=False):
        assert quiet and copy.parent == task.parent.resolve()
        assert (copy / "task.toml").read_text() == f.label_card()
        assert (copy / "labels.jsonl").read_bytes() == (
            task / "labels.jsonl"
        ).read_bytes()
        assert gone in {i["id"] for i in f.read_jsonl(copy / "items.jsonl")}
        return counted

    monkeypatch.setattr(f, "read_labels", arvio)
    capsys.readouterr()
    f.cmd_label_export(args)
    out = capsys.readouterr().out
    assert f"updated the card in {task}. {f.CARD_CHANGE_NOTE}" in out
    assert "kept 1 labeled items whose flag is gone" in out
    assert [i["id"] for i in f.read_jsonl(task / "items.jsonl")] == [_key(QUOTE), gone]

    counted.clear()
    f.cmd_label_export(args)
    assert "dropped 1 items whose flag is gone" in capsys.readouterr().out
    assert [i["id"] for i in f.read_jsonl(task / "items.jsonl")] == [_key(QUOTE)]
    assert not list(task.parent.glob(".label-check-*"))

    def refuses(copy, quiet=False):
        raise f.LabelError(f"{copy / 'labels.jsonl'}:1: error: item_id: '{gone}'")

    monkeypatch.setattr(f, "read_labels", refuses)
    before = {p.name: p.read_bytes() for p in task.iterdir()}
    error = f"{task / 'labels.jsonl'}:1: error: item_id: '{gone}'\nNothing was written"
    with pytest.raises(SystemExit, match=f.re.escape(error)):
        f.cmd_label_export(args)
    assert {p.name: p.read_bytes() for p in task.iterdir()} == before
    assert not list(task.parent.glob(".label-check-*"))


def _label(key: str, answer: str, locked: bool = True, severity: str = "") -> dict:
    return {
        "item_id": key,
        "answer": answer,
        "followups": {"severity": severity} if severity else {},
        "labeled_at": "2026-10-18T10:00:00Z",
        "locked": locked,
    }


def test_current_labels_prefers_locked_then_latest(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(f, "EVAL_DIR", tmp_path)
    a, b = f.label_task("a", "dev"), f.label_task("b", "held-out")
    stale = f.label_task("c", "dev")
    odd = f.system_dir("a") / "label-copy"
    for task in (a, b, stale, odd):
        task.mkdir(parents=True)
        (task / "task.toml").write_text(f.label_card())
    (stale / "task.toml").write_text("format = 1\n")
    later = "2026-10-19T10:00:00Z"
    rows = {
        a: [_label("k1", "real"), _label("k2", "real"), _label("k3", "real")],
        b: [
            _label("k1", "false_alarm", locked=False) | {"labeled_at": later},
            _label("k2", "false_alarm") | {"labeled_at": later},
            {"item_id": "k3", "answer": "real", "locked": True},
        ],
    }

    def fake(task):
        assert task != stale, "a folder with an older card must not be read"
        assert task != odd, "only label-dev and label-held-out are label folders"
        return rows[task]

    monkeypatch.setattr(f, "read_labels", fake)
    labels, old, mixed = f.current_labels()
    assert labels["k1"]["answer"] == "real"
    assert labels["k2"]["answer"] == "false_alarm"
    assert labels["k3"]["labeled_at"] == "2026-10-18T10:00:00Z"
    assert old == [stale]
    assert mixed == {"k1", "k2"}
    assert f.export_args(stale) == "--variant c --split dev"


def test_report_labels_counts_finished_labels_and_precision(
    tmp_path, monkeypatch, capsys
):
    quotes = ["Short.", "Start low", "Sonnet took", "Point [00:00:32].", "Intro."]
    part = [
        _row(0, [_issue(q) for q in quotes[:3]]),
        _row(1, [_issue(q) for q in quotes[3:]]),
    ]
    _export_setup(tmp_path, monkeypatch, part)
    keys = [_key(q) for q in quotes]
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
    monkeypatch.setattr(f, "read_labels", lambda t: labels)
    f.report_labels(part, "base", "dev")
    out = capsys.readouterr().out
    assert "has an older card; run label-export --variant old --split dev" in out
    assert "flagged lines labeled: 3/5 (real 2, false alarm 1, can't tell 0)" in out
    assert "1 more labeled in an arvio session not finished yet" in out
    lo, hi = f.wilson(2, 3)
    assert f"judge precision on labeled lines: 2/3 (95% CI {lo:.2f}-{hi:.2f}" in out
    assert "real errors per summary" not in out
    assert "label them" not in out

    labels[3]["locked"] = True
    labels.append(_label(keys[4], "cant_tell"))
    f.report_labels(part, "base", "dev")
    out = capsys.readouterr().out
    assert "real errors per summary: mean 1.50" in out
    assert "no real major error: 1/2" in out
    assert "can't tell counted as no error" in out


def test_report_labels_counts_a_repeated_line_once(tmp_path, monkeypatch, capsys):
    part = [_row(0, [_issue(), _issue()]), _row(1, [_issue()])]
    _export_setup(tmp_path, monkeypatch, part)
    label = _label(_key(QUOTE), "real") | {"followups": None}
    del label["labeled_at"]
    monkeypatch.setattr(f, "read_labels", lambda t: [label])
    f.label_task("base", "dev").mkdir(parents=True)
    (f.label_task("base", "dev") / "task.toml").write_text(f.label_card())
    f.report_labels(part, "base", "dev")
    out = capsys.readouterr().out
    assert "flagged lines labeled: 1/1" in out
    assert "from 3 flags: a line flagged in more than one summary counts once" in out
    assert "judge precision on labeled lines: 1/1" in out
    assert "real errors per summary: mean 1.00" in out
    assert "no real major error: 2/2" in out


def test_report_labels_says_why_a_count_is_missing(tmp_path, monkeypatch, capsys):
    part = [_row(0, [_issue(), _issue("")])]
    _export_setup(tmp_path, monkeypatch, part)
    monkeypatch.setattr(f, "read_labels", lambda t: [_label(_key(QUOTE), "real")])
    f.label_task("base", "dev").mkdir(parents=True)
    (f.label_task("base", "dev") / "task.toml").write_text(f.label_card())
    other = f.label_task("v2", "dev")
    other.mkdir(parents=True)
    (other / "task.toml").write_text(f.label_card())
    f.report_labels(part, "base", "dev")
    out = capsys.readouterr().out
    assert "flagged lines labeled: 1/1" in out
    assert "real errors per summary not shown: the flags with an empty" in out
    assert "different answers" not in out

    rows = {other: [_label(_key(QUOTE), "false_alarm")]}
    monkeypatch.setattr(f, "read_labels", lambda t: rows.get(t, []))
    f.report_labels(part, "base", "dev")
    assert "different answers" not in capsys.readouterr().out
    rows[f.label_task("base", "dev")] = [_label(_key(QUOTE), "real")]
    f.report_labels(part, "base", "dev")
    assert "1 flagged lines have different answers in two label folders" in (
        capsys.readouterr().out
    )


def test_report_labels_says_how_to_start(tmp_path, monkeypatch, capsys):
    part = [_row(0, [_issue(), _issue("")])]
    held = _row(1, [_issue()]) | {"split": "held-out"}
    _export_setup(tmp_path, monkeypatch, [*part, held])
    f.label_task("base", "held-out").mkdir(parents=True)
    f.report_labels(part, "base", "all")
    out = capsys.readouterr().out
    assert "flagged lines labeled: 0/1" in out
    assert "1 flags with an empty summary_quote can't be labeled" in out
    assert "label-export --variant base --split dev" in out
    f.label_task("base", "held-out").rmdir()
    f.report_labels([*part, held], "base", "all")
    out = capsys.readouterr().out
    assert "label-export --variant base --split dev" in out
    assert "label-export --variant base --split held-out" in out


def test_system_report_prints_spend_before_a_label_failure(
    tmp_path, monkeypatch, capsys
):
    args = _export_setup(tmp_path, monkeypatch, [_row(0, [_issue()])])
    task = f.label_task("base", "dev")
    task.mkdir(parents=True)
    (task / "task.toml").write_text(f.label_card())

    def broken(t):
        raise f.LabelError("arvio is not installed or not on PATH")

    monkeypatch.setattr(f, "read_labels", broken)
    assert f.cmd_system_report(args) == 1
    out = capsys.readouterr().out
    assert out.index("measured spend") < out.index("labels not counted: arvio is not")


def test_read_labels_parses_and_fails_closed(tmp_path, monkeypatch, capsys):
    def missing(*a, **kw):
        raise FileNotFoundError("arvio")

    monkeypatch.setattr(f.subprocess, "run", missing)
    with pytest.raises(f.ArvioMissing, match="not on PATH"):
        f.read_labels(tmp_path)
    failed = f.subprocess.CompletedProcess([], 1, "", "task.toml: bad")
    monkeypatch.setattr(f.subprocess, "run", lambda *a, **kw: failed)
    with pytest.raises(f.LabelError, match="task.toml: bad"):
        f.read_labels(tmp_path)
    row = _label("k1", "real") | {"note": "a b"}
    out = f.json.dumps(row, ensure_ascii=False) + "\n"
    ok = f.subprocess.CompletedProcess([], 0, out, "labels.jsonl:3: old card\n")
    calls = []
    monkeypatch.setattr(f.subprocess, "run", lambda *a, **kw: calls.append(a) or ok)
    assert f.read_labels(tmp_path) == [row]
    assert calls == [(["arvio", "labels", str(tmp_path)],)]
    assert "labels.jsonl:3: old card" in capsys.readouterr().err


def test_jsonl_keeps_a_line_separator_inside_a_row(tmp_path):
    rows = [{"id": "a", "text": "one\u2028two\u0085three"}, {"id": "b"}]
    (tmp_path / "items.jsonl").write_text(f.jsonl(rows), encoding="utf-8")
    assert f.read_jsonl(tmp_path / "items.jsonl") == rows
    label = {"kind": "label", "item_id": "a", "note": "x\u2028y"}
    (tmp_path / "labels.jsonl").write_text(f.jsonl([label]), encoding="utf-8")
    assert f.labeled_ids(tmp_path / "labels.jsonl") == {"a"}


def test_system_eval_succeeds_when_only_the_labels_fail(tmp_path, monkeypatch):
    args = _held_out_setup(tmp_path, monkeypatch)
    rows = [{"video": "a", "type": "t", "split": "dev"}]
    (tmp_path / "sets" / "s.json").write_text(f.json.dumps(rows))
    args.split, args.dry_run = "dev", False
    args.effort, args.timeout_s, args.workers = "medium", 1, 1
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(f, "run_parallel", lambda *a, **kw: None)
    reports = []
    monkeypatch.setattr(f, "cmd_system_report", lambda a: reports.append(a) or 1)
    assert f.cmd_system_eval(args) == 0
    assert [(a.variant, a.split) for a in reports] == [("v", "dev")]


@needs_arvio
def test_arvio_accepts_the_exported_folder(tmp_path, monkeypatch, capsys):
    rows = [
        _row(
            0,
            [
                _issue(),
                _issue("start LOW on the layers"),
                _issue("Not in it."),
                _issue("[00:00:32]", transcript_ts="00:00:02-00:00:32"),
            ],
        )
    ]
    f.cmd_label_export(_export_setup(tmp_path, monkeypatch, rows))
    done = _arvio_check(tmp_path, f.label_task("base", "dev"))
    assert done.returncode == 0, done.stderr


def _arvio_check(tmp_path, task):
    return f.subprocess.run(
        ["arvio", "check", str(task)],
        capture_output=True,
        text=True,
        env=f.os.environ | {"ARVIO_DATA_DIR": str(tmp_path / "arvio")},
        check=False,
    )


@needs_arvio
def test_arvio_decides_which_gone_flags_stay(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ARVIO_DATA_DIR", str(tmp_path / "arvio"))
    other = _issue("Point [00:00:32].")
    args = _export_setup(tmp_path, monkeypatch, [_row(0, [_issue(), other])])
    f.cmd_label_export(args)
    task = f.label_task("base", "dev")
    version = f.re.search(r"card version (\w+)", _arvio_check(tmp_path, task).stdout)
    gone = _key("Point [00:00:32].")
    label = {
        "kind": "label",
        "item_id": gone,
        "task_version": version.group(1),
        "answer": "real",
        "followups": {"severity": "minor"},
    }
    (task / "labels.jsonl").write_text(f.json.dumps(label) + "\n")
    _export_setup(tmp_path, monkeypatch, [_row(0, [_issue()])])
    capsys.readouterr()
    f.cmd_label_export(args)
    assert "kept 1 labeled items whose flag is gone" in capsys.readouterr().out
    assert [i["id"] for i in f.read_jsonl(task / "items.jsonl")] == [_key(QUOTE), gone]
    assert _arvio_check(tmp_path, task).returncode == 0

    label["task_version"] = "000000000000"
    (task / "labels.jsonl").write_text(f.json.dumps(label) + "\n")
    f.cmd_label_export(args)
    assert "dropped 1 items whose flag is gone" in capsys.readouterr().out
    assert [i["id"] for i in f.read_jsonl(task / "items.jsonl")] == [_key(QUOTE)]
    done = _arvio_check(tmp_path, task)
    assert done.returncode == 0, done.stderr
    assert "ok: 1 items, 0 labeled" in done.stdout

    label["task_version"] = version.group(1)
    (task / "labels.jsonl").write_text(f.json.dumps(label) + "\n")
    with pytest.raises(SystemExit, match="Nothing was written"):
        f.cmd_label_export(args)
    assert not list(task.parent.glob(".label-check-*"))


@needs_arvio
def test_report_reads_real_arvio_labels(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ARVIO_DATA_DIR", str(tmp_path / "arvio"))
    rows = [_row(0, [_issue(), _issue("Point [00:00:32].")])]
    f.cmd_label_export(_export_setup(tmp_path, monkeypatch, rows))
    task = f.label_task("base", "dev")
    version = f.re.search(r"card version (\w+)", _arvio_check(tmp_path, task).stdout)
    v = version.group(1)
    keys = [_key(QUOTE), _key("Point [00:00:32].")]
    log = [
        {
            "kind": "label",
            "item_id": keys[0],
            "task_version": v,
            "answer": "real",
            "followups": {"severity": "major"},
        },
        {
            "kind": "label",
            "item_id": keys[1],
            "task_version": v,
            "answer": "false_alarm",
        },
        {"kind": "reveal", "item_ids": keys[:1], "task_version": v},
    ]
    (task / "labels.jsonl").write_text(f.jsonl(log))
    capsys.readouterr()
    f.report_labels(rows, "base", "dev")
    out = capsys.readouterr().out
    assert "flagged lines labeled: 1/2 (real 1, false alarm 0" in out
    assert "1 more labeled in an arvio session not finished yet" in out
    log.append({"kind": "reveal", "item_ids": keys[1:], "task_version": v})
    (task / "labels.jsonl").write_text(f.jsonl(log))
    f.report_labels(rows, "base", "dev")
    out = capsys.readouterr().out
    assert "flagged lines labeled: 2/2 (real 1, false alarm 1" in out
    assert "real errors per summary: mean 1.00" in out
    assert "no real major error: 0/1" in out


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
