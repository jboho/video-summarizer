import importlib.util
import os
import shutil
import sys
import tomllib

import faithfulness as f
import pytest

# CI sets REQUIRE_ARVIO=1, so a missing arvio fails these tests there instead of
# skipping them.
needs_arvio = pytest.mark.skipif(
    shutil.which("arvio") is None and os.environ.get("REQUIRE_ARVIO") != "1",
    reason="arvio is not installed",
)
# The lock test imports arvio's own lock, so it needs the package, not only the
# command.
needs_arvio_package = pytest.mark.skipif(
    importlib.util.find_spec("arvio") is None
    and os.environ.get("REQUIRE_ARVIO") != "1",
    reason="the arvio package is not importable",
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
    # Labels are stored under these ids, so the normalization is pinned too.
    messy = "&gt;&gt; **Keep  GPU** `low` Ａ"
    assert f.key_norm(messy) == ">> keep gpu low a"
    assert f.flag_key("2026-01-01_x", "- " + messy, messy) == "vs1-73ca159efcf65ab6"


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
                assert h and block["text"].count(h) == 1, quote


def test_a_line_that_recurs_in_its_section_is_shown_alone():
    saved = {"title": "T", "duration": "00:01:00"}
    for second in ["- Use 8 GPU layers. It crashed.", "- Use 8 GPU layers."]:
        summary = SUMMARY.replace(
            "- Point [00:00:32].", "- Use 8 GPU layers.\n" + second
        )
        item = f.flag_item(
            _row(0, []), _issue("8 gpu layers"), summary, TRANSCRIPT, saved, "b"
        )
        block = item["display"][1]
        assert block["text"] == "- Use 8 GPU layers."
        assert block["highlight"] == ["- Use 8 GPU layers."]


def test_a_quote_that_recurs_in_its_section_marks_the_whole_line():
    summary = SUMMARY.replace(
        "- Point [00:00:32].",
        "- Use 8 GPU layers on the 3090 [00:00:02].\n"
        "- With 8 GPU layers it crashed [00:00:32].",
    )
    saved = {"title": "T", "duration": "00:01:00"}
    item = f.flag_item(
        _row(0, []), _issue("8 gpu layers"), summary, TRANSCRIPT, saved, "b"
    )
    block = item["display"][1]
    assert block["highlight"] == ["- Use 8 GPU layers on the 3090 [00:00:02]."]
    assert block["text"].count(block["highlight"][0]) == 1


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
    return f.argparse.Namespace(variant=variant, split="dev", sample=None, seed="1")


def test_label_export_writes_one_item_per_flagged_quote(tmp_path, monkeypatch, capsys):
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


def test_label_export_refuses_while_arvio_serves_the_folder(tmp_path, monkeypatch):
    fcntl = pytest.importorskip("fcntl")
    args = _export_setup(tmp_path, monkeypatch, [_row(0, [_issue()])])
    f.cmd_label_export(args)
    task = f.label_task("base", "dev")
    (task / "items.jsonl").unlink()
    # flock locks belong to an open file, so this conflicts with the export's
    # own open of the folder, as arvio's server lock does.
    fd = f.os.open(task, f.os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit, match="arvio or another label-export"):
            f.cmd_label_export(args)
    finally:
        f.os.close(fd)
    assert not (task / "items.jsonl").exists()
    f.cmd_label_export(args)
    assert (task / "items.jsonl").exists()


@needs_arvio_package
def test_label_export_refuses_a_folder_arvio_has_locked(tmp_path, monkeypatch):
    from arvio import store

    args = _export_setup(tmp_path, monkeypatch, [_row(0, [_issue()])])
    f.cmd_label_export(args)
    task = f.label_task("base", "dev")
    (task / "items.jsonl").unlink()
    with store.locked(task, wait=False):
        with pytest.raises(SystemExit, match="arvio or another label-export"):
            f.cmd_label_export(args)
    assert not (task / "items.jsonl").exists()


def test_label_export_locks_a_folder_it_creates(tmp_path, monkeypatch):
    args = _export_setup(tmp_path, monkeypatch, [_row(0, [_issue()])])
    task = f.label_task("base", "dev")
    assert not task.exists()
    held = []

    fcntl = pytest.importorskip("fcntl")

    def export(args, folder, part):
        fd = os.open(folder, os.O_RDONLY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            held.append(folder)
        finally:
            os.close(fd)
        return 0

    monkeypatch.setattr(f, "export_labels", export)
    assert f.cmd_label_export(args) == 0
    assert held == [task]


def test_label_export_without_fcntl_leaves_no_folder(tmp_path, monkeypatch):
    args = _export_setup(tmp_path, monkeypatch, [_row(0, [_issue()])])
    monkeypatch.setitem(sys.modules, "fcntl", None)
    with pytest.raises(ImportError):
        f.cmd_label_export(args)
    assert not f.label_task("base", "dev").exists()


def test_label_export_of_an_unknown_variant_leaves_no_folder(tmp_path, monkeypatch):
    _export_setup(tmp_path, monkeypatch, [_row(0, [_issue()])])
    args = f.argparse.Namespace(variant="typo", split="dev", sample=None, seed="1")
    with pytest.raises(SystemExit, match="no judged summaries for 'typo'"):
        f.cmd_label_export(args)
    assert not f.system_dir("typo").exists()


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
        a: [
            _label("k1", "real"),
            _label("k2", "real"),
            _label("k3", "real"),
            _label("k4", "real", severity="minor"),
            _label("k5", "real", severity="major"),
        ],
        b: [
            _label("k1", "false_alarm", locked=False) | {"labeled_at": later},
            _label("k2", "false_alarm") | {"labeled_at": later},
            {"item_id": "k3", "answer": "real", "locked": True},
            _label("k4", "real", severity="major"),
            _label("k5", "real", severity="major"),
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
    assert mixed == {"k1", "k2", "k4"}
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
    assert "flagged quotes labeled: 3/5 (real 2, false alarm 1, can't tell 0)" in out
    assert "1 more labeled in an arvio session not finished yet" in out
    lo, hi = f.wilson(2, 3)
    assert f"judge precision on labeled quotes: 2/3 (95% CI {lo:.2f}-{hi:.2f}" in out
    assert "real errors per summary: mean" not in out
    assert "not shown: no sample drawn: run label-export --sample N" in out
    assert "label them" not in out

    labels[3]["locked"] = True
    labels.append(_label(keys[4], "cant_tell"))
    f.report_labels(part, "base", "dev")
    out = capsys.readouterr().out
    assert "real errors per summary: mean 1.50" in out
    assert "no real major error: 1/2" in out
    assert "can't tell counted as no error" in out


def test_report_labels_counts_a_repeated_quote_once(tmp_path, monkeypatch, capsys):
    part = [_row(0, [_issue(), _issue()]), _row(1, [_issue()])]
    _export_setup(tmp_path, monkeypatch, part)
    label = _label(_key(QUOTE), "real") | {"followups": None}
    del label["labeled_at"]
    monkeypatch.setattr(f, "read_labels", lambda t: [label])
    f.label_task("base", "dev").mkdir(parents=True)
    (f.label_task("base", "dev") / "task.toml").write_text(f.label_card())
    f.report_labels(part, "base", "dev")
    out = capsys.readouterr().out
    assert "flagged quotes labeled: 1/1" in out
    assert (
        "from 3 flags: the same quote on the same line, flagged in more than "
        "one summary, counts once" in out
    )
    assert "judge precision on labeled quotes: 1/1" in out
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
    assert "flagged quotes labeled: 1/1" in out
    assert "real errors per summary not shown: the flags with an empty" in out
    assert "different answer" not in out

    rows = {other: [_label(_key(QUOTE), "false_alarm")]}
    monkeypatch.setattr(f, "read_labels", lambda t: rows.get(t, []))
    f.report_labels(part, "base", "dev")
    assert "different answer" not in capsys.readouterr().out
    rows[f.label_task("base", "dev")] = [_label(_key(QUOTE), "real")]
    f.report_labels(part, "base", "dev")
    assert (
        "1 flagged quotes have a different answer or severity in two label folders"
        in (capsys.readouterr().out)
    )


def test_report_labels_says_how_to_start(tmp_path, monkeypatch, capsys):
    part = [_row(0, [_issue(), _issue("")])]
    held = _row(1, [_issue()]) | {"split": "held-out"}
    _export_setup(tmp_path, monkeypatch, [*part, held])
    f.label_task("base", "held-out").mkdir(parents=True)
    f.report_labels(part, "base", "all")
    out = capsys.readouterr().out
    assert "flagged quotes labeled: 0/1" in out
    assert "1 flags with an empty summary_quote can't be labeled" in out
    assert "label-export --variant base --split dev" in out
    # An empty folder isn't a task yet, so its hint stays.
    f.report_labels([*part, held], "base", "all")
    out = capsys.readouterr().out
    assert "label-export --variant base --split dev" in out
    assert "label-export --variant base --split held-out" in out
    (f.label_task("base", "held-out") / "task.toml").write_text(f.label_card())
    monkeypatch.setattr(f, "read_labels", lambda t: [])
    f.report_labels([*part, held], "base", "all")
    out = capsys.readouterr().out
    assert "label-export --variant base --split dev" in out
    assert "label-export --variant base --split held-out" not in out


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

    def judged(jobs, one, results_path, **kw):
        results_path.write_text(f.json.dumps(_row(0, [], video="a")) + "\n")

    monkeypatch.setattr(f, "run_parallel", judged)
    reports = []
    monkeypatch.setattr(f, "cmd_system_report", lambda a: reports.append(a) or 1)
    assert f.cmd_system_eval(args) == 0
    assert [(a.variant, a.split) for a in reports] == [("v", "dev")]


def test_held_out_system_eval_succeeds_with_no_dev_rows(tmp_path, monkeypatch, capsys):
    args = _held_out_setup(tmp_path, monkeypatch)
    args.dry_run, args.effort, args.timeout_s, args.workers = False, "medium", 1, 1
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def judged(jobs, one, results_path, **kw):
        row = _row(0, [], video="a") | {"split": "held-out"}
        results_path.write_text(f.json.dumps(row) + "\n")

    monkeypatch.setattr(f, "run_parallel", judged)
    assert f.cmd_system_eval(args) == 0
    assert "no dev summaries to report" in capsys.readouterr().out


def _dev_rerun_setup(tmp_path, monkeypatch):
    """Video b has a judged rep0 on disk and a new rep1 to judge."""
    args = _held_out_setup(tmp_path, monkeypatch)
    rows = [{"video": "b", "type": "t", "split": "dev"}]
    (tmp_path / "sets" / "s.json").write_text(f.json.dumps(rows))
    for rep in (0, 1):
        rdir = f.gen_dir("v") / "b" / f"rep{rep}"
        rdir.mkdir(parents=True)
        (rdir / "summary.md").write_text("s")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "transcript.txt").write_text("t")
    run_dir = f.system_dir("v")
    run_dir.mkdir(parents=True)
    (run_dir / "results.jsonl").write_text(f.json.dumps(_row(0, [], video="b")) + "\n")
    args.split, args.dry_run = "dev", False
    args.effort, args.timeout_s, args.workers = "medium", 1, 1
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return args


def _all_calls_fail(jobs, one, results_path, **kw):
    with results_path.with_name("errors.jsonl").open("a") as out:
        for r, rep, _ in jobs:
            out.write(f.json.dumps({"video": r["video"], "rep": rep}) + "\n")


def test_system_eval_fails_when_no_judge_call_comes_back(tmp_path, monkeypatch, capsys):
    args = _dev_rerun_setup(tmp_path, monkeypatch)
    calls = []

    def failed(jobs, one, results_path, **kw):
        calls.extend((r["video"], rep) for r, rep, _ in jobs)
        _all_calls_fail(jobs, one, results_path)

    monkeypatch.setattr(f, "run_parallel", failed)
    reports = []
    monkeypatch.setattr(f, "cmd_system_report", lambda a: reports.append(a) or 0)
    # rep0 from an earlier run is still on disk; this run judged nothing.
    with pytest.raises(SystemExit, match="no dev judge call came back judged") as e:
        f.cmd_system_eval(args)
    assert "Dev results" not in str(e.value)
    assert calls == [("b", 1)]
    assert reports == []
    out = capsys.readouterr().out
    assert "dev: 0 of 1 judge calls came back judged (1 failed, see " in out
    assert "To judge the rest, rerun with --split dev." in out


def test_system_eval_names_the_calls_that_did_not_come_back(
    tmp_path, monkeypatch, capsys
):
    args = _dev_rerun_setup(tmp_path, monkeypatch)
    rdir = f.gen_dir("v") / "b" / "rep2"
    rdir.mkdir()
    (rdir / "summary.md").write_text("s")

    def half(jobs, one, results_path, **kw):
        (_, rep, _), *rest = jobs
        with results_path.open("a") as out:
            out.write(f.json.dumps(_row(rep, [], video="b")) + "\n")
        _all_calls_fail(rest, one, results_path)

    monkeypatch.setattr(f, "run_parallel", half)
    monkeypatch.setattr(f, "cmd_system_report", lambda a: 0)
    assert f.cmd_system_eval(args) == 0
    assert (
        "dev: 1 of 2 judge calls came back judged (1 failed" in capsys.readouterr().out
    )


def test_dev_system_eval_fails_when_every_dev_row_is_truncated(tmp_path, monkeypatch):
    args = _dev_rerun_setup(tmp_path, monkeypatch)
    rows = [_row(rep, [], video="b") | {"status": "truncated"} for rep in (0, 1)]
    results = f.system_dir("v") / "results.jsonl"
    results.write_text("".join(f.json.dumps(r) + "\n" for r in rows))
    # Truncated rows count as done, so this run sends nothing.
    monkeypatch.setattr(f, "run_parallel", lambda jobs, one, **kw: None)
    with pytest.raises(SystemExit, match="2 are truncated, and a rerun skips them"):
        f.cmd_system_eval(args)


def test_system_eval_names_a_call_truncated_in_this_run(tmp_path, monkeypatch, capsys):
    args = _dev_rerun_setup(tmp_path, monkeypatch)

    def truncated(jobs, one, results_path, **kw):
        with results_path.open("a") as out:
            row = _row(1, [], video="b") | {"status": "truncated"}
            out.write(f.json.dumps(row) + "\n")

    monkeypatch.setattr(f, "run_parallel", truncated)
    with pytest.raises(SystemExit, match="no dev judge call came back judged"):
        f.cmd_system_eval(args)
    out = capsys.readouterr().out
    assert "dev: 0 of 1 judge calls came back judged (1 truncated" in out
    assert "failed" not in out
    assert "remove the truncated rows (a rerun skips them), then rerun" in out


def _all_split_setup(tmp_path, monkeypatch):
    """Held-out video a and dev video b, one summary each, nothing judged."""
    args = _held_out_setup(tmp_path, monkeypatch)
    rows = [
        {"video": "a", "type": "t", "split": "held-out"},
        {"video": "b", "type": "t", "split": "dev"},
    ]
    (tmp_path / "sets" / "s.json").write_text(f.json.dumps(rows))
    rdir = f.gen_dir("v") / "b" / "rep0"
    rdir.mkdir(parents=True)
    (rdir / "summary.md").write_text("s")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "transcript.txt").write_text("t")
    args.split, args.dry_run = "all", False
    args.effort, args.timeout_s, args.workers = "medium", 1, 1
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return args


def _judge_only(split: str):
    def run(jobs, one, results_path, **kw):
        ok = [(r, rep) for r, rep, _ in jobs if r["split"] == split]
        with results_path.open("a") as out:
            for r, rep in ok:
                row = _row(rep, [], video=r["video"]) | {"split": r["split"]}
                out.write(f.json.dumps(row) + "\n")
        _all_calls_fail([j for j in jobs if j[0]["split"] != split], one, results_path)

    return run


def test_system_eval_fails_when_only_held_out_comes_back_empty(
    tmp_path, monkeypatch, capsys
):
    args = _all_split_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(f, "run_parallel", _judge_only("dev"))
    reports = []
    monkeypatch.setattr(f, "cmd_system_report", lambda a: reports.append(a) or 0)
    with pytest.raises(
        SystemExit,
        match="no held-out judge call came back judged; see above. Dev results: "
        "system-report --variant v --split dev",
    ):
        f.cmd_system_eval(args)
    assert reports == []
    out = capsys.readouterr().out
    assert "held-out: 0 of 1 judge calls came back judged (1 failed" in out
    assert "--split held-out --force-held-out." in out
    assert "dev:" not in out


def test_system_eval_names_held_out_calls_that_did_not_come_back(
    tmp_path, monkeypatch, capsys
):
    args = _all_split_setup(tmp_path, monkeypatch)
    rdir = f.gen_dir("v") / "a" / "rep1"
    rdir.mkdir()
    (rdir / "summary.md").write_text("s")

    def half(jobs, one, results_path, **kw):
        # Every rep0 comes back; held-out a's rep1 fails.
        with results_path.open("a") as out:
            for r, rep, _ in jobs:
                if rep == 0:
                    row = _row(0, [], video=r["video"]) | {"split": r["split"]}
                    out.write(f.json.dumps(row) + "\n")
        _all_calls_fail([j for j in jobs if j[1] != 0], one, results_path)

    monkeypatch.setattr(f, "run_parallel", half)
    reports = []
    monkeypatch.setattr(f, "cmd_system_report", lambda a: reports.append(a) or 0)
    assert f.cmd_system_eval(args) == 0
    assert [(a.variant, a.split) for a in reports] == [("v", "dev")]
    out = capsys.readouterr().out
    assert "held-out: 1 of 2 judge calls came back judged (1 failed" in out
    assert "--split held-out --force-held-out." in out


def test_load_set_refuses_a_video_listed_twice(tmp_path, monkeypatch):
    monkeypatch.setattr(f, "EVAL_DIR", tmp_path)
    (tmp_path / "sets").mkdir()
    rows = [
        {"video": "a", "type": "t", "split": "dev"},
        {"video": "a", "type": "t", "split": "held-out"},
    ]
    (tmp_path / "sets" / "s.json").write_text(f.json.dumps(rows))
    with pytest.raises(SystemExit, match="lists a more than once"):
        f.load_set("s", "dev")


def test_held_out_system_eval_fails_when_no_judge_call_comes_back(
    tmp_path, monkeypatch, capsys
):
    args = _held_out_setup(tmp_path, monkeypatch)
    args.dry_run, args.effort, args.timeout_s, args.workers = False, "medium", 1, 1
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(f, "run_parallel", _all_calls_fail)
    with pytest.raises(SystemExit, match="no held-out judge call came back judged"):
        f.cmd_system_eval(args)
    assert (f.system_dir("v") / f.HELD_OUT_MARKER).exists()
    assert "--split held-out --force-held-out." in capsys.readouterr().out


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
    assert "flagged quotes labeled: 1/2 (real 1, false alarm 0" in out
    assert "1 more labeled in an arvio session not finished yet" in out
    log.append({"kind": "reveal", "item_ids": keys[1:], "task_version": v})
    (task / "labels.jsonl").write_text(f.jsonl(log))
    f.report_labels(rows, "base", "dev")
    out = capsys.readouterr().out
    assert "flagged quotes labeled: 2/2 (real 1, false alarm 1" in out
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


def _seed_where(keys: list[str], n: int, want) -> str:
    """The first seed whose draw of n from keys passes want(draw)."""
    return next(
        s
        for s in map(str, range(1000))
        if want(set(sorted(keys, key=lambda k: f.sample_rank(s, k))[:n]))
    )


def _draw(variant: str = "base", split: str = "dev") -> dict:
    return f.json.loads((f.label_task(variant, split) / "sample.json").read_text())


def _label_folder(variant: str, split: str = "dev"):
    task = f.label_task(variant, split)
    task.mkdir(parents=True, exist_ok=True)
    (task / "task.toml").write_text(f.label_card())
    return task


QUOTES = ["Short.", "Start low", "Sonnet took", "Point [00:00:32].", "Intro."]


def test_label_export_sample_saves_a_draw_of_todays_flags(
    tmp_path, monkeypatch, capsys
):
    issues = [_issue(q) for q in QUOTES] + [_issue("")]
    args = _export_setup(tmp_path, monkeypatch, [_row(0, issues)])
    keys = [_key(q) for q in QUOTES]
    other = _label_folder("other")
    monkeypatch.setattr(
        f, "read_labels", lambda t: [_label(keys[0], "real")] if t == other else []
    )
    args.sample = 2
    args.seed = _seed_where(keys, 2, lambda d: keys[0] in d)
    f.cmd_label_export(args)
    task = f.label_task("base", "dev")
    draw = _draw()
    top = sorted(keys, key=lambda k: f.sample_rank(args.seed, k))[:2]
    assert draw == {"seed": args.seed, "n": 2, "keys": sorted(top)}
    # Labeled in another folder: drawn, but not exported again.
    items = [i["id"] for i in f.read_jsonl(task / "items.jsonl")]
    assert items == [k for k in keys if k in top and k != keys[0]]
    out = capsys.readouterr().out
    assert (
        f"drew 2 of them (seed {args.seed!r}, saved in sample.json): 1 to label in "
        "this folder, 1 already labeled in another label folder; kept 0 labeled "
        "items from before outside the draw -> 1 items" in out
    )
    assert not list(task.glob(".sample.json.*"))

    args.sample = 4
    f.cmd_label_export(args)
    assert set(_draw()["keys"]) > set(top)
    args.sample = 50
    f.cmd_label_export(args)
    assert _draw()["keys"] == sorted(keys)

    before = (task / "sample.json").read_bytes()
    args.sample = None
    f.cmd_label_export(args)
    assert (task / "sample.json").read_bytes() == before
    assert len(f.read_jsonl(task / "items.jsonl")) == 4

    no_draw = f.argparse.Namespace(variant="v2", split="dev", sample=None, seed="1")
    _export_setup(tmp_path, monkeypatch, [_row(0, [_issue()])], "v2")
    f.cmd_label_export(no_draw)
    assert not (f.label_task("v2", "dev") / "sample.json").exists()


def test_label_export_sample_keeps_labels_outside_the_draw(
    tmp_path, monkeypatch, capsys
):
    args = _export_setup(tmp_path, monkeypatch, [_row(0, [_issue(q) for q in QUOTES])])
    keys = [_key(q) for q in QUOTES]
    args.sample, args.seed = 2, "7"
    f.cmd_label_export(args)
    task = f.label_task("base", "dev")
    drawn = [i["id"] for i in f.read_jsonl(task / "items.jsonl")]
    assert set(drawn) == set(_draw()["keys"])
    f.cmd_label_export(args)
    assert [i["id"] for i in f.read_jsonl(task / "items.jsonl")] == drawn

    # A label outside a new draw keeps its item; arvio refuses it otherwise.
    (task / "labels.jsonl").write_text(
        f.json.dumps({"kind": "label", "item_id": drawn[0]}) + "\n"
    )
    monkeypatch.setattr(f, "counted_labels", lambda t, items: {drawn[0]})
    args.seed = _seed_where(keys, 2, lambda d: drawn[0] not in d)
    capsys.readouterr()
    f.cmd_label_export(args)
    again = [i["id"] for i in f.read_jsonl(task / "items.jsonl")]
    assert drawn[0] in again and len(again) == 3
    assert drawn[0] not in _draw()["keys"]
    assert (
        "2 to label in this folder, 0 already labeled in another label folder; "
        "kept 1 labeled items from before outside the draw -> 3 items"
        in capsys.readouterr().out
    )

    # Back to the first draw: the labeled item is in it and isn't counted as to label.
    args.seed = "7"
    f.cmd_label_export(args)
    assert (
        "1 to label in this folder (1 labeled already), 0 already labeled"
        in capsys.readouterr().out
    )

    # A counted label whose flag is gone keeps its item but isn't drawn.
    left = [q for q, k in zip(QUOTES, keys, strict=True) if k != drawn[0]]
    _export_setup(tmp_path, monkeypatch, [_row(0, [_issue(q) for q in left])])
    args.sample = 50
    f.cmd_label_export(args)
    assert _draw()["keys"] == sorted(k for k in keys if k != drawn[0])
    assert drawn[0] in [i["id"] for i in f.read_jsonl(task / "items.jsonl")]
    assert "outside the draw (1 whose flag is gone)" in capsys.readouterr().out


def test_label_export_sample_must_be_positive(monkeypatch, capsys):
    monkeypatch.setattr(
        f.sys, "argv", ["faithfulness.py", "label-export", "--sample", "0"]
    )
    with pytest.raises(SystemExit) as e:
        f.main()
    assert e.value.code == 2
    assert "must be 1 or more" in capsys.readouterr().err


def test_label_export_seed_needs_sample(tmp_path, monkeypatch, capsys):
    argv = ["faithfulness.py", "label-export", "--variant", "base", "--seed", "3"]
    monkeypatch.setattr(f.sys, "argv", argv)
    with pytest.raises(SystemExit) as e:
        f.main()
    assert e.value.code == 2
    assert "--seed needs --sample" in capsys.readouterr().err

    _export_setup(tmp_path, monkeypatch, [_row(0, [_issue(q) for q in QUOTES])])
    monkeypatch.setattr(f.sys, "argv", [*argv[:4], "--sample", "2"])
    monkeypatch.setattr(f, "read_labels", lambda t: [])
    assert f.main() == 0
    assert _draw()["seed"] == "1"


def _two_videos_two_reps(tmp_path, monkeypatch):
    """Each summary flags its video's real quote and a false alarm of its own."""
    rows, real, false = [], [], []
    for video in ("2026-01-01_x", "2026-01-02_x"):
        real.append(_key("Short.", video=video))
        for rep, alarm in enumerate(["Start low", "Point [00:00:32]."]):
            rows.append(_row(rep, [_issue("Short."), _issue(alarm)], video=video))
            false.append(_key(alarm, video=video))
    args = _export_setup(tmp_path, monkeypatch, rows)
    return rows, args, real, false


def test_estimate_weights_quotes_by_the_summaries_that_flag_them(
    tmp_path, monkeypatch, capsys
):
    rows, args, real, false = _two_videos_two_reps(tmp_path, monkeypatch)
    monkeypatch.setattr(f, "read_labels", lambda t: [])
    args.sample = 5
    args.seed = _seed_where(real + false, 5, lambda d: false[-1] not in d)
    f.cmd_label_export(args)
    labels = [_label(k, "real") for k in real] + [
        _label(k, "false_alarm") for k in false
    ]
    monkeypatch.setattr(f, "read_labels", lambda t: labels[:-1])
    capsys.readouterr()
    f.report_labels(rows, "base", "dev")
    out = capsys.readouterr().out
    # 2 flagged quotes per summary times 4/7: each real quote is flagged in 2
    # summaries, each false alarm in 1, and the undrawn false alarm is
    # unlabeled. The counted figure is 1.00.
    assert "estimated real errors per summary: 1.14 (range" in out
    assert "2 real of 5 drawn" in out
    # Can't tell counts as not real, as in the counted figure.
    unsure = [*labels[:-2], _label(false[-2], "cant_tell")]
    monkeypatch.setattr(f, "read_labels", lambda t: unsure)
    f.report_labels(rows, "base", "dev")
    assert "estimated real errors per summary: 1.14 (range" in capsys.readouterr().out

    monkeypatch.setattr(f, "read_labels", lambda t: labels)
    f.report_labels(rows, "base", "dev")
    out = capsys.readouterr().out
    assert "real errors per summary: mean 1.00" in out
    assert "estimated" not in out


def test_estimate_uses_only_the_variants_own_draw(tmp_path, monkeypatch, capsys):
    shared = QUOTES[:3]
    a = [_row(0, [_issue(q) for q in shared])]
    b = [_row(0, [_issue(q) for q in QUOTES])]
    _export_setup(tmp_path, monkeypatch, a)
    args = _export_setup(tmp_path, monkeypatch, b, "v2")
    keys = [_key(q) for q in QUOTES]
    base = _label_folder("base")
    labels = [
        _label(keys[0], "real"),
        _label(keys[1], "false_alarm"),
        _label(keys[2], "false_alarm"),
    ]
    monkeypatch.setattr(f, "read_labels", lambda t: labels if t == base else [])
    # v2 draws one quote base labeled real and one only v2 has.
    args.sample = 2
    args.seed = _seed_where(
        keys, 2, lambda d: keys[0] in d and len(d & set(keys[3:])) == 1
    )
    f.cmd_label_export(args)
    capsys.readouterr()
    # The quote base labeled is not a random half of the draw: wait for the rest.
    f.report_labels(b, "v2", "dev")
    out = capsys.readouterr().out
    assert "judge precision on labeled quotes: 1/3" in out
    assert "estimated" not in out
    assert (
        "real errors per summary not shown: 1 of 2 drawn quotes labeled: label "
        "the rest in arvio and press Finish\n" in out
    )
    f.cmd_compare(_compare_args("base", "v2"))
    out = capsys.readouterr().out
    assert "(est.)" not in out
    assert "v2: real errors per summary not shown: 1 of 2 drawn" in out

    v2 = f.label_task("v2", "dev")
    mine = next(k for k in _draw("v2")["keys"] if k != keys[0])
    labels_by = {base: labels, v2: [_label(mine, "false_alarm")]}
    monkeypatch.setattr(f, "read_labels", lambda t: labels_by.get(t, []))
    f.report_labels(b, "v2", "dev")
    out = capsys.readouterr().out
    lo, hi = f.poisson_range(5)
    p_lo, p_hi = f.wilson(1, 2)
    assert (
        f"estimated real errors per summary: 2.50 (range {lo * p_lo:.2f}-"
        f"{hi * p_hi:.2f}, approximate)" in out
    )
    assert "1 real of 2 drawn" in out

    assert f.cmd_compare(_compare_args("base", "v2")) == 0
    out = capsys.readouterr().out
    errors = next(ln for ln in out.splitlines() if "real errors per summary" in ln)
    assert " 1.00 (" in errors and "2.50 (est.) (" in errors
    assert errors.endswith("ranges overlap (one counted, one estimated)")

    (f.label_task("v2", "dev") / "sample.json").unlink()
    f.report_labels(b, "v2", "dev")
    out = capsys.readouterr().out
    assert "estimated" not in out
    assert (
        "real errors per summary not shown: no sample drawn: run label-export "
        "--sample N --variant v2 --split dev" in out
    )


def test_estimate_needs_the_draw_to_match_todays_flags(tmp_path, monkeypatch, capsys):
    rows = [_row(0, [_issue(q) for q in QUOTES[:3]])]
    args = _export_setup(tmp_path, monkeypatch, rows)
    first = _key(QUOTES[0])
    monkeypatch.setattr(f, "read_labels", lambda t: [_label(first, "real")])
    args.sample = 1
    args.seed = _seed_where([_key(q) for q in QUOTES[:3]], 1, lambda d: first in d)
    f.cmd_label_export(args)
    capsys.readouterr()
    f.report_labels(rows, "base", "dev")
    assert "estimated real errors per summary: 3.00" in capsys.readouterr().out
    more = [_row(0, [_issue(q) for q in QUOTES])]
    _export_setup(tmp_path, monkeypatch, more)
    f.report_labels(more, "base", "dev")
    out = capsys.readouterr().out
    assert "estimated" not in out
    assert (
        f"the flags changed since the draw: run label-export --sample 1 --seed "
        f"{args.seed} --variant base --split dev again" in out
    )
    drawn = {"seed": "my seed", "n": 1, "keys": ["gone"]}
    (f.label_task("base", "dev") / "sample.json").write_text(f.json.dumps(drawn))
    f.report_labels(more, "base", "dev")
    assert "--sample 1 --seed 'my seed' --variant" in capsys.readouterr().out


def test_estimate_waits_for_every_drawn_quote(tmp_path, monkeypatch, capsys):
    rows = [_row(0, [_issue(q) for q in QUOTES[:3]])]
    args = _export_setup(tmp_path, monkeypatch, rows)
    monkeypatch.setattr(f, "read_labels", lambda t: [])
    args.sample = 2
    f.cmd_label_export(args)
    drawn = _draw()["keys"]
    capsys.readouterr()
    f.report_labels(rows, "base", "dev")
    out = capsys.readouterr().out
    assert (
        "real errors per summary not shown: 0 of 2 drawn quotes labeled: label "
        "the rest in arvio and press Finish\n" in out
    )
    # A label in a session not finished yet doesn't count, and the reason says so.
    labels = [_label(drawn[0], "real"), _label(drawn[1], "real", locked=False)]
    monkeypatch.setattr(f, "read_labels", lambda t: labels)
    f.report_labels(rows, "base", "dev")
    out = capsys.readouterr().out
    assert (
        "1 of 2 drawn quotes labeled: label the rest in arvio and press Finish "
        "(1 labeled in a session not finished yet, in this or another label "
        "folder)" in out
    )
    labels[1]["locked"] = True
    f.report_labels(rows, "base", "dev")
    assert "estimated real errors per summary: 3.00" in capsys.readouterr().out


def test_estimate_cant_read_a_broken_draw(tmp_path, monkeypatch, capsys):
    rows = [_row(0, [_issue(q) for q in QUOTES[:3]])]
    args = _export_setup(tmp_path, monkeypatch, rows)
    monkeypatch.setattr(f, "read_labels", lambda t: [])
    args.sample = 1
    f.cmd_label_export(args)
    (f.label_task("base", "dev") / "sample.json").write_text("{not json")
    capsys.readouterr()
    f.report_labels(rows, "base", "dev")
    assert (
        "real errors per summary not shown: can't read the draw: "
        in capsys.readouterr().out
    )


def _split_rows(held_out: list[str]) -> list[dict]:
    """A dev summary flagging three quotes and a held-out one flagging these."""
    rows = [
        _row(0, [_issue(q) for q in QUOTES[:3]]),
        _row(0, [_issue(q) for q in held_out], video="2026-01-02_x"),
    ]
    rows[1]["split"] = "held-out"
    return rows


def test_estimate_for_split_all_needs_flags_in_one_split(tmp_path, monkeypatch, capsys):
    rows = _split_rows(QUOTES[3:])
    _export_setup(tmp_path, monkeypatch, rows)
    monkeypatch.setattr(f, "read_labels", lambda t: [])
    f.report_labels(rows, "base", "all")
    assert (
        "real errors per summary not shown: draws are per split: use --split dev "
        "or --split held-out" in capsys.readouterr().out
    )


def test_estimate_for_split_all_uses_the_one_flagged_splits_draw(
    tmp_path, monkeypatch, capsys
):
    rows = _split_rows([])
    args = _export_setup(tmp_path, monkeypatch, rows)
    first = _key(QUOTES[0])
    monkeypatch.setattr(f, "read_labels", lambda t: [_label(first, "real")])
    args.sample = 1
    args.seed = _seed_where([_key(q) for q in QUOTES[:3]], 1, lambda d: first in d)
    f.cmd_label_export(args)
    capsys.readouterr()
    f.report_labels(rows, "base", "all")
    # 3 flags over both summaries, all real in the draw.
    assert "estimated real errors per summary: 1.50" in capsys.readouterr().out


def test_no_estimate_or_count_while_a_quote_is_empty(tmp_path, monkeypatch, capsys):
    rows = [_row(0, [_issue(q) for q in QUOTES[:3]] + [_issue("")])]
    args = _export_setup(tmp_path, monkeypatch, rows)
    _export_setup(tmp_path, monkeypatch, [_row(0, [_issue()])], "v2")
    monkeypatch.setattr(f, "read_labels", lambda t: [_label(_key(QUOTES[0]), "real")])
    args.sample = 50
    f.cmd_label_export(args)
    capsys.readouterr()
    f.report_labels(rows, "base", "dev")
    out = capsys.readouterr().out
    assert "estimated" not in out
    assert (
        "real errors per summary not shown: the flags with an empty summary_quote"
        in out
    )
    f.cmd_compare(_compare_args("base", "v2"))
    out = capsys.readouterr().out
    errors = next(ln for ln in out.splitlines() if "real errors per summary" in ln)
    assert errors.split()[4] == "-"
    assert "base: real errors per summary not shown: the flags with an empty" in out


def test_mean_range_uses_t_or_an_exact_poisson_range():
    lo, hi = f.poisson_range(0)
    assert lo == 0 and round(hi, 3) == 3.689
    assert [round(x, 3) for x in f.poisson_range(4)] == [1.090, 10.242]
    assert [round(x, 2) for x in f.poisson_range(10)] == [4.80, 18.39]
    assert f.mean_range([2.0, 2.0]) == tuple(x / 2 for x in f.poisson_range(4))
    assert f.mean_range([3.0]) == f.poisson_range(3)
    lo, hi = f.mean_range([0.0, 1.0])
    assert lo == 0 and round(hi, 3) == round(0.5 + 12.706 * 0.5, 3)
    xs = [float(i % 3) for i in range(40)]
    m, sd = f.mean_sd(xs)
    want = m + 2.0227 * sd / f.math.sqrt(40)
    assert f.mean_range(xs)[1] == pytest.approx(want, rel=1e-4)
    # Exact t quantiles past the table, and no jump where it ends.
    for df, t in ((31, 2.0395), (60, 2.0003), (120, 1.9799)):
        assert f.t975(df) == pytest.approx(t, abs=0.0005)
    assert f.t975(30) - f.t975(31) < 0.01


def _compare_args(a: str, b: str):
    return f.argparse.Namespace(a=a, b=b, split="dev")


def _table(out: str) -> dict[str, str]:
    return {
        ln.split("  ")[1].strip(): ln for ln in out.splitlines() if ln.startswith("  ")
    }


def test_compare_shows_both_variants_and_overlap(tmp_path, monkeypatch, capsys):
    base = [
        _row(0, [_issue(), _issue("Short.", severity="major")]),
        _row(1, [_issue()]),
    ]
    _export_setup(tmp_path, monkeypatch, base)
    _export_setup(tmp_path, monkeypatch, [_row(0, []), _row(1, [_issue()])], "v2")
    labels = [_label(_key(QUOTE), "real"), _label(_key("Short."), "false_alarm")]
    monkeypatch.setattr(f, "read_labels", lambda t: labels)
    _label_folder("base")
    assert f.cmd_compare(_compare_args("base", "v2")) == 0
    out = capsys.readouterr().out
    lines = _table(out)
    lo, hi = f.wilson(1, 2)
    assert f"1/2 ({lo:.2f}-{hi:.2f})" in lines["no major flag"]
    assert "2/2 " in lines["no major flag"]
    assert lines["no major flag"].endswith("ranges overlap")
    assert (
        "2/2" in lines["flagged quotes labeled"]
        and "1/1" in lines["flagged quotes labeled"]
    )
    assert "1/2" in lines["judge precision"] and "1/1" in lines["judge precision"]
    assert "overlap" not in lines["judge precision"]
    assert out.count("Judge precision gets no verdict") == 1
    assert "(est.)" not in out
    assert "different videos" not in out and "different reps" not in out


def test_compare_overlap_with_a_zero_variant(tmp_path, monkeypatch, capsys):
    videos = [f"2026-01-{d:02}_x" for d in range(1, 13)]
    quiet = [_row(r, [], video=v) for v in videos for r in (0, 1)]
    _export_setup(tmp_path, monkeypatch, quiet)
    some = [
        _row(r, [_issue()] if i < 2 else [], video=v)
        for i, v in enumerate(videos)
        for r in (0, 1)
    ]
    _export_setup(tmp_path, monkeypatch, some, "v2")
    labels = [_label(_key(QUOTE, video=v), "real") for v in videos[:2]]
    monkeypatch.setattr(f, "read_labels", lambda t: labels)
    _label_folder("v2")
    assert sum(len(r["issues"]) for r in some) == 4
    f.cmd_compare(_compare_args("base", "v2"))
    lines = _table(capsys.readouterr().out)
    zero = f"0.00 (0.00-{f.poisson_range(0)[1] / 24:.2f})"
    assert lines["judge flags per summary"].split("  ")[2].strip() == zero
    assert "0/0" in lines["flagged quotes labeled"]
    assert lines["judge precision"].split()[2] == "-"
    errors = lines["real errors per summary"]
    assert zero in errors and "0.17 (" in errors
    assert errors.endswith("ranges overlap")
    assert lines["judge flags per summary"].endswith("ranges overlap")


def test_compare_names_the_better_variant(tmp_path, monkeypatch, capsys):
    many = [_issue(q, severity="major") for q in QUOTES[:4]]
    rows = [_row(r, many, video=f"2026-01-0{d}_x") for d in range(1, 6) for r in (0, 1)]
    _export_setup(tmp_path, monkeypatch, rows)
    quiet = [_row(r, [], video=f"2026-01-0{d}_x") for d in range(1, 6) for r in (0, 1)]
    _export_setup(tmp_path, monkeypatch, quiet, "v2")
    monkeypatch.setattr(f, "read_labels", lambda t: [])
    for a, b in (("base", "v2"), ("v2", "base")):
        f.cmd_compare(_compare_args(a, b))
        lines = _table(capsys.readouterr().out)
        flags = lines["judge flags per summary"]
        assert flags.endswith("v2 fewer flags")
        lo, hi = (x / 10 for x in f.poisson_range(40))
        assert f"4.00 ({lo:.2f}-{hi:.2f})" in flags
        assert lines["no major flag"].endswith("v2 more summaries with no major flag")
        cells = lines["real errors per summary"].split()[4:]
        assert sorted(cells) == ["(0.00-0.37)", "-", "0.00"]
    _label_folder("base")
    labels = [
        _label(_key(q, video=f"2026-01-0{d}_x"), "real")
        for q in QUOTES[:4]
        for d in range(1, 6)
    ]
    monkeypatch.setattr(f, "read_labels", lambda t: labels)
    f.cmd_compare(_compare_args("base", "v2"))
    errors = _table(capsys.readouterr().out)["real errors per summary"]
    assert "4.00 (" in errors and "(est.)" not in errors
    assert errors.endswith("v2 fewer real errors")
    with pytest.raises(SystemExit, match="no judged summaries for 'nope'"):
        f.cmd_compare(_compare_args("base", "nope"))


def test_compare_warns_about_different_videos(tmp_path, monkeypatch, capsys):
    _export_setup(tmp_path, monkeypatch, [_row(0, [])])
    _export_setup(tmp_path, monkeypatch, [_row(0, [], video="2026-01-02_x")], "v2")
    f.cmd_compare(_compare_args("base", "v2"))
    out = capsys.readouterr().out
    assert "the two variants cover different videos" in out
    assert "different reps" not in out


def test_compare_prints_flags_when_labels_fail(tmp_path, monkeypatch, capsys):
    _export_setup(tmp_path, monkeypatch, [_row(0, [_issue()]), _row(1, [_issue()])])
    _export_setup(tmp_path, monkeypatch, [_row(0, [])], "v2")
    _label_folder("base")

    def broken(t):
        raise f.LabelError("arvio is not installed or not on PATH")

    monkeypatch.setattr(f, "read_labels", broken)
    assert f.cmd_compare(_compare_args("base", "v2")) == 1
    out = capsys.readouterr().out
    lines = _table(out)
    assert "1.00 (" in lines["judge flags per summary"]
    for name in (
        "flagged quotes labeled",
        "judge precision",
        "real errors per summary",
    ):
        assert lines[name].split()[-2:] == ["-", "-"]
    assert "labels not counted: arvio is not installed" in out
    assert "different reps of 1 videos" in out
    assert "different videos" not in out


def test_compare_runs_from_the_command_line(tmp_path, monkeypatch, capsys):
    _export_setup(tmp_path, monkeypatch, [_row(0, [])])
    _export_setup(tmp_path, monkeypatch, [_row(0, [])], "v2")
    argv = ["faithfulness.py", "compare", "base", "v2", "--split", "dev"]
    monkeypatch.setattr(f.sys, "argv", argv)
    assert f.main() == 0
    assert "dev: base vs v2 (95% ranges, approximate)" in capsys.readouterr().out


@needs_arvio
def test_arvio_accepts_a_redrawn_sample(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ARVIO_DATA_DIR", str(tmp_path / "arvio"))
    args = _export_setup(tmp_path, monkeypatch, [_row(0, [_issue(q) for q in QUOTES])])
    keys = [_key(q) for q in QUOTES]
    args.sample, args.seed = 2, "1"
    f.cmd_label_export(args)
    task = f.label_task("base", "dev")
    version = f.re.search(r"card version (\w+)", _arvio_check(tmp_path, task).stdout)
    first = _draw()["keys"][0]
    label = {
        "kind": "label",
        "item_id": first,
        "task_version": version.group(1),
        "answer": "real",
        "followups": {"severity": "minor"},
    }
    (task / "labels.jsonl").write_text(f.json.dumps(label) + "\n")
    args.seed = _seed_where(keys, 2, lambda d: first not in d)
    f.cmd_label_export(args)
    assert first not in _draw()["keys"]
    assert first in [i["id"] for i in f.read_jsonl(task / "items.jsonl")]
    done = _arvio_check(tmp_path, task)
    assert done.returncode == 0, done.stderr
    assert "ok: 3 items, 1 labeled" in done.stdout
