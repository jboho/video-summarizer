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
