import importlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sv = importlib.import_module("summarize_video")


def test_resolve_provider_infers_anthropic_from_claude_model(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert sv.resolve_provider("claude-sonnet-5", "auto", None) == "anthropic"


def test_resolve_provider_infers_openai_from_non_claude_model(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert sv.resolve_provider("gpt-4o-mini", "auto", None) == "openai"


def test_resolve_provider_explicit_flag_wins_over_model(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert sv.resolve_provider("claude-sonnet-5", "openai", None) == "openai"


def test_resolve_provider_base_url_forces_openai(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert (
        sv.resolve_provider("claude-sonnet-5", "auto", "http://localhost:11434/v1")
        == "openai"
    )


def test_resolve_provider_env_base_url_forces_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    assert sv.resolve_provider("some-model", "auto", None) == "openai"


def test_normalize_usage_anthropic():
    u = SimpleNamespace(input_tokens=10, output_tokens=20)
    assert sv.normalize_usage("anthropic", u) == (10, 20)


def test_normalize_usage_openai():
    u = SimpleNamespace(prompt_tokens=5, completion_tokens=7)
    assert sv.normalize_usage("openai", u) == (5, 7)


def test_normalize_usage_none():
    assert sv.normalize_usage("openai", None) is None


def test_estimate_cost_known_model():
    got = sv.estimate_cost("claude-sonnet-5", (1_000_000, 1_000_000), None)
    assert got == "$12.000 (1000000 in / 1000000 out)"


def test_estimate_cost_unknown_model_returns_none():
    assert sv.estimate_cost("mystery-model", (100, 100), None) is None


def test_estimate_cost_override():
    got = sv.estimate_cost("mystery-model", (1_000_000, 1_000_000), (1.0, 2.0))
    assert got == "$3.000 (1000000 in / 1000000 out)"


def test_estimate_cost_none_usage():
    assert sv.estimate_cost("claude-sonnet-5", None, None) is None


class _FakeAnthropicStream:
    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self._response


class _FakeMessages:
    def __init__(self, response, recorder):
        self._response = response
        self._recorder = recorder

    def stream(self, **kwargs):
        self._recorder.update(kwargs)
        return _FakeAnthropicStream(self._response)


class _FakeAnthropicClient:
    def __init__(self, response, recorder):
        self.messages = _FakeMessages(response, recorder)


def _install_fake_anthropic(monkeypatch, response, recorder):
    fake_mod = SimpleNamespace(
        Anthropic=lambda: _FakeAnthropicClient(response, recorder)
    )
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)


def test_build_user_content_includes_key_parts():
    meta = {"title": "T", "uploader": "C", "duration": 65, "webpage_url": "http://x"}
    out = sv.build_user_content(meta, "the transcript body")
    assert "the transcript body" in out
    assert "T" in out and "http://x" in out
    assert "## TL;DR" in out


def test_parse_sections_orders_by_prompt_not_input():
    assert sv.parse_sections("quotes, tldr") == ["tldr", "quotes"]


def test_parse_sections_all_and_dedupe():
    assert sv.parse_sections("all") == list(sv.SECTION_PROMPTS)
    assert sv.parse_sections("tldr,tldr") == ["tldr"]


@pytest.mark.parametrize("bad", ["nope", "tldr,nope", "", " , "])
def test_parse_sections_rejects_unknown_or_empty(bad, capsys):
    with pytest.raises(SystemExit) as exc:
        sv.parse_sections(bad)
    assert exc.value.code == 1
    assert "Valid names" in capsys.readouterr().err


def test_build_user_content_only_requested_sections():
    out = sv.build_user_content({}, "body", ["actionable"])
    assert "## Actionable summary" in out
    for heading in ("## TL;DR", "## Key points", "## Notable quotes"):
        assert heading not in out


def test_default_sections_drop_quotes():
    out = sv.build_user_content({}, "body")
    assert "## Actionable summary" in out
    assert "## Notable quotes" not in out


def test_summarize_passes_sections_to_prompt(monkeypatch):
    recorder = {}
    _install_fake_anthropic(
        monkeypatch,
        SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="ok")],
            usage=None,
        ),
        recorder,
    )
    sv.summarize("anthropic", "claude-sonnet-5", {}, "body", None, ["quotes"])
    prompt = recorder["messages"][0]["content"]
    assert "## Notable quotes" in prompt and "## TL;DR" not in prompt


def test_summarize_anthropic_assembles_text(monkeypatch):
    block = SimpleNamespace(type="text", text="hello world")
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[block],
        usage=SimpleNamespace(input_tokens=3, output_tokens=2),
    )
    recorder = {}
    _install_fake_anthropic(monkeypatch, response, recorder)
    meta = {"title": "T", "duration": 0}
    text, usage = sv.summarize_anthropic("claude-sonnet-5", meta, "body")
    assert text == "hello world"
    assert usage.input_tokens == 3
    assert recorder["system"] == sv.SUMMARY_SYSTEM
    assert recorder["messages"][0]["role"] == "user"


def test_summarize_anthropic_refusal_exits(monkeypatch):
    response = SimpleNamespace(
        stop_reason="refusal",
        stop_details=SimpleNamespace(category="x"),
        content=[],
        usage=None,
    )
    _install_fake_anthropic(monkeypatch, response, {})
    with pytest.raises(SystemExit):
        sv.summarize_anthropic("claude-sonnet-5", {"title": "T", "duration": 0}, "body")


def _sse(content=None, finish_reason=None, refusal=None):
    delta = SimpleNamespace(content=content, refusal=refusal)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


def _usage_event(pin, pout):
    return SimpleNamespace(
        choices=[], usage=SimpleNamespace(prompt_tokens=pin, completion_tokens=pout)
    )


class _FakeCompletions:
    def __init__(self, events, recorder):
        self._events = events
        self._recorder = recorder

    def create(self, **kwargs):
        self._recorder.update(kwargs)
        return iter(self._events)


class _FakeOpenAIClient:
    def __init__(self, events, recorder, base_url=None):
        recorder["base_url"] = base_url
        self.chat = SimpleNamespace(completions=_FakeCompletions(events, recorder))


def _install_fake_openai(monkeypatch, events, recorder):
    fake_mod = SimpleNamespace(
        OpenAI=lambda base_url=None: _FakeOpenAIClient(events, recorder, base_url)
    )
    monkeypatch.setitem(sys.modules, "openai", fake_mod)


def test_summarize_openai_assembles_and_captures_usage(monkeypatch):
    events = [
        _sse("Hello "),
        _sse("world"),
        _sse(finish_reason="stop"),
        _usage_event(11, 22),
    ]
    recorder = {}
    _install_fake_openai(monkeypatch, events, recorder)
    text, usage = sv.summarize_openai(
        "gpt-4o-mini", {"title": "T", "duration": 0}, "body", None
    )
    assert text == "Hello world"
    assert usage.prompt_tokens == 11 and usage.completion_tokens == 22
    msgs = recorder["messages"]
    assert msgs[0] == {"role": "system", "content": sv.SUMMARY_SYSTEM}
    assert msgs[1]["role"] == "user"


def test_summarize_openai_passes_base_url(monkeypatch):
    events = [_sse("x"), _sse(finish_reason="stop")]
    recorder = {}
    _install_fake_openai(monkeypatch, events, recorder)
    sv.summarize_openai(
        "llama3", {"title": "T", "duration": 0}, "body", "http://localhost:11434/v1"
    )
    assert recorder["base_url"] == "http://localhost:11434/v1"


def test_summarize_openai_content_filter_exits(monkeypatch):
    events = [_sse("partial"), _sse(finish_reason="content_filter")]
    _install_fake_openai(monkeypatch, events, {})
    with pytest.raises(SystemExit):
        sv.summarize_openai("gpt-4o-mini", {"title": "T", "duration": 0}, "body", None)


def test_summarize_openai_refusal_exits(monkeypatch):
    events = [
        _sse("partial"),
        _sse(refusal="I can't help with that", finish_reason="stop"),
    ]
    _install_fake_openai(monkeypatch, events, {})
    with pytest.raises(SystemExit):
        sv.summarize_openai("gpt-4o-mini", {"title": "T", "duration": 0}, "body", None)


def test_summarize_dispatches_to_anthropic(monkeypatch):
    called = {}
    monkeypatch.setattr(
        sv,
        "summarize_anthropic",
        lambda m, meta, t, sections=None: (called.setdefault("p", "a"), None),
    )
    monkeypatch.setattr(
        sv,
        "summarize_openai",
        lambda m, meta, t, b, sections=None: (called.setdefault("p", "o"), None),
    )
    text, _ = sv.summarize("anthropic", "claude-sonnet-5", {}, "body", None)
    assert called["p"] == "a" and text == "a"


def test_summarize_dispatches_to_openai(monkeypatch):
    called = {}
    monkeypatch.setattr(
        sv,
        "summarize_anthropic",
        lambda m, meta, t, sections=None: (called.setdefault("p", "a"), None),
    )
    monkeypatch.setattr(
        sv,
        "summarize_openai",
        lambda m, meta, t, b, sections=None: (called.setdefault("p", "o"), None),
    )
    text, _ = sv.summarize("openai", "gpt-4o-mini", {}, "body", "http://h/v1")
    assert called["p"] == "o" and text == "o"


# --- main(): subtitle download failure handling ---

_VTT = "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nhello there\n"


def _run_main(
    monkeypatch, tmp_path, sub_rc, sub_stderr, write_vtt, extra_args=(), summary=False
):
    def fake_run(cmd, **kwargs):
        if "-J" in cmd:
            meta = '{"title": "T", "id": "abc", "upload_date": "20260101"}'
            return SimpleNamespace(returncode=0, stdout=meta, stderr="")
        if write_vtt:
            out = cmd[cmd.index("-o") + 1]
            (Path(out).parent / "abc.en.vtt").write_text(_VTT, encoding="utf-8")
        return SimpleNamespace(returncode=sub_rc, stdout="", stderr=sub_stderr)

    monkeypatch.setattr(sv.subprocess, "run", fake_run)
    monkeypatch.setattr(sv, "check_yt_dlp", lambda: None)
    out_dir = tmp_path / "out"
    argv = ["summarize_video.py", "http://x", "--out", str(out_dir)]
    if not summary:
        argv.append("--no-summary")
    monkeypatch.setattr(sys, "argv", [*argv, *extra_args])
    return out_dir


def _main_output(capsys):
    with pytest.raises(SystemExit) as e:
        sv.main()
    assert e.value.code != 0
    cap = capsys.readouterr()
    return cap.out + cap.err


def test_main_429_reports_rate_limit_not_no_captions(monkeypatch, tmp_path, capsys):
    err = "ERROR: Unable to download video subtitles: HTTP Error 429: Too Many Requests"
    _run_main(monkeypatch, tmp_path, 1, err, False)
    text = _main_output(capsys)
    assert "429" in text and "--cookies-from-browser" in text
    assert "No captions found" not in text


def test_main_other_ytdlp_error_shows_its_text(monkeypatch, tmp_path, capsys):
    err = "ERROR: [Errno 8] nodename nor servname provided, or not known"
    _run_main(monkeypatch, tmp_path, 1, err, False)
    text = _main_output(capsys)
    assert "nodename nor servname provided" in text
    assert "No captions found" not in text


def test_main_429_inside_a_video_id_is_not_a_rate_limit(monkeypatch, tmp_path, capsys):
    err = "ERROR: [youtube] aB4296xyz12: Video unavailable"
    _run_main(monkeypatch, tmp_path, 1, err, False)
    text = _main_output(capsys)
    assert "Video unavailable" in text
    assert "rate-limited" not in text and "--cookies-from-browser" not in text


def test_main_empty_stderr_still_reports_exit_code(monkeypatch, tmp_path, capsys):
    _run_main(monkeypatch, tmp_path, 3, "", False)
    text = _main_output(capsys)
    assert "exited with 3" in text
    assert "No captions found" not in text


def test_main_429_hint_skipped_when_cookies_already_passed(
    monkeypatch, tmp_path, capsys
):
    err = "ERROR: HTTP Error 429: Too Many Requests"
    _run_main(
        monkeypatch, tmp_path, 1, err, False, ["--cookies-from-browser", "chrome"]
    )
    text = _main_output(capsys)
    assert "HTTP Error 429" in text
    assert "rate-limited" not in text


def test_main_clean_exit_no_subs_says_no_captions(monkeypatch, tmp_path, capsys):
    _run_main(monkeypatch, tmp_path, 0, "", False)
    text = _main_output(capsys)
    assert "No captions found" in text
    assert "429" not in text


def test_main_nonzero_but_subs_written_continues(monkeypatch, tmp_path):
    err = "ERROR: HTTP Error 429: Too Many Requests"
    out_dir = _run_main(monkeypatch, tmp_path, 1, err, True)
    sv.main()
    files = {p.name for p in out_dir.rglob("*") if p.is_file()}
    assert {"transcript.txt", "transcript.md", "metadata.json"} <= files


# --- transcript pipeline ---


def _vtt(tmp_path, body):
    path = tmp_path / "x.en.vtt"
    path.write_text(body, encoding="utf-8")
    return path


def test_ts_to_seconds():
    assert sv._ts_to_seconds("01:02:03.500") == 3723.5
    assert sv._ts_to_seconds("00:00:01,250") == 1.25
    assert sv._ts_to_seconds("garbage") == 0.0


def test_fmt_clock():
    assert sv._fmt_clock(0) == "00:00:00"
    assert sv._fmt_clock(59.9) == "00:00:59"
    assert sv._fmt_clock(3661) == "01:01:01"


def test_parse_vtt_drops_rolling_duplicate_lines(tmp_path):
    vtt = _vtt(
        tmp_path,
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:03.000\nhello there\n\n"
        "00:00:03.000 --> 00:00:05.000\nhello there\ngeneral kenobi\n",
    )
    assert sv.parse_vtt(vtt) == [(1.0, "hello there"), (3.0, "general kenobi")]


def test_parse_vtt_strips_tags_and_collapses_whitespace(tmp_path):
    vtt = _vtt(
        tmp_path,
        "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n"
        "<c.colorE5E5E5>hi</c>  <00:00:01.500><c>you</c>\n",
    )
    assert sv.parse_vtt(vtt) == [(1.0, "hi you")]


def test_parse_vtt_skips_header_note_and_reads_cue_ids(tmp_path):
    vtt = _vtt(
        tmp_path,
        "WEBVTT\nKind: captions\nLanguage: en\n\n"
        "NOTE a comment\n\n"
        "1\n00:00:02,000 --> 00:00:03,000\nfirst\n",
    )
    assert sv.parse_vtt(vtt) == [(2.0, "first")]


def test_parse_vtt_empty_file(tmp_path):
    assert sv.parse_vtt(_vtt(tmp_path, "WEBVTT\n")) == []


def test_format_transcript_empty():
    assert sv.format_transcript([]) == "(no transcript text)"


def test_format_transcript_marker_every_thirty_seconds():
    segs = [(0.0, "a"), (10.0, "b"), (31.0, "c")]
    assert sv.format_transcript(segs) == "[00:00:00]\na b\n\n[00:00:31]\nc\n"


def test_format_transcript_marker_at_exactly_thirty_seconds():
    out = sv.format_transcript([(0.0, "a"), (30.0, "b")])
    assert "[00:00:30]" in out


def test_group_paragraphs_breaks_after_sentence_once_past_soft_limit():
    segs = [(0, "a b c"), (2, "d."), (4, "e")]
    got = sv.group_paragraphs(segs, soft_words=3, hard_words=100)
    assert got == [(0, "a b c d."), (4, "e")]


def test_group_paragraphs_hard_limit_without_punctuation():
    segs = [(0, "a b"), (2, "c d"), (4, "e")]
    got = sv.group_paragraphs(segs, soft_words=100, hard_words=3)
    assert got == [(0, "a b c d"), (4, "e")]


def test_group_paragraphs_chapter_boundary_forces_break():
    segs = [(0, "a"), (10, "b"), (20, "c")]
    got = sv.group_paragraphs(segs, boundaries=[10])
    assert got == [(0, "a"), (10, "b c")]


def test_build_transcript_md_without_chapters():
    meta = {"title": "T", "uploader": "C", "duration": 65, "webpage_url": "http://x"}
    out = sv.build_transcript_md(meta, [(1.0, "hello")])
    assert out.startswith("# T\n\n[Watch](http://x) · C · 00:01:05\n")
    assert "**[00:00:01]** hello" in out
    assert "## [" not in out


def test_build_transcript_md_with_chapters_places_paragraphs_under_headings():
    meta = {
        "title": "T",
        "chapters": [
            {"start_time": 0, "end_time": 10, "title": "Intro"},
            {"start_time": 10, "title": "Main"},
        ],
    }
    out = sv.build_transcript_md(meta, [(1.0, "hi"), (12.0, "yo")])
    order = [
        out.index(x)
        for x in (
            "## [00:00:00] Intro",
            "**[00:00:01]** hi",
            "## [00:00:10] Main",
            "**[00:00:12]** yo",
        )
    ]
    assert order == sorted(order)


def test_build_transcript_md_empty():
    assert "_(no transcript text)_" in sv.build_transcript_md({"title": "T"}, [])


def test_slugify():
    assert sv.slugify("Hello, World!") == "hello-world"
    assert sv.slugify("  A_b  c--d ") == "a-b-c-d"
    assert len(sv.slugify("x" * 100)) == 60
    assert sv.slugify("!!!") == "video"


# --- sections: prompt contract ---


def test_every_section_prompt_starts_with_a_heading():
    for name, block in sv.SECTION_PROMPTS.items():
        assert block.startswith("## "), name


def test_default_sections_exclude_quotes():
    assert "quotes" not in sv.parse_sections(sv.DEFAULT_SECTIONS)


def test_env_var_sets_default_sections(monkeypatch):
    monkeypatch.setenv("VIDEO_SUMMARY_SECTIONS", "tldr")
    spec = importlib.util.spec_from_file_location("sv_fresh", sv.__file__)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    assert fresh.DEFAULT_SECTIONS == "tldr"


# --- main(): summary run, end to end with fakes ---


def _run_summary_main(monkeypatch, tmp_path, extra_args=()):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    calls = {}

    def fake_summarize(provider, model, meta, transcript, base_url, sections=None):
        calls.update(provider=provider, model=model, sections=sections)
        return "SUMMARY BODY", None

    monkeypatch.setattr(sv, "summarize", fake_summarize)
    out_dir = _run_main(
        monkeypatch,
        tmp_path,
        0,
        "",
        True,
        ["--model", "claude-sonnet-5", *extra_args],
        summary=True,
    )
    sv.main()
    folder = next(out_dir.iterdir())
    return folder, calls


def test_main_writes_summary_and_records_run_info(monkeypatch, tmp_path):
    folder, calls = _run_summary_main(monkeypatch, tmp_path)
    summary = (folder / "summary.md").read_text(encoding="utf-8")
    assert "SUMMARY BODY" in summary and "[Watch]" in summary
    info = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    default = sv.parse_sections(sv.DEFAULT_SECTIONS)
    assert calls["sections"] == default
    assert info["summary_sections"] == default
    assert info["tool_version"] == sv.__version__
    assert info["summary_provider"] == "anthropic"


def test_main_passes_chosen_sections_in_prompt_order(monkeypatch, tmp_path):
    folder, calls = _run_summary_main(
        monkeypatch, tmp_path, ["--sections", "quotes,tldr"]
    )
    assert calls["sections"] == ["tldr", "quotes"]
    info = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert info["summary_sections"] == ["tldr", "quotes"]


def test_main_invalid_sections_exits_before_any_download(monkeypatch, tmp_path, capsys):
    def boom(*a, **k):
        raise AssertionError("network or subprocess touched")

    monkeypatch.setattr(sv.subprocess, "run", boom)
    monkeypatch.setattr(sv, "check_yt_dlp", boom)
    out_dir = tmp_path / "out"
    argv = ["summarize_video.py", "http://x", "--out", str(out_dir), "--sections", "x"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as e:
        sv.main()
    assert e.value.code == 1
    assert "Valid names" in capsys.readouterr().err
    assert not out_dir.exists()


def test_main_no_summary_writes_no_summary_file(monkeypatch, tmp_path):
    out_dir = _run_main(monkeypatch, tmp_path, 0, "", True)
    sv.main()
    folder = next(out_dir.iterdir())
    assert not (folder / "summary.md").exists()
    info = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert info["summary_sections"] is None


def test_main_missing_api_key_warns_and_keeps_transcript(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(sv, "summarize", lambda *a, **k: pytest.fail("called"))
    out_dir = _run_main(
        monkeypatch, tmp_path, 0, "", True, ["--model", "claude-sonnet-5"], summary=True
    )
    sv.main()
    folder = next(out_dir.iterdir())
    assert (folder / "transcript.txt").exists()
    assert not (folder / "summary.md").exists()
    assert "ANTHROPIC_API_KEY not set" in capsys.readouterr().err


# --- version and man page ---


def test_version_flag_prints_version(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["summarize_video.py", "--version"])
    with pytest.raises(SystemExit) as e:
        sv.main()
    assert e.value.code == 0
    assert sv.__version__ in capsys.readouterr().out


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", sv.__version__)


MAN_PAGE = Path(sv.__file__).parent / "man" / "summarize_video.1"


def test_man_page_documents_every_flag(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["summarize_video.py", "--help"])
    with pytest.raises(SystemExit):
        sv.main()
    flags = re.findall(r"^\s+(--[a-z][a-z-]*)", capsys.readouterr().out, re.M)
    assert "--sections" in flags and "--version" in flags
    man = MAN_PAGE.read_text(encoding="utf-8")
    for flag in flags:
        assert flag.replace("-", r"\-") in man, flag


def test_man_page_version_matches_code():
    first_line = MAN_PAGE.read_text(encoding="utf-8").splitlines()[0]
    assert f"video-summarizer {sv.__version__}" in first_line
