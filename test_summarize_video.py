import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

sv = importlib.import_module("summarize_video")

_ENV_VARS = (
    "VIDEO_SUMMARY_DIR",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "VIDEO_SUMMARY_MODEL",
    "VIDEO_SUMMARY_SECTIONS",
    "VIDEO_SUMMARY_COOKIES_BROWSER",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENV_VARS:
        # setenv first so teardown restores the original even if main() sets it.
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)


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


_DEFAULT_META = {"title": "T", "id": "abc", "upload_date": "20260101"}


def _run_main(
    monkeypatch,
    tmp_path,
    sub_rc,
    sub_stderr,
    write_vtt,
    extra_args=(),
    summary=False,
    meta=None,
):
    meta_json = json.dumps(_DEFAULT_META if meta is None else meta)

    def fake_run(cmd, **kwargs):
        if "-J" in cmd:
            return SimpleNamespace(returncode=0, stdout=meta_json, stderr="")
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
    _run_main(monkeypatch, tmp_path, 1, err, False, ["--no-cookies"])
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
    _run_main(monkeypatch, tmp_path, 1, err, False, ["--no-cookies"])
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


def test_main_429_hint_skipped_with_default_cookies(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sv, "DEFAULT_COOKIES_BROWSER", "chrome")
    _run_main(
        monkeypatch, tmp_path, 1, "ERROR: HTTP Error 429: Too Many Requests", False
    )
    text = _main_output(capsys)
    assert "HTTP Error 429" in text
    assert "rate-limited" not in text


def _record_cmds(monkeypatch):
    cmds = []
    run = sv.subprocess.run

    def recording_run(cmd, **kwargs):
        cmds.append(cmd)
        return run(cmd, **kwargs)

    monkeypatch.setattr(sv.subprocess, "run", recording_run)
    return cmds


def test_main_sends_default_browser_cookies(monkeypatch, tmp_path):
    monkeypatch.setattr(sv, "DEFAULT_COOKIES_BROWSER", "chrome")
    _run_main(monkeypatch, tmp_path, 0, "", True)
    cmds = _record_cmds(monkeypatch)
    sv.main()
    assert len(cmds) == 2
    for cmd in cmds:
        i = cmd.index("--cookies-from-browser")
        assert cmd[i + 1] == "chrome"


def test_main_no_cookies_sends_none(monkeypatch, tmp_path):
    monkeypatch.setattr(sv, "DEFAULT_COOKIES_BROWSER", "chrome")
    _run_main(monkeypatch, tmp_path, 0, "", True, ["--no-cookies"])
    cmds = _record_cmds(monkeypatch)
    sv.main()
    assert len(cmds) == 2
    assert all("--cookies-from-browser" not in cmd for cmd in cmds)


def test_default_cookies_browser_is_chrome_unless_env_overrides(monkeypatch):
    try:
        importlib.reload(sv)
        assert sv.DEFAULT_COOKIES_BROWSER == "chrome"
        monkeypatch.setenv("VIDEO_SUMMARY_COOKIES_BROWSER", "firefox")
        importlib.reload(sv)
        assert sv.DEFAULT_COOKIES_BROWSER == "firefox"
        monkeypatch.setenv("VIDEO_SUMMARY_COOKIES_BROWSER", "")
        importlib.reload(sv)
        assert sv.DEFAULT_COOKIES_BROWSER == ""
    finally:
        monkeypatch.delenv("VIDEO_SUMMARY_COOKIES_BROWSER", raising=False)
        importlib.reload(sv)


_NO_COOKIE_DB = (
    'ERROR: could not find vivaldi cookies database in "/Users/x/Library/'
    'Application Support/Vivaldi"'
)
_BOT_CHECK = (
    "ERROR: [youtube] abc: Sign in to confirm you're not a bot. Use "
    "--cookies-from-browser or --cookies for the authentication."
)


@pytest.mark.parametrize(
    "err",
    [
        _NO_COOKIE_DB,
        "ERROR: failed to load cookies",
        'yt-dlp: error: unsupported browser specified for cookies: "vivaldy".',
        "ERROR: unknown browser: vivaldy",
        "ERROR: failed to read from keyring",
    ],
)
def test_cookie_hint_on_cookie_read_failure(err):
    hint = sv._cookie_hint(err, "vivaldi")
    assert "vivaldi" in hint and "--no-cookies" in hint


@pytest.mark.parametrize(
    "err", [_BOT_CHECK, "ERROR: HTTP Error 429: Too Many Requests"]
)
def test_cookie_hint_ignores_other_errors(err):
    assert sv._cookie_hint(err, "chrome") == ""


def test_cookie_hint_needs_a_browser():
    assert sv._cookie_hint(_NO_COOKIE_DB, None) == ""


def test_main_cookie_read_failure_in_subs_shows_hint(monkeypatch, tmp_path, capsys):
    _run_main(
        monkeypatch,
        tmp_path,
        1,
        _NO_COOKIE_DB,
        False,
        ["--cookies-from-browser", "vivaldi"],
    )
    text = _main_output(capsys)
    assert "could not find vivaldi cookies database" in text
    assert "Could not read vivaldi's cookies" in text and "--no-cookies" in text


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


def _run_summary_main(
    monkeypatch, tmp_path, extra_args=(), meta=None, usage=None, model="claude-sonnet-5"
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    calls = {}

    def fake_summarize(provider, model, meta, transcript, base_url, sections=None):
        calls.update(
            provider=provider,
            model=model,
            sections=sections,
            base_url=base_url,
            openai_key=os.environ.get("OPENAI_API_KEY"),
        )
        return "SUMMARY BODY", usage

    monkeypatch.setattr(sv, "summarize", fake_summarize)
    out_dir = _run_main(
        monkeypatch,
        tmp_path,
        0,
        "",
        True,
        ["--model", model, *extra_args],
        summary=True,
        meta=meta,
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


# --- mutation-hardening: provider, pricing, prompt plumbing ---


def test_resolve_provider_explicit_anthropic_beats_non_claude_model():
    assert sv.resolve_provider("gpt-4o", "anthropic", None) == "anthropic"


def test_resolve_provider_auto_is_compared_by_value():
    auto = "".join(["au", "to"])
    assert sv.resolve_provider("claude-sonnet-5", auto, None) == "anthropic"
    assert sv.resolve_provider("gpt-4o", auto, None) == "openai"


def test_prices_table_is_exact():
    assert sv.PRICES == {
        "claude-opus-5": (5.0, 25.0),
        "claude-sonnet-5": (2.0, 10.0),
        "claude-haiku-4-5": (1.0, 5.0),
        "gpt-4o": (2.5, 10.0),
        "gpt-4o-mini": (0.15, 0.6),
        "gpt-4.1-mini": (0.4, 1.6),
    }


def test_estimate_cost_every_priced_model_with_uneven_counts():
    tin, tout = 1_234_567, 7_654
    for model, (pin, pout) in sv.PRICES.items():
        want = (tin * pin + tout * pout) / 1_000_000
        assert sv.estimate_cost(model, (tin, tout), None) == (
            f"${want:.3f} ({tin} in / {tout} out)"
        ), model
    assert sv.estimate_cost("claude-sonnet-5", (tin, tout), None) == (
        "$2.546 (1234567 in / 7654 out)"
    )


def test_estimate_cost_divides_by_exactly_one_million():
    # 1500 and 4500 tokens sit on rounding boundaries that a divisor of
    # 1_000_001 or 999_999 would tip the other way.
    assert sv.estimate_cost("m", (1500, 0), (1.0, 0.0)) == "$0.002 (1500 in / 0 out)"
    assert sv.estimate_cost("m", (4500, 0), (1.0, 0.0)) == "$0.004 (4500 in / 0 out)"


def test_summarize_anthropic_ignores_non_text_blocks(monkeypatch):
    content = [
        SimpleNamespace(type="thinking", thinking="hmm"),
        SimpleNamespace(type="text", text="  the answer "),
        SimpleNamespace(type="redacted_thinking", data="zz"),
        SimpleNamespace(type="tool_use", id="t"),
    ]
    response = SimpleNamespace(stop_reason="end_turn", content=content, usage=None)
    _install_fake_anthropic(monkeypatch, response, {})
    text, _ = sv.summarize_anthropic("claude-sonnet-5", {}, "body")
    assert text == "the answer"


@pytest.mark.parametrize(
    "reason", ["end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn"]
)
def test_summarize_anthropic_only_refusal_exits(monkeypatch, reason):
    response = SimpleNamespace(
        stop_reason=reason,
        content=[SimpleNamespace(type="text", text="ok")],
        usage=None,
    )
    _install_fake_anthropic(monkeypatch, response, {})
    assert sv.summarize_anthropic("claude-sonnet-5", {}, "body")[0] == "ok"


def test_summarize_anthropic_refusal_message_names_category(monkeypatch, capsys):
    response = SimpleNamespace(
        stop_reason="refusal",
        stop_details=SimpleNamespace(category="weapons"),
        content=[],
        usage=None,
    )
    _install_fake_anthropic(monkeypatch, response, {})
    with pytest.raises(SystemExit):
        sv.summarize_anthropic("claude-sonnet-5", {}, "body")
    assert "category: weapons" in capsys.readouterr().err


def test_summarize_anthropic_refusal_without_details(monkeypatch, capsys):
    response = SimpleNamespace(stop_reason="refusal", content=[], usage=None)
    _install_fake_anthropic(monkeypatch, response, {})
    with pytest.raises(SystemExit):
        sv.summarize_anthropic("claude-sonnet-5", {}, "body")
    assert "category: None" in capsys.readouterr().err


def test_summarize_anthropic_request_parameters(monkeypatch):
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="ok")],
        usage=None,
    )
    recorder = {}
    _install_fake_anthropic(monkeypatch, response, recorder)
    sv.summarize_anthropic("claude-opus-5", {}, "body")
    assert recorder["model"] == "claude-opus-5"
    assert recorder["max_tokens"] == 16000
    assert recorder["thinking"] == {"type": "adaptive"}


def test_summarize_openai_request_parameters(monkeypatch):
    recorder = {}
    _install_fake_openai(monkeypatch, [_sse("x"), _sse(finish_reason="stop")], recorder)
    sv.summarize_openai("gpt-4o-mini", {}, "body", None)
    assert recorder["model"] == "gpt-4o-mini"
    assert recorder["max_tokens"] == 16000
    assert recorder["stream"] is True
    assert recorder["stream_options"] == {"include_usage": True}


def test_summarize_openai_reads_base_url_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://env-host/v1")
    recorder = {}
    _install_fake_openai(monkeypatch, [_sse("x")], recorder)
    sv.summarize_openai("llama3", {}, "body", None)
    assert recorder["base_url"] == "http://env-host/v1"


def test_summarize_openai_arg_base_url_beats_env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://env-host/v1")
    recorder = {}
    _install_fake_openai(monkeypatch, [_sse("x")], recorder)
    sv.summarize_openai("llama3", {}, "body", "http://arg-host/v1")
    assert recorder["base_url"] == "http://arg-host/v1"


def test_summarize_openai_empty_env_base_url_is_none(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    recorder = {}
    _install_fake_openai(monkeypatch, [_sse("x")], recorder)
    sv.summarize_openai("gpt-4o-mini", {}, "body", None)
    assert recorder["base_url"] is None


def test_summarize_openai_stream_without_finish_reason(monkeypatch):
    events = [_sse("part one "), _sse("part two"), _usage_event(1, 2)]
    _install_fake_openai(monkeypatch, events, {})
    text, usage = sv.summarize_openai("gpt-4o-mini", {}, "body", None)
    assert text == "part one part two"
    assert usage.completion_tokens == 2


# --- mutation-hardening: prompt and transcript text ---


def test_parse_sections_error_names_unknown_sections(capsys):
    with pytest.raises(SystemExit):
        sv.parse_sections("tldr,nope,zip")
    err = capsys.readouterr().err
    assert "(unknown: nope, zip)" in err
    assert "(unknown: tldr" not in err


def test_parse_sections_empty_error_has_no_unknown_clause(capsys):
    with pytest.raises(SystemExit):
        sv.parse_sections(" , ")
    assert "unknown" not in capsys.readouterr().err


def test_build_user_content_channel_prefers_uploader():
    out = sv.build_user_content({"uploader": "U", "channel": "C"}, "body")
    assert "Channel: U\n" in out


def test_build_user_content_channel_falls_back_to_channel():
    out = sv.build_user_content({"channel": "Chan"}, "body")
    assert "Channel: Chan\n" in out


def test_build_user_content_missing_metadata():
    out = sv.build_user_content({}, "body")
    assert "Channel: None\n" in out
    assert "Duration: 00:00:00\n" in out


def test_build_transcript_md_header_without_metadata():
    out = sv.build_transcript_md({}, [(1.0, "hi")])
    assert out.startswith("# Video\n\n[Watch]() \u00b7  \u00b7 00:00:00\n")


def test_build_transcript_md_title_falls_back_to_id():
    assert sv.build_transcript_md({"id": "abc"}, []).startswith("# abc\n")


def test_build_transcript_md_channel_fallback():
    only_channel = sv.build_transcript_md({"channel": "Chan"}, [])
    assert "\u00b7 Chan \u00b7" in only_channel
    both = sv.build_transcript_md({"uploader": "U", "channel": "C"}, [])
    assert "\u00b7 U \u00b7" in both


def _chapter_sections(md):
    parts = re.split(r"^## ", md, flags=re.M)
    return {p.split("\n")[0]: p for p in parts[1:]}


def test_chapter_slicing_is_start_inclusive_end_exclusive():
    meta = {
        "title": "T",
        "chapters": [
            {"start_time": 5, "end_time": 10, "title": "A"},
            {"start_time": 10, "end_time": 20, "title": "B"},
        ],
    }
    segs = [(1.0, "before"), (5.0, "at-a"), (10.0, "at-b"), (15.0, "mid-b")]
    out = sv.build_transcript_md(meta, segs)
    assert "before" not in "".join(_chapter_sections(out).values())
    sections = _chapter_sections(out)
    a, b = sections["[00:00:05] A"], sections["[00:00:10] B"]
    assert "at-a" in a and "at-b" not in a
    assert "at-b" in b and "mid-b" in b and "at-a" not in b


# --- mutation-hardening: format_transcript / group_paragraphs ---


def test_format_transcript_no_marker_just_under_thirty_seconds():
    assert sv.format_transcript([(0.0, "a"), (29.0, "b")]) == "[00:00:00]\na b\n"


def test_group_paragraphs_default_soft_limit_is_seventy_words():
    segs = [(i, "w.") for i in range(75)]
    got = sv.group_paragraphs(segs)
    assert len(got[0][1].split()) == 70


def test_group_paragraphs_default_hard_limit_is_110_words():
    segs = [(i, "w") for i in range(120)]
    got = sv.group_paragraphs(segs)
    assert len(got[0][1].split()) == 110


def test_group_paragraphs_breaks_at_exactly_soft_words():
    segs = [(0, "a"), (1, "b."), (2, "c")]
    got = sv.group_paragraphs(segs, soft_words=2, hard_words=100)
    assert got == [(0, "a b."), (2, "c")]


def test_group_paragraphs_no_break_before_soft_words():
    segs = [(0, "a."), (1, "b"), (2, "c")]
    got = sv.group_paragraphs(segs, soft_words=2, hard_words=100)
    assert got == [(0, "a. b c")]


def test_group_paragraphs_breaks_at_exactly_hard_words():
    segs = [(0, "a b"), (2, "c"), (4, "d")]
    got = sv.group_paragraphs(segs, soft_words=100, hard_words=3)
    assert got == [(0, "a b c"), (4, "d")]


def test_parse_vtt_skips_cue_less_block_and_leading_blank_lines(tmp_path):
    vtt = _vtt(
        tmp_path,
        "\n\nWEBVTT\n\nstray text with no timing\n\n"
        "00:00:01.000 --> 00:00:02.000\nhello\n",
    )
    assert sv.parse_vtt(vtt) == [(1.0, "hello")]


def test_parse_vtt_tag_only_line_is_skipped_not_terminal(tmp_path):
    vtt = _vtt(
        tmp_path,
        "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<c></c>\nreal text\n",
    )
    assert sv.parse_vtt(vtt) == [(1.0, "real text")]


def test_parse_vtt_repeated_first_line_then_new_text(tmp_path):
    vtt = _vtt(
        tmp_path,
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:02.000\none\n\n"
        "00:00:02.000 --> 00:00:03.000\none\ntwo\nthree\n",
    )
    assert sv.parse_vtt(vtt) == [(1.0, "one"), (2.0, "two"), (2.0, "three")]


# --- mutation-hardening: yt-dlp wrappers ---


def test_check_yt_dlp_dies_when_missing(monkeypatch, capsys):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(SystemExit) as e:
        sv.check_yt_dlp()
    assert e.value.code == 1
    assert "yt-dlp not found" in capsys.readouterr().err


def test_check_yt_dlp_passes_when_present(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/yt-dlp")
    assert sv.check_yt_dlp() is None


def test_cookie_args():
    assert sv._cookie_args("chrome") == ["--cookies-from-browser", "chrome"]
    assert sv._cookie_args(None) == []
    assert sv._cookie_args("") == []


def test_fetch_metadata_runs_ytdlp_and_parses_json(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(cmd=cmd, kwargs=kwargs)
        return SimpleNamespace(stdout='{"title": "T"}')

    monkeypatch.setattr(sv.subprocess, "run", fake_run)
    assert sv.fetch_metadata("http://x", "firefox") == {"title": "T"}
    assert seen["kwargs"] == {"capture_output": True, "text": True, "check": True}
    assert seen["cmd"][0] == "yt-dlp" and seen["cmd"][-1] == "http://x"
    assert "--cookies-from-browser" in seen["cmd"] and "firefox" in seen["cmd"]


def test_fetch_metadata_dies_with_ytdlp_stderr(monkeypatch, capsys):
    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr="  Video unavailable\n")

    monkeypatch.setattr(sv.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as e:
        sv.fetch_metadata("http://x")
    assert e.value.code == 1
    err = capsys.readouterr().err
    assert "could not read the video" in err and "Video unavailable" in err


def test_fetch_metadata_cookie_read_failure_shows_hint(monkeypatch, capsys):
    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr=_NO_COOKIE_DB)

    monkeypatch.setattr(sv.subprocess, "run", fake_run)
    with pytest.raises(SystemExit):
        sv.fetch_metadata("http://x", "vivaldi")
    err = capsys.readouterr().err
    assert "Could not read vivaldi's cookies" in err and "--no-cookies" in err


def test_fetch_metadata_bot_check_gets_no_cookie_hint(monkeypatch, capsys):
    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr=_BOT_CHECK)

    monkeypatch.setattr(sv.subprocess, "run", fake_run)
    with pytest.raises(SystemExit):
        sv.fetch_metadata("http://x", "chrome")
    err = capsys.readouterr().err
    assert "Sign in to confirm" in err and "Could not read" not in err


def _fake_subs_run(monkeypatch, tmp_path, names, returncode=0, stderr=""):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(cmd=cmd, kwargs=kwargs)
        for n in names:
            (tmp_path / n).write_text(_VTT, encoding="utf-8")
        return SimpleNamespace(returncode=returncode, stdout="", stderr=stderr)

    monkeypatch.setattr(sv.subprocess, "run", fake_run)
    return seen


def test_download_subs_run_options(monkeypatch, tmp_path):
    seen = _fake_subs_run(monkeypatch, tmp_path, ["abc.en.vtt"])
    assert sv.download_subs("http://x", tmp_path, "chrome") == tmp_path / "abc.en.vtt"
    assert seen["kwargs"] == {"capture_output": True, "text": True, "check": False}
    assert "chrome" in seen["cmd"]


def test_download_subs_prefers_plain_en_over_variants(monkeypatch, tmp_path):
    names = ["abc.en-US.vtt", "abc.en-orig.vtt", "abc.en.vtt"]
    _fake_subs_run(monkeypatch, tmp_path, names)
    assert sv.download_subs("http://x", tmp_path).name == "abc.en.vtt"


def test_download_subs_without_plain_en_returns_first_sorted(monkeypatch, tmp_path):
    _fake_subs_run(monkeypatch, tmp_path, ["abc.en-orig.vtt", "abc.en-US.vtt"])
    assert sv.download_subs("http://x", tmp_path).name == "abc.en-US.vtt"


def test_download_subs_single_non_plain_track(monkeypatch, tmp_path):
    _fake_subs_run(monkeypatch, tmp_path, ["abc.en-GB.vtt"])
    assert sv.download_subs("http://x", tmp_path).name == "abc.en-GB.vtt"


def test_download_subs_signal_exit_without_files_dies(monkeypatch, tmp_path, capsys):
    _fake_subs_run(monkeypatch, tmp_path, [], returncode=-9)
    with pytest.raises(SystemExit):
        sv.download_subs("http://x", tmp_path)
    assert "exited with -9" in capsys.readouterr().err


def test_download_subs_clean_exit_without_files_is_none(monkeypatch, tmp_path):
    _fake_subs_run(monkeypatch, tmp_path, [])
    assert sv.download_subs("http://x", tmp_path) is None


# --- mutation-hardening: main() output and metadata ---


def _read_info(folder):
    return json.loads((folder / "metadata.json").read_text(encoding="utf-8"))


def _run_no_summary(monkeypatch, tmp_path, meta):
    out_dir = _run_main(monkeypatch, tmp_path, 0, "", True, meta=meta)
    sv.main()
    return next(out_dir.iterdir())


def test_folder_name_uses_upload_date_and_title_slug(monkeypatch, tmp_path):
    meta = {"title": "Hello World", "id": "abc", "upload_date": "20240315"}
    folder = _run_no_summary(monkeypatch, tmp_path, meta)
    assert folder.name == "2024-03-15_hello-world"
    assert _read_info(folder)["upload_date"] == "2024-03-15"


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2030, 5, 6, 7, 8, 9, tzinfo=tz)


@pytest.mark.parametrize("upload", [None, "", "2024031", "202403155"])
def test_folder_name_falls_back_to_today(monkeypatch, tmp_path, upload):
    monkeypatch.setattr(sv, "datetime", _FixedDatetime)
    meta = {"title": "T", "id": "abc"}
    if upload is not None:
        meta["upload_date"] = upload
    folder = _run_no_summary(monkeypatch, tmp_path, meta)
    assert folder.name == "2030-05-06_t"


def test_main_can_rerun_into_existing_folder(monkeypatch, tmp_path):
    out_dir = _run_main(monkeypatch, tmp_path, 0, "", True)
    sv.main()
    sv.main()
    assert len(list(out_dir.iterdir())) == 1


def test_main_title_falls_back_to_id_then_video(monkeypatch, tmp_path, capsys):
    folder = _run_no_summary(monkeypatch, tmp_path, {"id": "vid42"})
    assert folder.name.endswith("_vid42")
    assert _read_info(folder)["title"] == "vid42"
    assert "\n  vid42\n" in capsys.readouterr().out


def test_main_title_wins_over_id(monkeypatch, tmp_path):
    folder = _run_no_summary(monkeypatch, tmp_path, {"title": "T", "id": "vid42"})
    assert _read_info(folder)["title"] == "T"


def test_main_title_falls_back_to_video_without_id(monkeypatch, tmp_path):
    folder = _run_no_summary(monkeypatch, tmp_path, {"upload_date": "20240101"})
    assert folder.name == "2024-01-01_video"
    assert _read_info(folder)["title"] == "video"


def test_main_metadata_without_optional_fields(monkeypatch, tmp_path):
    info = _read_info(_run_no_summary(monkeypatch, tmp_path, {"title": "T"}))
    assert info["channel"] is None
    assert info["duration_seconds"] is None
    assert info["duration"] == "00:00:00"
    assert info["video_id"] is None


def test_main_metadata_channel_and_duration(monkeypatch, tmp_path):
    both = {"title": "T", "uploader": "U", "channel": "C", "duration": 3725}
    info = _read_info(_run_no_summary(monkeypatch, tmp_path, both))
    assert info["channel"] == "U"
    assert info["duration"] == "01:02:05"
    only_channel = {"title": "T", "channel": "C"}
    info = _read_info(_run_no_summary(monkeypatch, tmp_path / "b", only_channel))
    assert info["channel"] == "C"


def test_main_metadata_keeps_non_ascii_and_two_space_indent(monkeypatch, tmp_path):
    folder = _run_no_summary(monkeypatch, tmp_path, {"title": "Caf\u00e9 \u2615"})
    raw = (folder / "metadata.json").read_text(encoding="utf-8")
    assert '"title": "Caf\u00e9 \u2615"' in raw
    assert raw.startswith('{\n  "tool_version"')


def test_main_counts_only_chapters_with_a_start_time(monkeypatch, tmp_path, capsys):
    meta = {
        "title": "T",
        "chapters": [
            {"start_time": 0, "end_time": 5, "title": "One"},
            {"start_time": 5, "title": "Two"},
            {"start_time": None, "title": "Ghost"},
        ],
    }
    folder = _run_no_summary(monkeypatch, tmp_path, meta)
    assert _read_info(folder)["chapters"] == 2
    assert "Transcript saved: 1 segments, 2 chapters\n" in capsys.readouterr().out
    assert "Ghost" not in (folder / "transcript.md").read_text(encoding="utf-8")


def test_main_reports_no_chapters(monkeypatch, tmp_path, capsys):
    folder = _run_no_summary(monkeypatch, tmp_path, {"title": "T"})
    assert _read_info(folder)["chapters"] == 0
    assert "Transcript saved: 1 segments, no chapters\n" in capsys.readouterr().out


def test_main_records_summary_model_only_when_summarizing(monkeypatch, tmp_path):
    folder = _run_no_summary(monkeypatch, tmp_path, {"title": "T"})
    assert _read_info(folder)["summary_model"] is None
    assert _read_info(folder)["summary_provider"] is None


def test_main_summary_run_records_model(monkeypatch, tmp_path):
    folder, _ = _run_summary_main(monkeypatch, tmp_path)
    assert _read_info(folder)["summary_model"] == "claude-sonnet-5"


def test_main_summary_line_shows_cost_when_usage_present(monkeypatch, tmp_path, capsys):
    usage = SimpleNamespace(input_tokens=1_000_000, output_tokens=1_000_000)
    _run_summary_main(monkeypatch, tmp_path, usage=usage)
    out = capsys.readouterr().out
    assert "Summary saved (~$12.000 (1000000 in / 1000000 out))\n" in out


def test_main_summary_line_has_no_cost_without_usage(monkeypatch, tmp_path, capsys):
    _run_summary_main(monkeypatch, tmp_path)
    assert "Summary saved\n" in capsys.readouterr().out


def test_main_price_override_needs_both_flags(monkeypatch, tmp_path, capsys):
    usage = SimpleNamespace(input_tokens=1_000_000, output_tokens=1_000_000)
    args = ["--provider", "anthropic", "--price-in", "1.5", "--price-out", "2.5"]
    _run_summary_main(monkeypatch, tmp_path, args, usage=usage, model="mystery")
    assert "Summary saved (~$4.000 (1000000 in / 1000000 out))\n" in (
        capsys.readouterr().out
    )


@pytest.mark.parametrize("flag", ["--price-in", "--price-out"])
def test_main_single_price_flag_is_ignored(monkeypatch, tmp_path, capsys, flag):
    usage = SimpleNamespace(input_tokens=1_000_000, output_tokens=1_000_000)
    args = ["--provider", "anthropic", flag, "1.5"]
    _run_summary_main(monkeypatch, tmp_path, args, usage=usage, model="mystery")
    assert "Summary saved\n" in capsys.readouterr().out


# --- mutation-hardening: API key and base URL handling in main() ---


def test_main_base_url_flag_sets_placeholder_key(monkeypatch, tmp_path):
    args = ["--base-url", "http://localhost:11434/v1"]
    _, calls = _run_summary_main(monkeypatch, tmp_path, args, model="llama3")
    assert calls["provider"] == "openai"
    assert calls["openai_key"] == "not-needed"


def test_main_env_base_url_sets_placeholder_key(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    _, calls = _run_summary_main(monkeypatch, tmp_path, model="llama3")
    assert calls["openai_key"] == "not-needed"


def test_main_real_openai_key_is_not_overwritten(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    args = ["--base-url", "http://localhost:11434/v1"]
    _, calls = _run_summary_main(monkeypatch, tmp_path, args, model="llama3")
    assert calls["openai_key"] == "sk-real"


def test_main_no_placeholder_without_base_url(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sv, "summarize", lambda *a, **k: pytest.fail("called"))
    out_dir = _run_main(
        monkeypatch,
        tmp_path,
        0,
        "",
        True,
        ["--provider", "openai", "--model", "gpt-4o-mini"],
        summary=True,
    )
    sv.main()
    assert "OPENAI_API_KEY not set" in capsys.readouterr().err
    assert "OPENAI_API_KEY" not in os.environ
    assert not (next(out_dir.iterdir()) / "summary.md").exists()


def test_main_openai_provider_checks_openai_key_not_anthropic(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(sv, "summarize", lambda *a, **k: pytest.fail("called"))
    out_dir = _run_main(
        monkeypatch,
        tmp_path,
        0,
        "",
        True,
        ["--provider", "openai", "--model", "gpt-4o-mini"],
        summary=True,
    )
    sv.main()
    assert "OPENAI_API_KEY not set" in capsys.readouterr().err
    assert not (next(out_dir.iterdir()) / "summary.md").exists()


def test_main_anthropic_end_to_end_with_provider_flag(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="REAL BODY")],
        usage=SimpleNamespace(input_tokens=1_000_000, output_tokens=0),
    )
    _install_fake_anthropic(monkeypatch, response, {})
    provider = "".join(["anthro", "pic"])
    out_dir = _run_main(
        monkeypatch,
        tmp_path,
        0,
        "",
        True,
        ["--provider", provider, "--model", "claude-sonnet-5"],
        summary=True,
    )
    sv.main()
    folder = next(out_dir.iterdir())
    assert "REAL BODY" in (folder / "summary.md").read_text(encoding="utf-8")
    assert "(~$2.000 (1000000 in / 0 out))" in capsys.readouterr().out


# --- mutation-hardening: script entry point ---


def test_script_version_flag_runs_as_a_program():
    proc = subprocess.run(
        [sys.executable, sv.__file__, "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0
    assert sv.__version__ in proc.stdout
