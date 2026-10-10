import importlib
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
        lambda m, meta, t: (called.setdefault("p", "a"), None),
    )
    monkeypatch.setattr(
        sv,
        "summarize_openai",
        lambda m, meta, t, b: (called.setdefault("p", "o"), None),
    )
    text, _ = sv.summarize("anthropic", "claude-sonnet-5", {}, "body", None)
    assert called["p"] == "a" and text == "a"


def test_summarize_dispatches_to_openai(monkeypatch):
    called = {}
    monkeypatch.setattr(
        sv,
        "summarize_anthropic",
        lambda m, meta, t: (called.setdefault("p", "a"), None),
    )
    monkeypatch.setattr(
        sv,
        "summarize_openai",
        lambda m, meta, t, b: (called.setdefault("p", "o"), None),
    )
    text, _ = sv.summarize("openai", "gpt-4o-mini", {}, "body", "http://h/v1")
    assert called["p"] == "o" and text == "o"


# --- main(): subtitle download failure handling ---

_VTT = "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nhello there\n"


def _run_main(monkeypatch, tmp_path, sub_rc, sub_stderr, write_vtt, extra_args=()):
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
    argv = ["summarize_video.py", "http://x", "--no-summary", "--out", str(out_dir)]
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
