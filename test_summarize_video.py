import importlib
import sys
from types import SimpleNamespace

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
    fake_mod = SimpleNamespace(Anthropic=lambda: _FakeAnthropicClient(response, recorder))
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
    import pytest
    with pytest.raises(SystemExit):
        sv.summarize_anthropic("claude-sonnet-5", {"title": "T", "duration": 0}, "body")
