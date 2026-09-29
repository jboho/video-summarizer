import importlib

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
