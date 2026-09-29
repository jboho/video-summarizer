# Multi-provider Summarization Implementation Plan

> **Execution:** hand off to the `run-plan` skill to implement this task-by-task (fresh subagent per task + spec/quality review). Steps use `- [ ]` checkboxes for tracking.

**Goal:** Let `summarize_video.py` summarize with any OpenAI-compatible model (OpenAI, OpenRouter, Groq, local Ollama/LM Studio/vLLM) while keeping the native Claude path as the built-for default.

**Architecture:** Two summarizer adapters behind one `summarize()` dispatcher — `summarize_anthropic()` (today's code, unchanged behavior) and `summarize_openai()` (OpenAI Chat Completions, which every compatible provider speaks). The provider is chosen by `--provider`, an explicit `--base-url`, or inferred from the model name. Everything upstream (transcript fetch/format) and the prompts are already provider-neutral and stay as-is.

**Tech stack:** Python 3.11 single-file `uv` script (PEP 723 inline deps), `anthropic` + `openai` SDKs, `pytest` (run via `uv run`), `yt-dlp` (unchanged).

---

## Why two adapters, not a router (LiteLLM)

LiteLLM would reach ~100 providers through one call but is a heavy dependency and hides the two Anthropic-specific features this script uses (adaptive thinking, the `refusal` stop reason). That fights the repo's lean single-file style. If you would rather have one unified call and drop those niceties, that is the single decision to flip before starting — it collapses Tasks 4–6 into one adapter. This plan assumes two adapters.

## File structure

| File | Responsibility | Change |
|------|----------------|--------|
| `summarize_video.py` | The tool. Provider resolution, two adapters, dispatcher, usage/cost. | Modify |
| `test_summarize_video.py` | Unit tests for the pure logic + adapters (with fake SDK clients, no network). | Create |
| `README.md` | Add a "Using other models" section. | Modify |

No new modules — the tool stays one file. Tests import it as a module (`main()` is already guarded by `if __name__ == "__main__"`).

## Provider-parity notes (bake into the tasks)

- **Adaptive thinking** and **`stop_reason == "refusal"`** are Anthropic-only. The OpenAI path omits thinking and maps a `content_filter` finish reason (or a `delta.refusal`) to the same "model declined" error via `die()`.
- **System prompt**: Anthropic takes a top-level `system=`; OpenAI takes a leading `{"role": "system"}` message.
- **Usage field names differ** (`input_tokens`/`output_tokens` vs `prompt_tokens`/`completion_tokens`). Normalize to an `(in, out)` tuple before costing.
- **`max_tokens`**: kept for broad compatibility (Ollama/OpenRouter/older OpenAI). Known limitation: newer OpenAI reasoning models (o1/o3) reject it in favor of `max_completion_tokens`; documented, not handled in v1.

## Test setup

The repo has no test harness today. Tests run without touching the network, using fake SDK clients injected via `sys.modules`. Run them with:

```bash
uv run --with pytest --with anthropic --with openai pytest test_summarize_video.py -q
```

---

### Task 1: Add the `openai` dependency and provider CLI flags

Config/scaffolding task — no test (a dependency line and argparse wiring; exercised by later tasks).

**Files:**
- Modify: `summarize_video.py:4` (inline deps), `summarize_video.py:378-388` (`main` args)

- [ ] **Step 1: Add `openai` to inline deps**

Change line 4 from:

```python
# dependencies = ["anthropic>=1.0"]
```

to:

```python
# dependencies = ["anthropic>=1.0", "openai>=1.0"]
```

- [ ] **Step 2: Add the new arguments in `main()`**

After the existing `--out` argument (line 382), add:

```python
    ap.add_argument(
        "--provider",
        choices=["auto", "anthropic", "openai"],
        default="auto",
        help="Model provider. 'auto' infers from --model/--base-url (default: auto).",
    )
    ap.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible endpoint (e.g. http://localhost:11434/v1 for Ollama, "
        "https://openrouter.ai/api/v1 for OpenRouter). Forces the openai provider.",
    )
    ap.add_argument("--price-in", type=float, default=None, help="$ per 1M input tokens (cost override)")
    ap.add_argument("--price-out", type=float, default=None, help="$ per 1M output tokens (cost override)")
```

- [ ] **Step 3: Commit**

```bash
git add summarize_video.py
git commit -m "feat: add openai dep and --provider/--base-url/--price flags"
```

---

### Task 2: `resolve_provider()` — pick the provider

**Files:**
- Modify: `summarize_video.py` (add function near `DEFAULT_MODEL`, ~line 46)
- Test: `test_summarize_video.py`

- [ ] **Step 1: Write the failing test**

Create `test_summarize_video.py`:

```python
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
    assert sv.resolve_provider("claude-sonnet-5", "auto", "http://localhost:11434/v1") == "openai"


def test_resolve_provider_env_base_url_forces_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    assert sv.resolve_provider("some-model", "auto", None) == "openai"
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `uv run --with pytest --with anthropic --with openai pytest test_summarize_video.py -q -k resolve_provider`
Expected: FAIL — `AttributeError: module 'summarize_video' has no attribute 'resolve_provider'`

- [ ] **Step 3: Write the minimal implementation**

Add after `DEFAULT_OUT` (~line 42):

```python
def resolve_provider(model: str, provider_arg: str, base_url: str | None) -> str:
    """Pick the summarizer provider. Explicit flag wins; a base URL forces the
    OpenAI-compatible path; otherwise infer from the model name."""
    if provider_arg and provider_arg != "auto":
        return provider_arg
    if base_url or os.environ.get("OPENAI_BASE_URL"):
        return "openai"
    return "anthropic" if model.startswith("claude") else "openai"
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `uv run --with pytest --with anthropic --with openai pytest test_summarize_video.py -q -k resolve_provider`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add summarize_video.py test_summarize_video.py
git commit -m "feat: resolve_provider() to choose anthropic vs openai path"
```

---

### Task 3: `normalize_usage()` — one usage shape

**Files:**
- Modify: `summarize_video.py` (add near `estimate_cost`)
- Test: `test_summarize_video.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
from types import SimpleNamespace


def test_normalize_usage_anthropic():
    u = SimpleNamespace(input_tokens=10, output_tokens=20)
    assert sv.normalize_usage("anthropic", u) == (10, 20)


def test_normalize_usage_openai():
    u = SimpleNamespace(prompt_tokens=5, completion_tokens=7)
    assert sv.normalize_usage("openai", u) == (5, 7)


def test_normalize_usage_none():
    assert sv.normalize_usage("openai", None) is None
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `uv run --with pytest --with anthropic --with openai pytest test_summarize_video.py -q -k normalize_usage`
Expected: FAIL — `AttributeError: ... 'normalize_usage'`

- [ ] **Step 3: Write the minimal implementation**

Add above `estimate_cost` (~line 363):

```python
def normalize_usage(provider: str, usage) -> tuple[int, int] | None:
    """Collapse the two SDKs' usage objects into one (input, output) tuple."""
    if usage is None:
        return None
    if provider == "anthropic":
        return (usage.input_tokens, usage.output_tokens)
    return (usage.prompt_tokens, usage.completion_tokens)
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `uv run --with pytest --with anthropic --with openai pytest test_summarize_video.py -q -k normalize_usage`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add summarize_video.py test_summarize_video.py
git commit -m "feat: normalize_usage() to unify token-count fields"
```

---

### Task 4: Make `estimate_cost()` take a normalized tuple + overrides

**Files:**
- Modify: `summarize_video.py:364-375`
- Test: `test_summarize_video.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
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
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `uv run --with pytest --with anthropic --with openai pytest test_summarize_video.py -q -k estimate_cost`
Expected: FAIL — the old signature takes a `usage` object, so `test_estimate_cost_known_model` errors on `usage.input_tokens`.

- [ ] **Step 3: Write the minimal implementation**

Replace `estimate_cost` (lines 364-375) with:

```python
# input/output $ per 1M tokens for common models. Unknown models -> no estimate.
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4.1-mini": (0.4, 1.6),
}


def estimate_cost(
    model: str,
    usage: tuple[int, int] | None,
    price_override: tuple[float, float] | None,
) -> str | None:
    if usage is None:
        return None
    price = price_override or PRICES.get(model)
    if price is None:
        return None
    tin, tout = usage
    pin, pout = price
    cost = (tin * pin + tout * pout) / 1_000_000
    return f"${cost:.3f} ({tin} in / {tout} out)"
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `uv run --with pytest --with anthropic --with openai pytest test_summarize_video.py -q -k estimate_cost`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add summarize_video.py test_summarize_video.py
git commit -m "feat: estimate_cost() takes normalized usage + price override"
```

---

### Task 5: Extract `build_user_content()` and `summarize_anthropic()`

Refactor of the existing `summarize()` — behavior for Claude must not change.

**Files:**
- Modify: `summarize_video.py:328-361`
- Test: `test_summarize_video.py`

- [ ] **Step 1: Write the failing test**

Append (a fake Anthropic client, no network):

```python
import sys


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
    assert "## TL;DR" in out  # from SUMMARY_INSTRUCTIONS


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
    # system prompt passed at top level, user message carries the content
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
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `uv run --with pytest --with anthropic --with openai pytest test_summarize_video.py -q -k "user_content or summarize_anthropic"`
Expected: FAIL — `AttributeError: ... 'build_user_content'` / `'summarize_anthropic'`

- [ ] **Step 3: Write the minimal implementation**

Replace the whole `summarize()` function (lines 328-361) with:

```python
def build_user_content(meta: dict, transcript: str) -> str:
    header = (
        f"Title: {meta.get('title')}\n"
        f"Channel: {meta.get('uploader') or meta.get('channel')}\n"
        f"Duration: {_fmt_clock(meta.get('duration') or 0)}\n"
        f"URL: {meta.get('webpage_url')}\n"
    )
    return (
        f"{SUMMARY_INSTRUCTIONS}\n"
        f"---\nVideo details:\n{header}\n"
        f"---\nTranscript:\n\n{transcript}"
    )


def summarize_anthropic(model: str, meta: dict, transcript: str) -> tuple[str, object]:
    import anthropic

    client = anthropic.Anthropic()
    user_content = build_user_content(meta, transcript)

    # Stream because a long transcript is a long request; get_final_message()
    # returns the assembled response. Adaptive thinking improves structure.
    with client.messages.stream(
        model=model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=SUMMARY_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    ) as stream:
        response = stream.get_final_message()

    if response.stop_reason == "refusal":
        detail = getattr(response, "stop_details", None)
        cat = getattr(detail, "category", None) if detail else None
        die(f"Claude declined to summarize this video (category: {cat}).")

    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return text, response.usage
```

(The `import anthropic` stays inside the function so a Claude-free run never imports it — and so the test's `sys.modules` fake is picked up.)

- [ ] **Step 4: Run the test, verify it passes**

Run: `uv run --with pytest --with anthropic --with openai pytest test_summarize_video.py -q -k "user_content or summarize_anthropic"`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add summarize_video.py test_summarize_video.py
git commit -m "refactor: extract build_user_content + summarize_anthropic"
```

---

### Task 6: `summarize_openai()` — the OpenAI-compatible adapter

**Files:**
- Modify: `summarize_video.py` (add after `summarize_anthropic`)
- Test: `test_summarize_video.py`

- [ ] **Step 1: Write the failing test**

Append (a fake OpenAI client that yields streamed chunks):

```python
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
    events = [_sse("Hello "), _sse("world"), _sse(finish_reason="stop"), _usage_event(11, 22)]
    recorder = {}
    _install_fake_openai(monkeypatch, events, recorder)
    text, usage = sv.summarize_openai("gpt-4o-mini", {"title": "T", "duration": 0}, "body", None)
    assert text == "Hello world"
    assert usage.prompt_tokens == 11 and usage.completion_tokens == 22
    # system prompt is a message, not a top-level param
    msgs = recorder["messages"]
    assert msgs[0] == {"role": "system", "content": sv.SUMMARY_SYSTEM}
    assert msgs[1]["role"] == "user"


def test_summarize_openai_passes_base_url(monkeypatch):
    events = [_sse("x"), _sse(finish_reason="stop")]
    recorder = {}
    _install_fake_openai(monkeypatch, events, recorder)
    sv.summarize_openai("llama3", {"title": "T", "duration": 0}, "body", "http://localhost:11434/v1")
    assert recorder["base_url"] == "http://localhost:11434/v1"


def test_summarize_openai_content_filter_exits(monkeypatch):
    events = [_sse("partial"), _sse(finish_reason="content_filter")]
    _install_fake_openai(monkeypatch, events, {})
    import pytest
    with pytest.raises(SystemExit):
        sv.summarize_openai("gpt-4o-mini", {"title": "T", "duration": 0}, "body", None)
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `uv run --with pytest --with anthropic --with openai pytest test_summarize_video.py -q -k summarize_openai`
Expected: FAIL — `AttributeError: ... 'summarize_openai'`

- [ ] **Step 3: Write the minimal implementation**

Add after `summarize_anthropic`:

```python
def summarize_openai(
    model: str, meta: dict, transcript: str, base_url: str | None
) -> tuple[str, object]:
    from openai import OpenAI

    resolved_base = base_url or os.environ.get("OPENAI_BASE_URL") or None
    client = OpenAI(base_url=resolved_base)
    user_content = build_user_content(meta, transcript)

    stream = client.chat.completions.create(
        model=model,
        max_tokens=16000,  # see plan: newer OpenAI reasoning models want max_completion_tokens
        stream=True,
        stream_options={"include_usage": True},
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": user_content},
        ],
    )

    parts: list[str] = []
    usage = None
    finish = None
    refused = False
    for event in stream:
        if event.choices:
            choice = event.choices[0]
            delta = getattr(choice, "delta", None)
            if delta is not None:
                if getattr(delta, "content", None):
                    parts.append(delta.content)
                if getattr(delta, "refusal", None):
                    refused = True
            if choice.finish_reason:
                finish = choice.finish_reason
        if getattr(event, "usage", None):
            usage = event.usage

    if refused or finish == "content_filter":
        die(f"The model declined to summarize this video (finish_reason: {finish}).")

    text = "".join(parts).strip()
    return text, usage
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `uv run --with pytest --with anthropic --with openai pytest test_summarize_video.py -q -k summarize_openai`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add summarize_video.py test_summarize_video.py
git commit -m "feat: summarize_openai() adapter for OpenAI-compatible endpoints"
```

---

### Task 7: `summarize()` dispatcher

**Files:**
- Modify: `summarize_video.py` (add after the two adapters)
- Test: `test_summarize_video.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_summarize_dispatches_to_anthropic(monkeypatch):
    called = {}
    monkeypatch.setattr(sv, "summarize_anthropic", lambda m, meta, t: called.setdefault("p", "a") or ("a", None))
    monkeypatch.setattr(sv, "summarize_openai", lambda m, meta, t, b: called.setdefault("p", "o") or ("o", None))
    text, _ = sv.summarize("anthropic", "claude-sonnet-5", {}, "body", None)
    assert called["p"] == "a" and text == "a"


def test_summarize_dispatches_to_openai(monkeypatch):
    called = {}
    monkeypatch.setattr(sv, "summarize_anthropic", lambda m, meta, t: called.setdefault("p", "a") or ("a", None))
    monkeypatch.setattr(sv, "summarize_openai", lambda m, meta, t, b: called.setdefault("p", "o") or ("o", None))
    text, _ = sv.summarize("openai", "gpt-4o-mini", {}, "body", "http://h/v1")
    assert called["p"] == "o" and text == "o"
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `uv run --with pytest --with anthropic --with openai pytest test_summarize_video.py -q -k "summarize_dispatches"`
Expected: FAIL — `TypeError`/`AttributeError` (old `summarize` has a different signature / is gone).

- [ ] **Step 3: Write the minimal implementation**

Add after `summarize_openai`:

```python
def summarize(
    provider: str, model: str, meta: dict, transcript: str, base_url: str | None
) -> tuple[str, object]:
    if provider == "anthropic":
        return summarize_anthropic(model, meta, transcript)
    return summarize_openai(model, meta, transcript, base_url)
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `uv run --with pytest --with anthropic --with openai pytest test_summarize_video.py -q -k "summarize_dispatches"`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add summarize_video.py test_summarize_video.py
git commit -m "feat: summarize() provider dispatcher"
```

---

### Task 8: Wire provider selection, the key gate, and cost into `main()`

Integration task — verified by the full test run in Task 9 and the manual matrix in Task 10 (the branches here touch argv/env, not pure logic).

**Files:**
- Modify: `summarize_video.py:390-472` (the `main()` body from `check_yt_dlp()` down through the summary block) and `run_info` (443)

- [ ] **Step 1: Resolve the provider right after parsing args**

Immediately after `args = ap.parse_args()` (line 388) add:

```python
    provider = resolve_provider(args.model, args.provider, args.base_url)
    price_override = (
        (args.price_in, args.price_out)
        if args.price_in is not None and args.price_out is not None
        else None
    )
```

- [ ] **Step 2: Record the provider in `run_info`**

In the `run_info` dict (around line 443), change:

```python
        "summary_model": None if args.no_summary else args.model,
```

to:

```python
        "summary_model": None if args.no_summary else args.model,
        "summary_provider": None if args.no_summary else provider,
```

- [ ] **Step 3: Replace the summary block with a provider-aware key gate**

Replace the `if not args.no_summary:` block (lines 454-472) with:

```python
    if not args.no_summary:
        if provider == "anthropic":
            key_name = "ANTHROPIC_API_KEY"
        else:
            key_name = "OPENAI_API_KEY"
            # Local runtimes (Ollama/LM Studio/vLLM) ignore the key but the SDK
            # still requires one; supply a harmless placeholder when a base URL
            # is set and no key is present.
            base = args.base_url or os.environ.get("OPENAI_BASE_URL")
            if base and not os.environ.get("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = "not-needed"

        if not os.environ.get(key_name):
            print(
                f"warning: {key_name} not set; skipping summary. "
                "Transcript files were still saved.",
                file=sys.stderr,
            )
        else:
            print(f"Summarizing with {args.model} ({provider})...")
            summary, usage = summarize(provider, args.model, meta, transcript, args.base_url)
            (folder / "summary.md").write_text(
                f"# {title}\n\n"
                f"[Watch]({meta.get('webpage_url')}) · "
                f"{run_info['channel']} · {run_info['duration']}\n\n"
                f"{summary}\n",
                encoding="utf-8",
            )
            cost = estimate_cost(
                args.model, normalize_usage(provider, usage), price_override
            )
            print("Summary saved" + (f" (~{cost})" if cost else ""))
```

- [ ] **Step 4: Sanity-check it imports and the CLI parses**

Run: `uv run summarize_video.py --help`
Expected: usage text showing `--provider`, `--base-url`, `--price-in`, `--price-out`; exit 0.

- [ ] **Step 5: Commit**

```bash
git add summarize_video.py
git commit -m "feat: wire provider selection, key gate, and cost into main()"
```

---

### Task 9: Full test run + lint

Verification task — no new code.

- [ ] **Step 1: Run the whole suite**

Run: `uv run --with pytest --with anthropic --with openai pytest test_summarize_video.py -q`
Expected: PASS (all tasks' tests green, ~20 tests).

- [ ] **Step 2: Lint**

Run: `uvx ruff check summarize_video.py test_summarize_video.py`
Expected: `All checks passed!` (fix any E/F/I/UP/B findings, then re-run).

- [ ] **Step 3: Commit any lint fixes**

```bash
git add -A
git commit -m "chore: lint clean for multi-provider changes"
```

---

### Task 10: README "Using other models" section

Docs task — no test.

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the section** (after the "Make it a command" block, before "## Output")

```markdown
## Using other models

This tool was built for Claude, and Claude stays the default. It also works with
any endpoint that speaks the OpenAI Chat Completions API — hosted or local.

The provider is chosen automatically: a `claude-*` model uses the Anthropic SDK; any
other model, or setting `--base-url`, uses the OpenAI-compatible path. Force it with
`--provider anthropic|openai`.

```bash
# Claude (default) — needs ANTHROPIC_API_KEY
vidsum "https://youtu.be/..."

# OpenAI — needs OPENAI_API_KEY
vidsum "https://youtu.be/..." --model gpt-4o-mini

# OpenRouter — needs OPENAI_API_KEY set to your OpenRouter key
vidsum "https://youtu.be/..." --model anthropic/claude-sonnet-5 \
  --base-url https://openrouter.ai/api/v1

# Local Ollama — no key needed
vidsum "https://youtu.be/..." --model llama3.1 \
  --base-url http://localhost:11434/v1
```

Cost is estimated only for models in the built-in price table; for anything else,
pass `--price-in` and `--price-out` (dollars per 1M tokens) to get an estimate.

Note: adaptive thinking is a Claude-only feature and is skipped on the OpenAI path.
```

Also update the "Requires" block near the top of the README to say the key depends on
the provider: `ANTHROPIC_API_KEY` for Claude, `OPENAI_API_KEY` for OpenAI-compatible
endpoints (unless `--no-summary`).

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document multi-provider model support"
```

---

### Task 11: Manual verification matrix

Verification task — needs real keys/endpoints, so it is a human step, not automated.

- [ ] Claude default unchanged: `uv run summarize_video.py "<short-captioned-video>"` → `summary.md` written, cost line shown, `metadata.json` has `summary_provider: "anthropic"`.
- [ ] OpenAI: `--model gpt-4o-mini` with `OPENAI_API_KEY` set → summary written, `summary_provider: "openai"`.
- [ ] OpenRouter: `--model anthropic/claude-sonnet-5 --base-url https://openrouter.ai/api/v1` with the OpenRouter key in `OPENAI_API_KEY` → summary written.
- [ ] Local Ollama (if available): `--model llama3.1 --base-url http://localhost:11434/v1`, no key → summary written, no cost line (unknown model).
- [ ] `--no-summary` still writes transcript files only and sets `summary_provider: null`.
- [ ] Missing key path: unset the relevant key → warning printed, transcript still saved, exit 0.

---

## Self-review

**Spec coverage:**
- "Option of API or Claude SDK" → Tasks 2, 6, 7 (resolve + openai adapter + dispatch).
- "Built for Claude, keep it default" → Task 2 inference + Task 5 (anthropic path unchanged) + Task 10 README wording.
- Provider-aware key handling → Task 8.
- Cost for non-Claude models → Task 4 (table + override) + Task 10 docs.
- Parity gaps (thinking, refusal, system prompt, usage) → Tasks 5, 6 + notes section.

**Placeholder scan:** none — every step has concrete code/commands.

**Type consistency:** `estimate_cost(model, usage_tuple, price_override)` is defined in Task 4 and called with exactly that shape in Task 8; `normalize_usage(provider, usage)` (Task 3) feeds it; `summarize(provider, model, meta, transcript, base_url)` (Task 7) matches the call site in Task 8; adapters return `(text, usage_obj)` consistently.

**No-silent-failure check:** refusal/content-filter map to `die()` (a non-zero exit), not a blank summary; a missing key warns and skips rather than crashing, matching today's behavior.

---

## Out of scope

- Local Whisper transcription (already flagged separately in the code's "no captions" message).
- Streaming tokens to the terminal, retries/backoff, provider auto-failover.
- A config file — flags + env vars are enough for a single-file tool.
- `max_completion_tokens` handling for OpenAI reasoning models (o1/o3) — documented limitation.
- LiteLLM unified router — the one alternative to the two-adapter approach (see top).
