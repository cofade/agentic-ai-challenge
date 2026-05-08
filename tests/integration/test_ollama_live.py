"""Live Ollama smoke test (issue #55 acceptance gate).

Two-stage gate so the test is harmless on a developer's box even when
they happen to have Ollama installed but in a partially-configured state:

1. ``pyproject.toml``'s default ``addopts`` adds ``-m "not live_api"`` so
   ``pytest tests/`` excludes this test from regular runs.
2. The test calls ``pytest.skip()`` when:
   - the Ollama server is not reachable at ``OLLAMA_BASE_URL``, OR
   - the configured model is not pulled (``ollama pull <model>`` not run).

Run locally with::

    ollama pull gpt-oss:20b
    ollama serve  # if not already running
    uv run pytest -m live_api tests/integration/test_ollama_live.py

The assertion is intentionally minimal — upstream prose drifts and
``gpt-oss:20b`` is a reasoning model whose output shape evolves with
Ollama versions. We verify the round trip works end-to-end through the
real SDK against a real local server, including a tool-use roundtrip
(the workers depend on it).
"""

from __future__ import annotations

import os

import httpx
import pytest

from wscad_triage.llm import Message, OllamaBackend, ToolSpec

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL = "gpt-oss:20b"


def _resolve_base_url() -> str:
    return (
        os.environ.get("OLLAMA_BASE_URL")
        or os.environ.get("WSCAD_TRIAGE_OLLAMA_BASE_URL")
        or _DEFAULT_BASE_URL
    )


def _resolve_model() -> str:
    return os.environ.get("WSCAD_TRIAGE_OLLAMA_MODEL", _DEFAULT_MODEL)


def _skip_unless_server_and_model_available(base_url: str, model: str) -> None:
    """Two-stage gate: server reachable, then model present in /api/tags."""
    try:
        version_resp = httpx.get(f"{base_url}/api/version", timeout=2.0)
        version_resp.raise_for_status()
    except (httpx.HTTPError, httpx.ConnectError) as exc:
        pytest.skip(f"Ollama server not reachable at {base_url}: {exc}")

    try:
        tags = httpx.get(f"{base_url}/api/tags", timeout=5.0).json()
    except (httpx.HTTPError, ValueError) as exc:
        pytest.skip(f"Ollama /api/tags unavailable at {base_url}: {exc}")

    pulled = {t.get("name", "") for t in tags.get("models", [])}
    if model not in pulled:
        pytest.skip(
            f"Ollama model {model!r} not pulled (have: {sorted(pulled)}); "
            f"run `ollama pull {model}` to enable this test."
        )


@pytest.mark.live_api
def test_ollama_backend_real_roundtrip() -> None:
    """End-to-end: real Ollama server, real model, real generate()."""
    base_url = _resolve_base_url()
    model = _resolve_model()
    _skip_unless_server_and_model_available(base_url, model)

    backend = OllamaBackend(base_url=base_url, model=model, timeout=120.0)
    response = backend.generate(
        [
            Message(role="system", content="Reply with the single word: ok"),
            Message(role="user", content="ping"),
        ],
    )
    assert response.content.strip(), "live Ollama returned empty content"
    assert response.stop_reason in {"end_turn", "max_tokens"}
    # Ollama always returns prompt_eval_count + eval_count for non-cached calls.
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0


@pytest.mark.live_api
def test_ollama_backend_real_tool_use() -> None:
    """Tool-use roundtrip — the workers depend on this. Skips with the same
    two-stage gate as the text-roundtrip test.
    """
    base_url = _resolve_base_url()
    model = _resolve_model()
    _skip_unless_server_and_model_available(base_url, model)

    backend = OllamaBackend(base_url=base_url, model=model, timeout=120.0)
    spec = ToolSpec(
        name="record_classification",
        description="Record a single ticket classification.",
        input_schema={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["Licensing", "Installation", "Errors", "Performance", "Other"],
                },
                "priority": {
                    "type": "string",
                    "enum": ["Low", "Medium", "High", "Critical"],
                },
            },
            "required": ["category", "priority"],
        },
    )
    response = backend.generate(
        [
            Message(
                role="system",
                content=(
                    "You are a triage classifier. Call record_classification "
                    "exactly once with category=Licensing, priority=High."
                ),
            ),
            Message(role="user", content="Activation Error 504. License is dead."),
        ],
        tools=[spec],
    )
    # The model may emit text alongside the tool call; we only assert the
    # tool was called with the right shape (matches what the workers expect).
    assert len(response.tool_calls) >= 1, (
        f"expected at least one tool call, got: stop_reason={response.stop_reason!r} "
        f"content={response.content!r}"
    )
    tc = response.tool_calls[0]
    assert tc.name == "record_classification"
    assert tc.id  # synthesised UUID, non-empty
    assert "category" in tc.arguments
    assert "priority" in tc.arguments
