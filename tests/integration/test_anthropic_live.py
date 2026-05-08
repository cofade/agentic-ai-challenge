"""Live Anthropic API smoke test (issue #17 acceptance gate).

Two layers of gating ensure no accidental billing:

1. ``pyproject.toml``'s default ``addopts`` adds ``-m "not live_api"`` so
   ``pytest tests/`` (or any unfiltered run) excludes this collection.
2. The test still calls ``pytest.skip()`` when ``ANTHROPIC_API_KEY`` is
   absent, so an explicit ``pytest -m live_api`` without a key is also
   harmless.

Run locally with: ``ANTHROPIC_API_KEY=... uv run pytest -m live_api``.

Cost is one short ``messages.create`` call; the assertion is intentionally
minimal because upstream prose can drift over time — we just verify the
round trip works end-to-end through the real SDK with a real key.
"""

from __future__ import annotations

import os

import pytest

from wscad_triage.llm import AnthropicBackend, Message


@pytest.mark.live_api
def test_anthropic_backend_real_roundtrip() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set; skipping live API smoke test")

    model = os.environ.get("WSCAD_TRIAGE_ANTHROPIC_MODEL", "claude-sonnet-4-6")
    backend = AnthropicBackend(api_key=api_key, model=model, max_tokens=16)

    response = backend.generate(
        [
            Message(role="system", content="Reply with the single word: ok"),
            Message(role="user", content="ping"),
        ],
    )
    assert response.content.strip(), "live API returned empty content"
    assert response.stop_reason in {"end_turn", "max_tokens"}
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
