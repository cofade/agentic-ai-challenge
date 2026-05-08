"""Shared pytest setup for unit tests.

Env-isolation autouse fixture: every unit test runs with our project's
settings env vars unset so the suite's outcome cannot depend on a
developer's shell configuration (e.g., a real ``ANTHROPIC_API_KEY``
exported in their session silently turning a "no creds" assertion green).

Tests that need a specific value should set it explicitly via
``monkeypatch.setenv`` or pass it as a kwarg to ``Settings``.

Scoped to ``tests/unit/`` deliberately. Integration tests under
``tests/integration/`` (notably the ``live_api``-marked Anthropic smoke)
need ``ANTHROPIC_API_KEY`` to be passed through from the developer's
environment; scrubbing them there would make the acceptance gate for
issue #17 unreachable.
"""

from __future__ import annotations

import pytest

_SCRUBBED_ENV_VARS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "WSCAD_TRIAGE_PROVIDER",
    "WSCAD_TRIAGE_ANTHROPIC_MODEL",
    "WSCAD_TRIAGE_OLLAMA_BASE_URL",
    "WSCAD_TRIAGE_OLLAMA_MODEL",
    "WSCAD_TRIAGE_OLLAMA_TIMEOUT",
    "OLLAMA_BASE_URL",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_API_VERSION",
)


@pytest.fixture(autouse=True)
def _scrub_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every Settings-relevant env var for the duration of one test."""
    for name in _SCRUBBED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
