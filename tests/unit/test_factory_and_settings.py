"""Tests for ``Settings`` parsing and ``make_client`` provider dispatch.

The ``_scrub_settings_env`` autouse fixture in ``tests/conftest.py`` clears
every Settings-relevant env var before each test, so the suite's outcome
does not depend on a developer's shell having (or not having) an Anthropic
API key exported.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from wscad_triage.llm import (
    AnthropicBackend,
    AzureOpenAIBackend,
    ConfigurationError,
    OllamaBackend,
    make_client,
)
from wscad_triage.settings import Settings


def _settings(**overrides: object) -> Settings:
    """Build Settings with explicit Anthropic overrides for tests that exercise
    the Anthropic dispatch path. Env is already scrubbed by conftest, so the
    fields not overridden here fall back to module defaults.
    """
    base: dict[str, object] = {
        "WSCAD_TRIAGE_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": "sk-test",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[call-arg]


def test_settings_overridden_to_anthropic() -> None:
    """Explicit override puts us on the Anthropic path."""
    s = _settings()
    assert s.llm_provider == "anthropic"
    assert s.anthropic_model == "claude-sonnet-4-6"
    assert isinstance(s.anthropic_api_key, SecretStr)


def test_settings_default_provider_is_ollama() -> None:
    """No-override default is Ollama (issue #55) — runnable offline.

    Pinned because flipping the default back to a paid provider would
    silently regress contributor onboarding (clone-and-run breaks).
    """
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.llm_provider == "ollama"
    assert s.anthropic_api_key is None


def test_settings_ollama_defaults() -> None:
    """gpt-oss:20b is the project's tested model; localhost:11434 is the
    Ollama daemon's default port. Both are pinned to catch a silent change
    that would put a different model in front of the workers without a
    fidelity benchmark.
    """
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.ollama_model == "gpt-oss:20b"
    assert s.ollama_timeout_seconds == 120.0


def test_settings_ollama_overrides_via_env() -> None:
    s = Settings(  # type: ignore[call-arg]
        _env_file=None,
        WSCAD_TRIAGE_OLLAMA_BASE_URL="http://gpu-host:11434",
        WSCAD_TRIAGE_OLLAMA_MODEL="qwen2.5:14b",
        WSCAD_TRIAGE_OLLAMA_TIMEOUT=300.0,
    )
    assert s.ollama_base_url == "http://gpu-host:11434"
    assert s.ollama_model == "qwen2.5:14b"
    assert s.ollama_timeout_seconds == 300.0


def test_settings_reads_aliased_env_names() -> None:
    s = _settings(WSCAD_TRIAGE_PROVIDER="azure", AZURE_OPENAI_DEPLOYMENT="dep1")
    assert s.llm_provider == "azure"
    assert s.azure_openai_deployment == "dep1"


def test_make_client_returns_ollama_backend_by_default() -> None:
    """Default Settings select Ollama; the factory must dispatch accordingly."""
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    client = make_client(s)
    assert isinstance(client, OllamaBackend)
    # Constructor parameters from Settings reach the backend.
    assert client._model == "gpt-oss:20b"
    assert client._base_url == "http://localhost:11434"


def test_make_client_ollama_without_credentials_does_not_raise() -> None:
    """Ollama requires no env-var credentials; the factory must not fail-fast.

    The runtime contract is the local server's reachability, validated lazily
    on the first ``generate()`` call. Mirrors AnthropicBackend's permissive
    constructor; contrasts with the Azure stub.
    """
    s = Settings(_env_file=None, WSCAD_TRIAGE_PROVIDER="ollama")  # type: ignore[call-arg]
    client = make_client(s)
    assert isinstance(client, OllamaBackend)


def test_make_client_returns_anthropic_backend_with_key() -> None:
    client = make_client(_settings())
    assert isinstance(client, AnthropicBackend)


def test_make_client_anthropic_without_key_raises() -> None:
    s = Settings(_env_file=None, WSCAD_TRIAGE_PROVIDER="anthropic")  # type: ignore[call-arg]
    assert s.anthropic_api_key is None  # belt-and-braces against env leaks
    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        make_client(s)


def test_make_client_azure_raises_via_stub() -> None:
    s = _settings(WSCAD_TRIAGE_PROVIDER="azure")
    with pytest.raises(ConfigurationError, match="not yet implemented"):
        make_client(s)


def test_azure_backend_constructor_raises_directly() -> None:
    """Match on a stable substring so future changes to the redirect message
    don't double-pin the wording. The contract is "raises with a usable
    redirect to a working provider"; the exact provider names + ordering
    are documentation, not API.
    """
    with pytest.raises(ConfigurationError, match=r"WSCAD_TRIAGE_PROVIDER="):
        AzureOpenAIBackend(endpoint="x", api_key="y", deployment="z")


def test_azure_backend_accepts_api_version_kwarg() -> None:
    """Constructor signature includes api_version so factory wiring is symmetric."""
    with pytest.raises(ConfigurationError):
        AzureOpenAIBackend(api_version="2024-10-21")
