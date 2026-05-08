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
    make_client,
)
from wscad_triage.settings import Settings


def _settings(**overrides: object) -> Settings:
    """Build Settings with explicit overrides; env is already scrubbed by conftest."""
    base: dict[str, object] = {
        "WSCAD_TRIAGE_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": "sk-test",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[call-arg]


def test_settings_defaults_when_provider_unset() -> None:
    s = _settings()
    assert s.llm_provider == "anthropic"
    assert s.anthropic_model == "claude-sonnet-4-6"
    assert isinstance(s.anthropic_api_key, SecretStr)


def test_settings_defaults_with_no_overrides() -> None:
    """With env scrubbed and no overrides, defaults apply and key is None."""
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.llm_provider == "anthropic"
    assert s.anthropic_api_key is None


def test_settings_reads_aliased_env_names() -> None:
    s = _settings(WSCAD_TRIAGE_PROVIDER="azure", AZURE_OPENAI_DEPLOYMENT="dep1")
    assert s.llm_provider == "azure"
    assert s.azure_openai_deployment == "dep1"


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
    with pytest.raises(ConfigurationError, match="WSCAD_TRIAGE_PROVIDER=anthropic"):
        AzureOpenAIBackend(endpoint="x", api_key="y", deployment="z")


def test_azure_backend_accepts_api_version_kwarg() -> None:
    """Constructor signature includes api_version so factory wiring is symmetric."""
    with pytest.raises(ConfigurationError):
        AzureOpenAIBackend(api_version="2024-10-21")
