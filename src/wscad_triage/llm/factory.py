"""Factory: build an ``LLMClient`` from a typed ``Settings`` object."""

from __future__ import annotations

from wscad_triage.llm.anthropic_backend import AnthropicBackend
from wscad_triage.llm.azure_openai_backend import AzureOpenAIBackend
from wscad_triage.llm.client import LLMClient
from wscad_triage.llm.errors import ConfigurationError
from wscad_triage.settings import Settings


def make_client(settings: Settings) -> LLMClient:
    """Return the backend selected by ``settings.llm_provider``.

    Raises :class:`ConfigurationError` if the chosen provider's required
    credentials are missing — fail-fast at startup beats failing mid-pipeline.
    """
    if settings.llm_provider == "anthropic":
        if settings.anthropic_api_key is None:
            raise ConfigurationError(
                "WSCAD_TRIAGE_PROVIDER=anthropic requires ANTHROPIC_API_KEY. "
                "Add it to .env or your shell environment."
            )
        return AnthropicBackend(
            api_key=settings.anthropic_api_key.get_secret_value(),
            model=settings.anthropic_model,
        )

    if settings.llm_provider == "azure":
        return AzureOpenAIBackend(
            endpoint=settings.azure_openai_endpoint,
            api_key=(
                settings.azure_openai_api_key.get_secret_value()
                if settings.azure_openai_api_key
                else None
            ),
            deployment=settings.azure_openai_deployment,
            api_version=settings.azure_openai_api_version,
        )

    raise ConfigurationError(  # pragma: no cover — Literal exhausts
        f"Unknown WSCAD_TRIAGE_PROVIDER={settings.llm_provider!r}"
    )
