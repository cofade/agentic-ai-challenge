"""Typed configuration for the LLM layer (Phase 3 — minimal scope).

Phase 4 (#27) layers the rubric weights and threshold knobs on top via
``config.yaml``; Phase 3 only needs provider selection so the factory in
:mod:`wscad_triage.llm.factory` can build the right backend.

Values are read from environment variables and an optional ``.env`` file.
The env-var names match the existing ``.env.example`` template authored in
Phase 0:

- ``WSCAD_TRIAGE_PROVIDER`` — provider selection (``anthropic`` | ``azure``).
- ``ANTHROPIC_API_KEY`` — Anthropic credential (the SDK's convention name).
- ``WSCAD_TRIAGE_ANTHROPIC_MODEL`` — model id; defaults to a current Sonnet.
- ``AZURE_OPENAI_API_KEY`` / ``AZURE_OPENAI_ENDPOINT`` /
  ``AZURE_OPENAI_DEPLOYMENT`` / ``AZURE_OPENAI_API_VERSION`` — Azure
  credentials (consumed only when the Azure backend lands; today the stub
  raises before reading them).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["anthropic", "azure"]


class Settings(BaseSettings):
    """Runtime settings sourced from env vars / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    llm_provider: LLMProvider = Field(default="anthropic", alias="WSCAD_TRIAGE_PROVIDER")
    anthropic_model: str = Field(default="claude-sonnet-4-6", alias="WSCAD_TRIAGE_ANTHROPIC_MODEL")
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    azure_openai_endpoint: str | None = Field(default=None, alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: SecretStr | None = Field(default=None, alias="AZURE_OPENAI_API_KEY")
    azure_openai_deployment: str | None = Field(default=None, alias="AZURE_OPENAI_DEPLOYMENT")
    azure_openai_api_version: str = Field(default="2024-10-21", alias="AZURE_OPENAI_API_VERSION")
