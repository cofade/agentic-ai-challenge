"""Typed configuration for the LLM layer (Phase 3 — minimal scope).

Phase 4 (#27) layers the rubric weights and threshold knobs on top via
``config.yaml``; Phase 3 only needs provider selection so the factory in
:mod:`wscad_triage.llm.factory` can build the right backend.

Values are read from environment variables and an optional ``.env`` file.
The env-var names match the existing ``.env.example`` template:

- ``WSCAD_TRIAGE_PROVIDER`` — provider selection
  (``ollama`` | ``anthropic`` | ``azure``); default ``ollama`` so the
  pipeline is runnable offline against a local server (issue #55).
- ``WSCAD_TRIAGE_OLLAMA_BASE_URL`` / ``WSCAD_TRIAGE_OLLAMA_MODEL`` /
  ``WSCAD_TRIAGE_OLLAMA_TIMEOUT`` — Ollama server config; defaults
  point at ``localhost:11434`` and the project's tested model
  (``gpt-oss:20b``).
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

LLMProvider = Literal["ollama", "anthropic", "azure"]


class Settings(BaseSettings):
    """Runtime settings sourced from env vars / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    llm_provider: LLMProvider = Field(default="ollama", alias="WSCAD_TRIAGE_PROVIDER")

    ollama_base_url: str = Field(
        default="http://localhost:11434", alias="WSCAD_TRIAGE_OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(default="gpt-oss:20b", alias="WSCAD_TRIAGE_OLLAMA_MODEL")
    ollama_timeout_seconds: float = Field(default=120.0, alias="WSCAD_TRIAGE_OLLAMA_TIMEOUT")

    anthropic_model: str = Field(default="claude-sonnet-4-6", alias="WSCAD_TRIAGE_ANTHROPIC_MODEL")
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    azure_openai_endpoint: str | None = Field(default=None, alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: SecretStr | None = Field(default=None, alias="AZURE_OPENAI_API_KEY")
    azure_openai_deployment: str | None = Field(default=None, alias="AZURE_OPENAI_DEPLOYMENT")
    azure_openai_api_version: str = Field(default="2024-10-21", alias="AZURE_OPENAI_API_VERSION")
