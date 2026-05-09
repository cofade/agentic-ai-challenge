"""Provider-agnostic LLM client layer.

The pipeline calls LLMs only through the :class:`LLMClient` Protocol defined
in :mod:`wscad_triage.llm.client`. Three concrete backends ship today:

- :class:`OllamaBackend` (default) — self-hosted, runs against a local
  Ollama server. Issue #55. Use this for development and offline testing.
- :class:`AnthropicBackend` — cloud, paid. Use this when an API key is
  available; prompt caching makes it the cheapest cloud path. Issue #17.
- :class:`AzureOpenAIBackend` — production target, stub raising
  :class:`ConfigurationError`. Issue #18.

Tests use :class:`MockLLMClient`. See ADR-004 for the full rationale.
"""

from wscad_triage.llm.anthropic_backend import AnthropicBackend
from wscad_triage.llm.azure_openai_backend import AzureOpenAIBackend
from wscad_triage.llm.client import LLMClient, MockLLMClient
from wscad_triage.llm.errors import ConfigurationError
from wscad_triage.llm.factory import make_client
from wscad_triage.llm.ollama_backend import OllamaBackend
from wscad_triage.llm.types import LLMResponse, Message, ToolCall, ToolSpec, Usage

__all__ = [
    "AnthropicBackend",
    "AzureOpenAIBackend",
    "ConfigurationError",
    "LLMClient",
    "LLMResponse",
    "Message",
    "MockLLMClient",
    "OllamaBackend",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "make_client",
]
