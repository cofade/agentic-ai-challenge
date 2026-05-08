"""Provider-agnostic LLM client layer (Phase 3 — issue #17).

The pipeline calls LLMs only through the :class:`LLMClient` Protocol defined
in :mod:`wscad_triage.llm.client`. Two concrete backends ship today:

- :class:`AnthropicBackend` — production default, with prompt-caching support.
- :class:`AzureOpenAIBackend` — stub raising :class:`ConfigurationError`;
  documented production target (see ADR-004).

Tests use :class:`MockLLMClient`. See ADR-004 for the full rationale.
"""

from wscad_triage.llm.anthropic_backend import AnthropicBackend
from wscad_triage.llm.azure_openai_backend import AzureOpenAIBackend
from wscad_triage.llm.client import LLMClient, MockLLMClient
from wscad_triage.llm.errors import ConfigurationError
from wscad_triage.llm.factory import make_client
from wscad_triage.llm.types import LLMResponse, Message, ToolCall, ToolSpec, Usage

__all__ = [
    "AnthropicBackend",
    "AzureOpenAIBackend",
    "ConfigurationError",
    "LLMClient",
    "LLMResponse",
    "Message",
    "MockLLMClient",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "make_client",
]
