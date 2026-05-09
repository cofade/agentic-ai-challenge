"""Azure OpenAI backend — stub.

The challenge brief names Azure OpenAI as the production target. We lock the
provider-selection surface in Phase 3 (issue #18) without committing to a
specific Azure SDK version: this class implements the :class:`LLMClient`
Protocol so the factory selects it cleanly, but its constructor always
raises :class:`ConfigurationError` redirecting the operator to one of the
two working backends (Ollama, Anthropic). Full implementation is tracked
as future work in ADR-004.

Constructor kwargs mirror the four ``AZURE_OPENAI_*`` settings carried by
:class:`wscad_triage.settings.Settings`. They are unused today but
preserved so the future implementation can drop in without changing the
factory's call site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wscad_triage.llm.errors import ConfigurationError
from wscad_triage.llm.types import LLMResponse, Message, ToolSpec

if TYPE_CHECKING:
    from pydantic import BaseModel


class AzureOpenAIBackend:
    """Azure OpenAI provider — not yet implemented.

    Construction always raises so the failure surfaces at startup, not on
    the first ``generate`` call halfway through processing a ticket.
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,  # noqa: ARG002 — preserved for the future impl
        api_key: str | None = None,  # noqa: ARG002 — preserved for the future impl
        deployment: str | None = None,  # noqa: ARG002 — preserved for the future impl
        api_version: str | None = None,  # noqa: ARG002 — preserved for the future impl
    ) -> None:
        raise ConfigurationError(
            "Azure OpenAI backend is not yet implemented. "
            "Set WSCAD_TRIAGE_PROVIDER=ollama (default; runs against a local "
            "Ollama server) or WSCAD_TRIAGE_PROVIDER=anthropic (with "
            "ANTHROPIC_API_KEY). "
            "See docs/09-architecture-decisions/ADR-004-llm-provider-abstraction.md."
        )

    def generate(  # pragma: no cover — unreachable; constructor raises.
        self,
        messages: list[Message],  # noqa: ARG002
        *,
        tools: list[ToolSpec] | None = None,  # noqa: ARG002
        response_format: type[BaseModel] | None = None,  # noqa: ARG002
    ) -> LLMResponse:
        raise ConfigurationError("AzureOpenAIBackend is not implemented.")
