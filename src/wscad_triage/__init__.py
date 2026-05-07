"""WSCAD ticket-triage agent package."""

from wscad_triage.schemas import (
    Category,
    Classification,
    KBChunk,
    Output,
    Priority,
    ReasoningStep,
    ResolutionKind,
    RetrievalResult,
    RetrieverName,
    Ticket,
    TicketMetadata,
    TicketState,
)

__version__ = "0.1.0"

__all__ = [
    "Category",
    "Classification",
    "KBChunk",
    "Output",
    "Priority",
    "ReasoningStep",
    "ResolutionKind",
    "RetrievalResult",
    "RetrieverName",
    "Ticket",
    "TicketMetadata",
    "TicketState",
    "__version__",
]
