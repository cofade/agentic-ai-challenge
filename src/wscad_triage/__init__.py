"""WSCAD ticket-triage agent package."""

from wscad_triage import pipeline
from wscad_triage.kb import load_kb
from wscad_triage.schemas import (
    Category,
    ClaimEvidence,
    ClaimVerdict,
    Classification,
    DraftSolution,
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
    VerifierVerdict,
)

__version__ = "0.1.0"

__all__ = [
    "Category",
    "ClaimEvidence",
    "ClaimVerdict",
    "Classification",
    "DraftSolution",
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
    "VerifierVerdict",
    "__version__",
    "load_kb",
    "pipeline",
]
