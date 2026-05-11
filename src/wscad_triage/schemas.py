"""Typed contracts for the WSCAD ticket-triage pipeline.

Every value that crosses a component boundary in this codebase — ticket input,
KB chunk, retrieval result, agent decision, reasoning step, final output, and
the LangGraph state object — is a Pydantic model defined in this module.

This is the realisation of the **"Pydantic at every boundary"** Architecture
Principle in `CLAUDE.md`. The design conventions (closed `Literal` enums,
``extra="forbid"``, the cross-field validator on :class:`Output`) are
documented in ADR-002 — Schema Conventions.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Category = Literal["Licensing", "Installation", "Errors", "Performance", "Other"]
Priority = Literal["Low", "Medium", "High", "Critical"]
ResolutionKind = Literal["solve", "clarify"]
RetrieverName = Literal["bm25", "embedding", "rrf"]
CoverageCase = Literal[
    "resolvable EN",
    "resolvable DE",
    "clarify missing OS",
    "clarify vague crash",
    "licensing vs installation",
    "ungrounded claim trap",
    "multilingual mixed",
    "release notes grounded",
]


class TicketMetadata(BaseModel):
    """Metadata block on a support ticket.

    All three named fields are optional because real tickets routinely omit
    them. ``extra="allow"`` keeps unknown source-system fields intact rather
    than rejecting tickets that carry vendor-specific keys; the triage agent
    is responsible for deciding whether a missing field is critical.
    """

    model_config = ConfigDict(extra="allow")

    product: str | None = None
    version: str | None = None
    os: str | None = None


class Ticket(BaseModel):
    """A single input ticket as parsed from ``tickets.json``."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    text: str
    metadata: TicketMetadata = Field(default_factory=TicketMetadata)


class KBChunk(BaseModel):
    """One chunk of a knowledge-base document — the unit retrieval works on."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    source_file: str
    text: str
    language: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """One hit returned by a retriever, with provenance."""

    model_config = ConfigDict(extra="forbid")

    chunk: KBChunk
    score: float
    rank: int
    retriever: RetrieverName


class Classification(BaseModel):
    """The triage agent's decision for a ticket."""

    model_config = ConfigDict(extra="forbid")

    category: Category
    priority: Priority
    missing_critical_fields: list[str] = Field(default_factory=list)
    rationale: str


class ReasoningStep(BaseModel):
    """One entry in the per-ticket reasoning trace.

    Every agent step appends one of these to :attr:`TicketState.reasoning_trace`
    so the final output can render an explainable trail back to the user.
    """

    model_config = ConfigDict(extra="forbid")

    actor: str
    action: str
    evidence_refs: list[str] = Field(default_factory=list)
    rationale: str


class ClaimEvidence(BaseModel):
    """One claim in a proposed solution mapped to a retrieved chunk.

    Produced by the reason agent (issue #21); consumed by the verifier
    agent (issue #23). The ``chunk_id`` MUST be present in the parent
    ``TicketState.retrievals`` — the reason agent enforces this; a
    fabricated id is the trivial hallucination case.

    ``quote`` is the exact substring of the chunk text that supports the
    claim. Strict equality (``quote in chunk.text``) is the cheap
    defence-in-depth alongside the verifier's LLM-as-judge verdict
    (ADR-008). ``min_length=1`` defends against the empty-string
    no-op — every non-empty string is a substring of every chunk.
    """

    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class DraftSolution(BaseModel):
    """The reason agent's structured output before verifier scrutiny."""

    model_config = ConfigDict(extra="forbid")

    solution: str
    claims: list[ClaimEvidence] = Field(default_factory=list)


class ClaimVerdict(BaseModel):
    """The verifier's per-claim judgment."""

    model_config = ConfigDict(extra="forbid")

    claim: str
    grounded: bool
    rationale: str


class VerifierVerdict(BaseModel):
    """Aggregate output of the verify agent (issue #23, ADR-008)."""

    model_config = ConfigDict(extra="forbid")

    grounding_score: float = Field(ge=0.0, le=1.0)
    per_claim: list[ClaimVerdict] = Field(default_factory=list)


class Output(BaseModel):
    """The final per-ticket result emitted by the pipeline.

    Exactly one of :attr:`proposed_solution` and :attr:`preliminary_assessment`
    is populated, determined by :attr:`resolution_kind`. The cross-field
    validator below enforces that invariant — a malformed Output cannot leave
    the pipeline silently.
    """

    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    category: Category
    priority: Priority
    resolution_kind: ResolutionKind
    proposed_solution: str | None = None
    preliminary_assessment: str | None = None
    followup_questions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_breakdown: dict[str, float] = Field(default_factory=dict)
    reasoning_trace: list[ReasoningStep] = Field(default_factory=list)
    cited_sources: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_resolution_body(self) -> Output:
        if self.resolution_kind == "solve":
            if self.proposed_solution is None:
                raise ValueError("resolution_kind='solve' requires proposed_solution")
            if self.preliminary_assessment is not None:
                raise ValueError("resolution_kind='solve' must not set preliminary_assessment")
        else:
            if self.preliminary_assessment is None:
                raise ValueError("resolution_kind='clarify' requires preliminary_assessment")
            if self.proposed_solution is not None:
                raise ValueError("resolution_kind='clarify' must not set proposed_solution")
        return self


class EvalTicket(Ticket):
    """A labelled ticket used by the Phase 5 evaluation harness.

    Extends :class:`Ticket` with ground-truth fields required by the runner
    (issue #33). ``coverage_case`` identifies which of the eight test cases
    this ticket exercises (seven from the Phase 5 roadmap plus
    ``"release notes grounded"`` added by issue #64) so the runner can
    compute per-case metrics.
    """

    model_config = ConfigDict(extra="forbid")

    expected_category: Category
    expected_priority: Priority
    should_clarify: bool
    coverage_case: CoverageCase
    notes: str = ""


class TicketState(BaseModel):
    """Mutable state object the LangGraph supervisor + workers read and write.

    The supervisor reads the current state and decides which worker to call
    next; each worker returns a state update. Optional fields are populated
    progressively as the pipeline runs.
    """

    model_config = ConfigDict(extra="forbid")

    ticket: Ticket
    classification: Classification | None = None
    retrievals: list[RetrievalResult] = Field(default_factory=list)
    draft_solution: DraftSolution | None = None
    verifier_verdict: VerifierVerdict | None = None
    proposed_solution: str | None = None
    preliminary_assessment: str | None = None
    followup_questions: list[str] = Field(default_factory=list)
    reasoning_trace: list[ReasoningStep] = Field(default_factory=list)
    confidence_components: dict[str, float] = Field(default_factory=dict)
    final_confidence: float | None = None
    resolution_kind: ResolutionKind | None = None


__all__ = [
    "Category",
    "ClaimEvidence",
    "ClaimVerdict",
    "Classification",
    "CoverageCase",
    "DraftSolution",
    "EvalTicket",
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
]
