"""Phase 5 evaluation runner (issue #33, ADR-010).

Loads the hand-labeled eval set, runs the triage pipeline ticket-by-ticket
against the configured LLM provider, scores each output with
:mod:`eval.metrics`, and writes a single canonical artefact at
``eval/results/latest.json`` plus a timestamped sibling for local
history (gitignored).

Entry points:

- ``run_eval`` — pure-Python kernel; takes pre-built ``EvalTicket``,
  ``LLMClient``, and ``HybridRetriever`` objects plus a ``RunMetadata``
  envelope. Tests inject a fake ``pipeline_run`` so they can exercise the
  loop without scripting the full LangGraph pipeline.
- ``main(argv)`` — the ``python -m eval.runner`` argparse layer; mirrors
  ``wscad_triage.cli`` for argument shape and Settings construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from eval.metrics import AggregateMetrics, PerTicketResult, compute_metrics, score_error, score_one
from wscad_triage import pipeline
from wscad_triage.kb import BM25, Embedding, HybridRetriever, load_kb
from wscad_triage.llm.client import LLMClient
from wscad_triage.llm.errors import ConfigurationError
from wscad_triage.llm.factory import make_client
from wscad_triage.schemas import EvalTicket, Output, Ticket
from wscad_triage.settings import LLMProvider, Settings

if TYPE_CHECKING:
    from collections.abc import Sequence

PipelineRunFn = Callable[[Ticket, LLMClient, HybridRetriever], Output]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_SET = REPO_ROOT / "tickets" / "eval_set.json"
DEFAULT_KB_DIR = REPO_ROOT / "kb"
DEFAULT_RESULTS_DIR = REPO_ROOT / "eval" / "results"


class RunMetadata(BaseModel):
    """Reproducibility envelope embedded in every eval results doc.

    "What did we run, against what, on which commit?" — the minimum a
    reviewer needs to take a baseline number at face value or to
    reproduce it.
    """

    model_config = ConfigDict(extra="forbid")

    timestamp_utc: str
    provider: str
    model: str
    kb_chunk_count: int
    eval_set_path: str
    eval_set_sha256: str
    git_commit: str
    wscad_triage_version: str


class EvalResults(BaseModel):
    """The top-level shape of ``eval/results/latest.json``."""

    model_config = ConfigDict(extra="forbid")

    metadata: RunMetadata
    aggregate: AggregateMetrics
    per_ticket: list[PerTicketResult]


def _resolve_model_name(settings: Settings) -> str:
    """Provider-specific model identifier for the metadata envelope."""
    if settings.llm_provider == "ollama":
        return settings.ollama_model
    if settings.llm_provider == "anthropic":
        return settings.anthropic_model
    if settings.llm_provider == "azure":
        return settings.azure_openai_deployment or "azure-openai-deployment-unset"
    return "unknown"


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _wscad_version() -> str:
    try:
        return importlib_metadata.version("wscad-triage")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_metadata(
    settings: Settings,
    eval_set_path: Path,
    kb_chunk_count: int,
) -> RunMetadata:
    """Gather provenance fields for a run. Pure of side effects beyond
    reading the eval-set file and shelling out to ``git rev-parse``.
    """
    return RunMetadata(
        timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        provider=settings.llm_provider,
        model=_resolve_model_name(settings),
        kb_chunk_count=kb_chunk_count,
        eval_set_path=str(eval_set_path),
        eval_set_sha256=_hash_file(eval_set_path),
        git_commit=_git_commit(),
        wscad_triage_version=_wscad_version(),
    )


def run_eval(
    eval_tickets: list[EvalTicket],
    llm: LLMClient,
    retriever: HybridRetriever,
    metadata: RunMetadata,
    results_dir: Path,
    *,
    pipeline_run: PipelineRunFn = pipeline.run,
    progress: Callable[[str], None] | None = print,
) -> EvalResults:
    """Score each eval ticket and write ``latest.json`` + a timestamped
    sibling.

    Per-ticket exceptions are caught and recorded as ``error`` rows so
    one bad ticket cannot abort the whole run. Construction errors
    (missing eval set, missing KB, missing credentials) belong upstream
    and should not reach this function.
    """
    per_ticket: list[PerTicketResult] = []
    for et in eval_tickets:
        try:
            output = pipeline_run(et, llm, retriever)
        except Exception as exc:
            per_ticket.append(score_error(et, exc))
            if progress:
                progress(f"[eval] {et.ticket_id} ({et.coverage_case}) -> ERROR: {exc!r}")
            continue
        per_ticket.append(score_one(et, output))
        if progress:
            progress(
                f"[eval] {et.ticket_id} ({et.coverage_case}) -> "
                f"{output.resolution_kind} cat={output.category} prio={output.priority} "
                f"conf={output.confidence:.2f}"
            )

    aggregate = compute_metrics(per_ticket)
    results = EvalResults(metadata=metadata, aggregate=aggregate, per_ticket=per_ticket)

    results_dir.mkdir(parents=True, exist_ok=True)
    payload = results.model_dump_json(indent=2) + "\n"
    (results_dir / "latest.json").write_text(payload, encoding="utf-8")
    timestamp_safe = metadata.timestamp_utc.replace(":", "-")
    (results_dir / f"{timestamp_safe}.json").write_text(payload, encoding="utf-8")

    return results


def _format_summary(results: EvalResults) -> str:
    """Plain ASCII aligned summary for stdout. Captures cleanly into PR
    descriptions and CI logs.
    """
    m = results.aggregate
    md = results.metadata
    lines: list[str] = []
    lines.append(
        f"=== eval results ({md.provider}/{md.model}, n={m.n_tickets}, errors={m.n_errors}) ==="
    )

    def _fmt(value: float | None, *, percent: bool = False) -> str:
        if value is None:
            return "n/a"
        return f"{value * 100:5.1f}%" if percent else f"{value:.3f}"

    lines.append(f"  Category accuracy             : {_fmt(m.category_accuracy, percent=True)}")
    lines.append(f"  Priority accuracy             : {_fmt(m.priority_accuracy, percent=True)}")
    lines.append(f"  Clarification precision       : {_fmt(m.clarification_precision)}")
    lines.append(f"  Clarification recall          : {_fmt(m.clarification_recall)}")
    lines.append(f"  Clarification F1              : {_fmt(m.clarification_f1)}")
    lines.append(f"  Mean confidence (overall)     : {_fmt(m.mean_confidence_overall)}")
    lines.append(f"  Mean confidence (solve only)  : {_fmt(m.mean_confidence_solve)}")
    lines.append(f"  Mean confidence (clarify only): {_fmt(m.mean_confidence_clarify)}")
    lines.append("")
    lines.append("  Per-coverage-case:")
    lines.append(f"    {'case':<28}  {'n':>3}  {'cat':>6}  {'prio':>6}  {'conf':>6}")
    for case in sorted(m.per_coverage_case):
        cm = m.per_coverage_case[case]
        lines.append(
            f"    {case:<28}  {cm.n:>3}  "
            f"{_fmt(cm.category_accuracy, percent=True):>6}  "
            f"{_fmt(cm.priority_accuracy, percent=True):>6}  "
            f"{_fmt(cm.mean_confidence):>6}"
        )
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    import typing

    providers = list(typing.get_args(LLMProvider))
    parser = argparse.ArgumentParser(
        prog="eval.runner",
        description="Run the WSCAD ticket-triage pipeline over the hand-labeled eval set.",
    )
    parser.add_argument(
        "--eval-set",
        default=str(DEFAULT_EVAL_SET),
        metavar="PATH",
        help=f"Hand-labeled eval set JSON (default: {DEFAULT_EVAL_SET}).",
    )
    parser.add_argument(
        "--kb-dir",
        default=str(DEFAULT_KB_DIR),
        metavar="DIR",
        help=f"KB directory (default: {DEFAULT_KB_DIR}).",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        metavar="DIR",
        help=f"Results output directory (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--provider",
        choices=providers,
        default=None,
        metavar="PROVIDER",
        help=(f"LLM provider: {', '.join(providers)}. Overrides WSCAD_TRIAGE_PROVIDER env var."),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m eval.runner`` entry point."""
    args = _parse_args(argv)

    eval_set_path = Path(args.eval_set)
    if not eval_set_path.is_file():
        print(f"eval.runner: eval set not found: {eval_set_path}", file=sys.stderr)
        return 2

    kb_dir = Path(args.kb_dir)
    if not kb_dir.is_dir():
        print(f"eval.runner: KB directory not found: {kb_dir}", file=sys.stderr)
        return 2

    raw = json.loads(eval_set_path.read_text(encoding="utf-8"))
    eval_tickets = [EvalTicket.model_validate(item) for item in raw]

    settings: Settings = (
        Settings(**{"WSCAD_TRIAGE_PROVIDER": args.provider}) if args.provider else Settings()
    )
    try:
        llm = make_client(settings)
    except ConfigurationError as exc:
        print(f"eval.runner: configuration error: {exc}", file=sys.stderr)
        return 1

    chunks = load_kb(kb_dir)
    retriever = HybridRetriever(BM25(chunks), Embedding(chunks))

    metadata = collect_metadata(settings, eval_set_path, kb_chunk_count=len(chunks))

    results = run_eval(
        eval_tickets,
        llm,
        retriever,
        metadata,
        Path(args.results_dir),
    )

    print(_format_summary(results))
    return 0 if results.aggregate.n_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
