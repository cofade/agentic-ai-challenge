"""Command-line entry point for wscad-triage (Phase 4 — issue #30).

Usage::

    wscad-triage <tickets.json> [--out out/] [--kb-dir kb/] [--provider ollama|anthropic|azure]

For each ticket in *tickets.json* the pipeline is run end-to-end and two
artefacts are written to the output directory:

- ``{ticket_id}.json`` — canonical JSON (Pydantic model_dump_json)
- ``{ticket_id}.txt`` — human-readable text (mirrors Sample_Output.txt)
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from wscad_triage import pipeline
from wscad_triage.kb import BM25, Embedding, HybridRetriever, load_kb
from wscad_triage.llm.errors import ConfigurationError
from wscad_triage.llm.factory import make_client
from wscad_triage.output import write_json, write_text
from wscad_triage.schemas import Ticket
from wscad_triage.settings import LLMProvider, Settings


def main(argv: list[str] | None = None) -> int:
    """Entry point referenced by ``[project.scripts]``."""
    args = _parse_args(argv)

    # Load tickets
    tickets_path = Path(args.tickets_json)
    try:
        raw = json.loads(tickets_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"wscad-triage: file not found: {tickets_path}", file=sys.stderr)
        return 2

    tickets = [Ticket.model_validate(t) for t in raw]

    # Validate KB directory
    kb_dir = Path(args.kb_dir)
    if not kb_dir.is_dir():
        print(
            f"wscad-triage: KB directory not found: {kb_dir}\n"
            "Run from the project root or pass --kb-dir.",
            file=sys.stderr,
        )
        return 2

    # Build LLM client — pass provider via the alias kwarg understood by pydantic-settings
    settings: Settings = (
        Settings(**{"WSCAD_TRIAGE_PROVIDER": args.provider}) if args.provider else Settings()
    )
    try:
        llm = make_client(settings)
        if settings.llm_provider == "ollama":
            _preflight_ollama(settings.ollama_base_url, settings.ollama_model)
    except ConfigurationError as exc:
        print(f"wscad-triage: configuration error: {exc}", file=sys.stderr)
        return 1

    # Build retriever
    chunks = load_kb(kb_dir)
    retriever = HybridRetriever(BM25(chunks), Embedding(chunks))

    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process tickets
    for ticket in tickets:
        output = pipeline.run(ticket, llm, retriever)
        write_json(output, out_dir / f"{ticket.ticket_id}.json")
        write_text(output, out_dir / f"{ticket.ticket_id}.txt")
        print(f"[wscad-triage] {ticket.ticket_id} -> {out_dir}")

    return 0


def _preflight_ollama(base_url: str, model: str) -> None:
    """Verify the Ollama server is up and the configured model is pulled.

    Raises ``ConfigurationError`` with an actionable message if either check
    fails. The verifier-style two-stage gate (server reachable, then model
    present) names the right cause in the error message so the reviewer can
    fix it without reading the traceback.
    """
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/version", timeout=3.0):
            pass
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ConfigurationError(
            f"Ollama server not reachable at {base_url}: {exc}. "
            "Start it with `ollama serve` or set WSCAD_TRIAGE_OLLAMA_BASE_URL."
        ) from exc

    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=5.0) as resp:
            tags = json.load(resp)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Ollama /api/tags unavailable at {base_url}: {exc}.") from exc

    pulled = {t.get("name", "") for t in tags.get("models", [])}
    if model not in pulled:
        raise ConfigurationError(
            f"Ollama model {model!r} not pulled (available: {sorted(pulled)}). "
            f"Run `ollama pull {model}` to fix."
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    import typing

    providers = list(typing.get_args(LLMProvider))
    parser = argparse.ArgumentParser(
        prog="wscad-triage",
        description="Agentic ticket-triage pipeline for WSCAD support tickets.",
    )
    parser.add_argument(
        "tickets_json",
        metavar="tickets.json",
        help="Path to the JSON file containing a list of tickets.",
    )
    parser.add_argument(
        "--out",
        default="out",
        metavar="DIR",
        help="Output directory for JSON and text artefacts (default: out/).",
    )
    parser.add_argument(
        "--kb-dir",
        default="kb",
        metavar="DIR",
        help="Directory containing the KB documents (default: kb/).",
    )
    parser.add_argument(
        "--provider",
        choices=providers,
        default=None,
        metavar="PROVIDER",
        help=(f"LLM provider: {', '.join(providers)}. Overrides WSCAD_TRIAGE_PROVIDER env var."),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
