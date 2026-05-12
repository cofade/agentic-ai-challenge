"""Command-line entry point for ``wscad-triage``.

Two subcommands::

    wscad-triage chat  [--out DIR] [--kb-dir DIR] [--provider PROVIDER]
        Interactive REPL for human-in-the-loop ticket triage (Phase 8 — #65).

    wscad-triage batch <tickets.json> [--out DIR] [--kb-dir DIR] [--provider PROVIDER]
        Process every ticket in *tickets.json* end-to-end and write two
        artefacts per ticket (Phase 4 — #30).

The historical positional form ``wscad-triage <tickets.json>`` (no
``batch`` keyword) still routes to the batch subcommand so existing
quickstart commands and scripts continue to work.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from wscad_triage import chat as chat_module
from wscad_triage import pipeline
from wscad_triage.kb import BM25, Embedding, HybridRetriever, load_kb
from wscad_triage.llm import LLMClient
from wscad_triage.llm.errors import ConfigurationError
from wscad_triage.llm.factory import make_client
from wscad_triage.output import write_json, write_text
from wscad_triage.schemas import Ticket
from wscad_triage.settings import LLMProvider, Settings

_SUBCOMMANDS = ("chat", "batch")


def main(argv: list[str] | None = None) -> int:
    """Entry point referenced by ``[project.scripts]``.

    Dispatch order:

    1. First arg is ``chat`` → run the REPL.
    2. First arg is ``batch`` → run the batch processor on the next arg.
    3. Otherwise → route to the batch processor for backward compatibility
       with the historical ``wscad-triage tickets.json`` form.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if args and args[0] == "chat":
        return _run_chat(args[1:])
    if args and args[0] == "batch":
        return _run_batch(args[1:])
    return _run_batch(args)


# ---------------------------------------------------------------------------
# Batch subcommand (legacy positional form preserved)


def _run_batch(argv: list[str]) -> int:
    args = _parse_batch_args(argv)

    tickets_path = Path(args.tickets_json)
    try:
        raw = json.loads(tickets_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"wscad-triage: file not found: {tickets_path}", file=sys.stderr)
        return 2

    tickets = [Ticket.model_validate(t) for t in raw]

    kb_dir = Path(args.kb_dir)
    if not kb_dir.is_dir():
        print(
            f"wscad-triage: KB directory not found: {kb_dir}\n"
            "Run from the project root or pass --kb-dir.",
            file=sys.stderr,
        )
        return 2

    try:
        llm, retriever = _setup_runtime(args.provider, kb_dir)
    except ConfigurationError as exc:
        print(f"wscad-triage: configuration error: {exc}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ticket in tickets:
        output = pipeline.run(ticket, llm, retriever)
        # Sanitise the user-controlled ``ticket_id`` before path
        # concatenation — same protection the chat REPL applies — so a
        # hostile tickets.json entry like ``{"ticket_id": "../escape"}``
        # cannot land artefacts outside ``out_dir``.
        stem = chat_module.safe_filename(ticket.ticket_id)
        write_json(output, out_dir / f"{stem}.json")
        write_text(output, out_dir / f"{stem}.txt")
        print(f"[wscad-triage] {ticket.ticket_id} -> {out_dir}")

    return 0


# ---------------------------------------------------------------------------
# Chat subcommand


def _run_chat(argv: list[str]) -> int:
    args = _parse_chat_args(argv)

    kb_dir = Path(args.kb_dir)
    if not kb_dir.is_dir():
        print(
            f"wscad-triage: KB directory not found: {kb_dir}\n"
            "Run from the project root or pass --kb-dir.",
            file=sys.stderr,
        )
        return 2

    try:
        llm, retriever = _setup_runtime(args.provider, kb_dir)
    except ConfigurationError as exc:
        print(f"wscad-triage: configuration error: {exc}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    return chat_module.run_repl(llm=llm, retriever=retriever, out_dir=out_dir)


# ---------------------------------------------------------------------------
# Shared runtime setup


def _setup_runtime(provider: str | None, kb_dir: Path) -> tuple[LLMClient, HybridRetriever]:
    """Build the LLM client and retriever for both subcommands.

    Provider preflight (Ollama health check) runs before any KB work so
    a misconfigured backend fails fast with a friendly message rather
    than after the embedder has loaded.
    """
    # The alias-keyed kwarg lets pydantic-settings see the override via the
    # public env-var name; typed as ``dict[str, Any]`` because Settings has
    # heterogeneously-typed fields and the ``**`` expansion would otherwise
    # narrow to one field's type.
    overrides: dict[str, Any] = {}
    if provider:
        overrides["WSCAD_TRIAGE_PROVIDER"] = provider
    settings = Settings(**overrides)
    llm = make_client(settings)
    if settings.llm_provider == "ollama":
        _preflight_ollama(settings.ollama_base_url, settings.ollama_model)

    chunks = load_kb(kb_dir)
    retriever = HybridRetriever(BM25(chunks), Embedding(chunks))
    return llm, retriever


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


# ---------------------------------------------------------------------------
# Argparse


def _parse_batch_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="wscad-triage batch",
        description="Process every ticket in a JSON file end-to-end.",
    )
    parser.add_argument(
        "tickets_json",
        metavar="tickets.json",
        help="Path to the JSON file containing a list of tickets.",
    )
    _add_shared_flags(parser)
    return parser.parse_args(argv)


def _parse_chat_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="wscad-triage chat",
        description="Interactive chat REPL for human-in-the-loop ticket triage.",
    )
    _add_shared_flags(parser)
    return parser.parse_args(argv)


def _add_shared_flags(parser: argparse.ArgumentParser) -> None:
    import typing

    providers = list(typing.get_args(LLMProvider))
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


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
