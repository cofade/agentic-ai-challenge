"""Command-line entry point for wscad-triage.

The CLI is fleshed out in Phase 4 (issue #30). This stub exists so the package
is installable from Phase 0 and `wscad-triage --help` resolves.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Entry point referenced by `[project.scripts]`."""
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] in {"-h", "--help"}:
        print("wscad-triage: agentic ticket triage (CLI implemented in Phase 4 / issue #30)")
        return 0
    print("wscad-triage CLI not yet implemented. See docs/ROADMAP.md (issue #30).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
