"""Output serialisation: JSON writer and human-readable text renderers."""

from wscad_triage.output.json_writer import write_json
from wscad_triage.output.text_renderer import render_text, render_text_compact, write_text

__all__ = ["render_text", "render_text_compact", "write_json", "write_text"]
