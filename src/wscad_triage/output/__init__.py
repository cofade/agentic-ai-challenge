"""Output serialisation: JSON writer and human-readable text renderer."""

from wscad_triage.output.json_writer import write_json
from wscad_triage.output.text_renderer import render_text, write_text

__all__ = ["render_text", "write_json", "write_text"]
