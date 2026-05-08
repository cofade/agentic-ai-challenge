"""Errors raised by the LLM layer."""

from __future__ import annotations


class ConfigurationError(Exception):
    """Raised when an LLM backend cannot be constructed for the current config.

    Distinguished from generic ``ValueError`` so callers (CLI, factory, tests)
    can catch the configuration-specific case without swallowing unrelated
    errors. Message must name the offending setting and the resolution path.
    """
