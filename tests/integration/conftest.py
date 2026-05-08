"""Shared fixtures for integration tests (issue #25).

The integration suite drives the full LangGraph pipeline with a scripted
``MockLLMClient`` and a ``_StubRetriever`` -- no network, no disk-cached
embedder, no real Anthropic SDK calls. The conftest provides the two
factories every scenario reuses:

- ``stub_retriever_factory(*results)`` -- a minimal HybridRetriever
  stand-in that returns a fixed result list.
- ``mock_llm(**scripts)`` -- thin wrapper over ``MockLLMClient`` so test
  bodies stay focused on the assertion surface.

The unit-test ``_scrub_settings_env`` autouse fixture (in
``tests/unit/conftest.py``) is intentionally NOT inherited here: the
live-API smoke test (``test_anthropic_live.py``) needs real env vars to
pass through. Integration tests in this directory don't read env vars,
so the lack of scrubbing is harmless.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import pytest

from wscad_triage.llm import LLMResponse, MockLLMClient
from wscad_triage.schemas import RetrievalResult


@dataclass
class StubRetriever:
    """HybridRetriever stand-in that returns a fixed result list.

    Records every ``retrieve`` call so tests can assert on the rewritten
    query and ``k``. Mirrors the unit-test stub at
    ``tests/unit/test_agent_retrieve.py`` so the supervisor sees the
    same shape.
    """

    results: list[RetrievalResult]
    calls: list[tuple[str, int]] = field(default_factory=list)

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        self.calls.append((query, k))
        return self.results[:k]


@pytest.fixture
def stub_retriever_factory() -> Any:
    """Factory: ``stub_retriever_factory(result, ...)`` -> ``StubRetriever``."""

    def _make(*results: RetrievalResult) -> StubRetriever:
        return StubRetriever(results=list(results))

    return _make


@pytest.fixture
def mock_llm() -> Any:
    """Factory for a scripted MockLLMClient.

    Usage: ``mock_llm(**{"triage agent": resp_a, "retrieve agent": resp_b})``.
    Keys are substring triggers; values are ``LLMResponse`` or iterables
    thereof (consumed FIFO).
    """

    def _make(**scripts: LLMResponse | Iterable[LLMResponse]) -> MockLLMClient:
        return MockLLMClient(script=dict(scripts))

    return _make
