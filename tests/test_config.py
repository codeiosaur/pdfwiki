"""
Shared test fixtures for the pipeline test suite.

The MockBackend lets you test pipeline logic without hitting a real LLM.
Set mock_responses on the fixture to control what the "model" returns.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

try:
    import pytest
except ImportError:
    pytest = None  # type: ignore[assignment]

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backend.base import BackendConfig, LLMBackend, LLMBackendError
from extract.fact_extractor import Fact


class MockBackend(LLMBackend):
    """
    A fake LLM backend for testing.

    Set `responses` to a list of strings — each call to generate()
    pops and returns the next one.  If the list is empty, returns
    `default_response`.  If `should_fail` is True, raises LLMBackendError.
    """

    def __init__(
        self,
        responses: Optional[list[str]] = None,
        default_response: str = "{}",
        should_fail: bool = False,
    ) -> None:
        config = BackendConfig(
            provider="openai_compat",
            base_url="http://test:0/v1",
            model="mock-model",
            label="mock",
        )
        super().__init__(config)
        self.responses: list[str] = responses or []
        self.default_response = default_response
        self.should_fail = should_fail
        self.call_log: list[dict] = []

    def generate(self, prompt: str, max_tokens: Optional[int] = None, json_schema: Optional[dict] = None) -> str:
        self.call_log.append({"prompt": prompt, "max_tokens": max_tokens, "json_schema": json_schema})

        if self.should_fail:
            raise LLMBackendError("Mock failure")

        if self.responses:
            return self.responses.pop(0)
        return self.default_response


def _fixture(func):
    """Apply @pytest.fixture when pytest is available, otherwise return func as-is."""
    if pytest is not None:
        return pytest.fixture(func)
    return func


@_fixture
def mock_backend():
    """Provides a fresh MockBackend for each test."""
    return MockBackend()


@_fixture
def failing_backend():
    """Provides a MockBackend that always raises LLMBackendError."""
    return MockBackend(should_fail=True)


@_fixture
def sample_facts():
    """A small set of realistic Fact objects for testing."""
    return [
        Fact(id="1", concept="First In First Out",
             content="FIFO is a method where the oldest costs are used.",
             source_chunk_id="chunk-1"),
        Fact(id="2", concept="First In First Out",
             content="Under FIFO, ending inventory reflects recent prices.",
             source_chunk_id="chunk-1"),
        Fact(id="3", concept="Last In First Out",
             content="LIFO assigns the most recent costs to COGS.",
             source_chunk_id="chunk-1"),
        Fact(id="4", concept="Last In First Out",
             content="LIFO is not permitted under IFRS.",
             source_chunk_id="chunk-2"),
        Fact(id="5", concept="Inventory Turnover Ratio",
             content="Inventory turnover measures how often inventory is sold.",
             source_chunk_id="chunk-2"),
        Fact(id="6", concept="Inventory Turnover Ratio",
             content="Inventory Turnover = COGS / Average Inventory",
             source_chunk_id="chunk-2"),
        Fact(id="7", concept="Inventory Fraud",
             content="Inventory fraud involves manipulating values to misstate results.",
             source_chunk_id="chunk-3"),
        Fact(id="8", concept="Perpetual Inventory",
             content="A perpetual system updates records continuously.",
             source_chunk_id="chunk-3"),
        Fact(id="9", concept="Perpetual Inventory",
             content="Under perpetual, COGS is recorded at time of each sale.",
             source_chunk_id="chunk-1"),
    ]


@dataclass
class MockChunk:
    """Minimal chunk for testing without importing pdf_loader."""
    id: str
    text: str
    source: str = "test.pdf"
    chapter: Optional[str] = None


@_fixture
def sample_chunks():
    """A small set of text chunks for extraction testing."""
    return [
        MockChunk(
            id="chunk-aaa",
            text="The FIFO method assumes the oldest inventory is sold first. "
                 "Under FIFO, ending inventory reflects the most recent costs.",
        ),
        MockChunk(
            id="chunk-bbb",
            text="The LIFO method assumes the newest inventory is sold first. "
                 "LIFO is not permitted under International Financial Reporting Standards.",
        ),
    ]