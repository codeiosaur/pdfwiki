"""
Shared test fixtures for the pipeline test suite.

The MockBackend lets you test pipeline logic without hitting a real LLM.
Set responses on the fixture to control what the "model" returns.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import pytest

# Ensure src/ is importable from all test files
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ---------------------------------------------------------------------------
# Register domain antonym pairs once per session so tests that exercise
# domain-specific antonym logic (e.g. periodic/perpetual, debit/credit) work
# without needing to spin up a full pipeline.  This mirrors what pipeline.py
# does at startup when a seeds file is provided.
# ---------------------------------------------------------------------------
_SEEDS_FILE = Path(__file__).resolve().parent.parent / "seeds" / "accounting-top500.json"


@pytest.fixture(autouse=True, scope="session")
def _register_accounting_antonyms():
    from extract.fact_extractor import load_antonyms_from_file
    from transform.matching import register_antonym_pairs
    if _SEEDS_FILE.exists():
        pairs = load_antonyms_from_file(str(_SEEDS_FILE))
        if pairs:
            register_antonym_pairs(pairs)

# ---------------------------------------------------------------------------
# LLM env-var prefixes that should never leak from the real shell or .env
# file into tests. Any os.environ key that starts with one of these is
# cleared at the start of every test.  Individual tests that need a specific
# value use monkeypatch.setenv() explicitly.
# ---------------------------------------------------------------------------
_LLM_ENV_PREFIXES = (
    "PASS1_", "PASS2_", "PASS3_",
    "LLM_",
    "OPENROUTER_", "ANTHROPIC_", "OPENAI_",
    "PIPELINE_", "TWO_PASS", "ENHANCED_",
    "OLLAMA_",
)


@pytest.fixture(autouse=True)
def _isolated_llm_env(monkeypatch):
    """
    Prevent real shell variables and .env file contents from leaking into
    any test.

    Two-step isolation:
      1. Patch _load_dotenv to a no-op so the .env file is never read.
         (Without this, monkeypatch.delenv triggers a re-read on the next
         get_env() call because _load_dotenv only skips keys already present
         in os.environ.)
      2. Remove every LLM-config var that is currently set in the shell so
         tests start from a known-empty state.
    """
    monkeypatch.setattr("backend.config._load_dotenv", lambda: None)
    for key in list(os.environ):
        if any(key.startswith(p) or key == p.rstrip("_") for p in _LLM_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)

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

    def generate(self, prompt: str, max_tokens: Optional[int] = None, json_schema: Optional[dict] = None, context: str = "", system_prompt: Optional[str] = None) -> str:
        self.call_log.append({"prompt": prompt, "max_tokens": max_tokens, "json_schema": json_schema})

        if self.should_fail:
            raise LLMBackendError("Mock failure")

        if self.responses:
            return self.responses.pop(0)
        return self.default_response


@pytest.fixture
def mock_backend():
    """Provides a fresh MockBackend for each test."""
    return MockBackend()


@pytest.fixture
def failing_backend():
    """Provides a MockBackend that always raises LLMBackendError."""
    return MockBackend(should_fail=True)


@pytest.fixture
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


@pytest.fixture
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
