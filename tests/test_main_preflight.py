import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backend.base import BackendConfig
from pipeline import PipelineMetrics, run_pipeline_two_pass, validate_pipeline_inputs


def test_validate_pipeline_inputs_accepts_existing_paths(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_text("pdf", encoding="utf-8")
    output_dir = tmp_path / "vault"
    seeds_file = tmp_path / "seeds.json"
    seeds_file.write_text('["FIFO"]', encoding="utf-8")

    validate_pipeline_inputs(str(pdf_path), output_dir, seeds_file=str(seeds_file))


def test_validate_pipeline_inputs_rejects_missing_pdf(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_dir = tmp_path / "vault"

    with pytest.raises(SystemExit):
        validate_pipeline_inputs(str(tmp_path / "missing.pdf"), output_dir)

    captured = capsys.readouterr()
    assert "PDF file not found" in captured.out


def test_validate_pipeline_inputs_rejects_missing_seeds(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_text("pdf", encoding="utf-8")

    with pytest.raises(SystemExit):
        validate_pipeline_inputs(str(pdf_path), tmp_path / "vault", seeds_file=str(tmp_path / "missing.json"))

    captured = capsys.readouterr()
    assert "Seeds file not found" in captured.out


def _make_mock_backend() -> MagicMock:
    backend = MagicMock()
    backend.provider = "openai_compat"
    backend.base_url = "http://localhost:11434/v1"
    backend.label = "mock"
    backend.model = "mock-model"
    backend.is_openrouter = False
    return backend


def test_pass1_batch_size_controls_chunk_batching() -> None:
    """pass1_batch_size controls how many chunks go into each Pass 1 extraction batch."""
    from ingest.pdf_loader import Chunk

    fake_chunks = [
        Chunk(id=f"c{i}", text="some text", source="test.pdf", chapter=None)
        for i in range(6)
    ]
    captured_batch_sizes: list[int] = []

    def mock_extract(batch, backend, batch_size):
        captured_batch_sizes.append(len(batch))
        return []

    backend = _make_mock_backend()
    with patch("pipeline.load_pdf_chunks", return_value=fake_chunks), \
         patch("pipeline.extract_raw_statements_batched", side_effect=mock_extract), \
         patch("pipeline.resolve_seed_concepts", return_value=["FIFO"]), \
         patch("pipeline.assign_concepts_to_statements", return_value=[]):

        run_pipeline_two_pass(
            "dummy.pdf",
            pass1_backend=backend,
            pass2_backend=backend,
            batch_size=2,
            pass1_batch_size=3,
        )

    # 6 chunks with pass1_batch_size=3 → 2 batches of 3 each
    assert captured_batch_sizes == [3, 3]


def test_pass1_batch_size_falls_back_to_batch_size() -> None:
    """When pass1_batch_size is None, batch_size is used for Pass 1."""
    from ingest.pdf_loader import Chunk

    fake_chunks = [
        Chunk(id=f"c{i}", text="some text", source="test.pdf", chapter=None)
        for i in range(4)
    ]
    captured_batch_sizes: list[int] = []

    def mock_extract(batch, backend, batch_size):
        captured_batch_sizes.append(len(batch))
        return []

    backend = _make_mock_backend()
    with patch("pipeline.load_pdf_chunks", return_value=fake_chunks), \
         patch("pipeline.extract_raw_statements_batched", side_effect=mock_extract), \
         patch("pipeline.resolve_seed_concepts", return_value=["FIFO"]), \
         patch("pipeline.assign_concepts_to_statements", return_value=[]):

        run_pipeline_two_pass(
            "dummy.pdf",
            pass1_backend=backend,
            pass2_backend=backend,
            batch_size=2,
        )

    # 4 chunks with batch_size=2 (no pass1_batch_size override) → 2 batches of 2
    assert captured_batch_sizes == [2, 2]


# ── Observability / metrics tests ─────────────────────────────────────────────

def test_openai_compat_backend_metrics_structure() -> None:
    """metrics() returns a dict with the expected non-negative integer keys."""
    from backend.openai_compat import OpenAICompatBackend

    # Build a mock openai client that returns a valid response immediately.
    mock_choice = SimpleNamespace(
        message=SimpleNamespace(content="hello", tool_calls=None, refusal=None),
    )
    mock_response = SimpleNamespace(choices=[mock_choice], usage=None)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    config = BackendConfig(
        provider="openai_compat",
        base_url="http://localhost:11434/v1",
        model="mock-model",
        label="test",
    )
    with patch("backend.openai_compat.openai") as mock_openai:
        mock_openai.OpenAI.return_value = mock_client
        backend = OpenAICompatBackend(config)

    m = backend.metrics()
    assert isinstance(m, dict)
    assert set(m.keys()) == {"total_requests", "retry_count", "fallback_hops"}
    assert all(isinstance(v, int) and v >= 0 for v in m.values())


def test_openai_compat_backend_metrics_increment_on_request() -> None:
    """total_requests increments once per generate() call."""
    from backend.openai_compat import OpenAICompatBackend

    mock_choice = SimpleNamespace(
        message=SimpleNamespace(content="ok", tool_calls=None, refusal=None),
    )
    mock_response = SimpleNamespace(choices=[mock_choice], usage=None)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    config = BackendConfig(
        provider="openai_compat",
        base_url="http://localhost:11434/v1",
        model="mock-model",
        label="test",
    )
    with patch("backend.openai_compat.openai") as mock_openai:
        mock_openai.OpenAI.return_value = mock_client
        backend = OpenAICompatBackend(config)

    assert backend.metrics()["total_requests"] == 0
    backend.generate("prompt one")
    assert backend.metrics()["total_requests"] == 1
    backend.generate("prompt two")
    assert backend.metrics()["total_requests"] == 2


def test_pipeline_metrics_dataclass_defaults() -> None:
    """PipelineMetrics defaults to all-zero values and correct field names."""
    m = PipelineMetrics()
    assert m.total_chunks == 0
    assert m.total_statements == 0
    assert m.total_facts == 0
    assert m.pass1_time_s == 0.0
    assert m.pass2_time_s == 0.0
    assert m.pass1_retries == 0
    assert m.pass1_fallback_hops == 0
    assert m.pass2_retries == 0
    assert m.pass2_fallback_hops == 0


def test_pipeline_metrics_print_summary_contains_key_fields(capsys: pytest.CaptureFixture[str]) -> None:
    """print_summary() outputs all metric labels."""
    m = PipelineMetrics(total_chunks=10, pass1_retries=3, pass2_fallback_hops=1)
    m.print_summary()
    out = capsys.readouterr().out
    assert "pass1_retries" in out
    assert "pass2_fallback_hops" in out
    assert "chunks" in out


def test_pass2_batch_size_passed_to_assign_concepts() -> None:
    """pass2_batch_size is forwarded to assign_concepts_to_statements."""
    from ingest.pdf_loader import Chunk

    fake_chunks = [Chunk(id="c1", text="some text", source="test.pdf", chapter=None)]
    fake_statements = [
        {"statement": f"fact {i}", "chunk_id": "c1", "source": "test.pdf"}
        for i in range(5)
    ]
    captured_inner_batch_sizes: list[int] = []

    def mock_assign(statements, backend, seeds, batch_size, strict):
        captured_inner_batch_sizes.append(batch_size)
        return []

    backend = _make_mock_backend()
    with patch("pipeline.load_pdf_chunks", return_value=fake_chunks), \
         patch("pipeline.extract_raw_statements_batched", return_value=fake_statements), \
         patch("pipeline.resolve_seed_concepts", return_value=["FIFO"]), \
         patch("pipeline.assign_concepts_to_statements", side_effect=mock_assign):

        run_pipeline_two_pass(
            "dummy.pdf",
            pass1_backend=backend,
            pass2_backend=backend,
            batch_size=2,
            pass2_batch_size=8,
        )

    assert all(bs == 8 for bs in captured_inner_batch_sizes)