from pathlib import Path

import pytest

from pipeline import validate_pipeline_inputs


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