from pathlib import Path

import pytest

import pdf_to_obsidian.main as main


def test_run_cli_processes_each_pdf_in_multi_mode(monkeypatch, tmp_path):
    calls = []

    def fake_process_pdf(pdf_path, output_dir, subject_override="", batch_mode=False):
        calls.append((pdf_path, output_dir, subject_override, batch_mode))

    monkeypatch.setattr(main, "process_pdf", fake_process_pdf)

    code = main.run_cli(["one.pdf", "two.pdf", "--vault", str(tmp_path)])

    assert code == 0
    assert len(calls) == 2
    assert calls[0][0] == "one.pdf"
    assert calls[1][0] == "two.pdf"
    assert calls[0][1] == str(tmp_path)
    assert calls[1][1] == str(tmp_path)
    assert calls[0][2] == ""
    assert calls[1][2] == ""
    assert calls[0][3] is True
    assert calls[1][3] is True


def test_run_cli_single_pdf_respects_subject_override(monkeypatch, tmp_path):
    calls = []

    def fake_process_pdf(pdf_path, output_dir, subject_override="", batch_mode=False):
        calls.append((pdf_path, output_dir, subject_override, batch_mode))

    monkeypatch.setattr(main, "process_pdf", fake_process_pdf)

    code = main.run_cli([
        "lecture.pdf",
        "--vault",
        str(tmp_path),
        "--subject",
        "Cryptography",
    ])

    assert code == 0
    assert len(calls) == 1
    assert calls[0][2] == "Cryptography"
    assert calls[0][3] is False


def test_run_cli_rejects_subject_with_multiple_pdfs(tmp_path):
    with pytest.raises(SystemExit):
        main.run_cli([
            "one.pdf",
            "two.pdf",
            "--vault",
            str(tmp_path),
            "--subject",
            "Cryptography",
        ])
