"""Test cross-PDF deduplication in batch mode."""

import pytest
import pdfwiki.main as main
import pdfwiki.concept_quality as concept_quality


def test_dedupe_concepts_for_run_with_existing_concepts():
    """Verify concepts from first PDF are deduped in second PDF."""
    existing = ["RSA", "Hash Function", "Symmetric Encryption"]
    new_concepts = [
        "RSA (Rivest-Shamir-Adleman)",  # Should be deduped (near-match with RSA)
        "SHA-256",  # New concept
        "Asymmetric Encryption",  # Should NOT be deduped (different from Symmetric)
        "Elliptic Curve",  # New concept
    ]

    kept, dropped = concept_quality.dedupe_concepts_for_run(new_concepts, existing_concepts=existing)

    # Should keep only new concepts that don't match existing ones
    assert len(dropped) == 1, f"Expected 1 dropped (RSA), got {dropped}"
    assert "SHA-256" in kept
    assert "Elliptic Curve" in kept
    assert "Asymmetric Encryption" in kept
    # RSA should be dropped as near-duplicate
    assert any("RSA" in concept for concept, _ in dropped)


def test_process_pdf_accumulates_concepts_across_batch(monkeypatch, tmp_path):
    """Verify process_pdf returns master concept list for next PDF."""
    
    # Mock the pipeline to return a fixed concept list
    def mock_build_index(*args, **kwargs):
        return (["Concept A", "Concept B"], "INDEX_OUTPUT")

    def mock_filter_concepts(*args, **kwargs):
        return (["Concept A", "Concept B"], [])

    def mock_dedupe(*args, **kwargs):
        return (["Concept A", "Concept B"], [])

    # Mock other pipeline functions to do minimal work
    monkeypatch.setattr(main, "extract_text", lambda *a, **k: "text")
    monkeypatch.setattr(main, "split_into_chapters", lambda *a, **k: [{"content": "test"}])
    monkeypatch.setattr(main, "smart_chunk", lambda *a, **k: ["chunk1"])
    monkeypatch.setattr(main, "_build_index", mock_build_index)
    monkeypatch.setattr(main, "_filter_concepts_with_evidence", mock_filter_concepts)
    monkeypatch.setattr(main, "_dedupe_concepts_for_run", mock_dedupe)
    monkeypatch.setattr(main, "_get_subject", lambda *a, **k: "Test")
    monkeypatch.setattr(main, "run_concept_indexing", lambda **k: type('obj', (object,), {
        'concepts': mock_dedupe(mock_filter_concepts(mock_build_index()[0])[0])[0],
        'index_text': "INDEX",
        'dropped_concepts': [],
        'deduped_pairs': []
    })())
    monkeypatch.setattr(main, "load_vault_state", lambda *a, **k: {"pages": {}})
    monkeypatch.setattr(main, "_collect_vault_pages", lambda *a, **k: [])
    monkeypatch.setattr(main, "run_concept_page_workflow", lambda **k: type('obj', (object,), {
        'wiki_pages': {},
        'merged_pages': {},
        'skipped': []
    })())
    monkeypatch.setattr(main, "load_prompt", lambda *a, **k: "")
    monkeypatch.setattr(main, "maybe_regenerate_moc", lambda **k: None)
    monkeypatch.setattr(main, "generate_study_aids", lambda **k: None)

    # First PDF returns concepts
    master_list = main.process_pdf("pdf1.pdf", str(tmp_path), batch_mode=True, existing_concepts=None)
    assert "Concept A" in master_list
    assert "Concept B" in master_list

    # Second PDF should get the master list (in real scenario would dedupe against it)
    master_list_2 = main.process_pdf("pdf2.pdf", str(tmp_path), batch_mode=True, existing_concepts=master_list)
    # In batch mode, this would contain both previous + new concepts
    assert len(master_list_2) >= 2


def test_run_cli_passes_master_concepts_between_pdfs(monkeypatch, tmp_path):
    """Verify CLI maintains and passes master concept list across PDFs."""
    concept_calls = []

    def track_process_pdf(pdf_path, output_dir, subject_override="", batch_mode=False, max_workers=None, profile=None, existing_concepts=None):
        concept_calls.append({
            'pdf': pdf_path,
            'existing': existing_concepts,
        })
        # Return updated master list (in real scenario, would accumulate)
        return (existing_concepts or []) + [f"concept_from_{pdf_path}"]

    monkeypatch.setattr(main, "process_pdf", track_process_pdf)

    code = main.run_cli(["first.pdf", "second.pdf", "--vault", str(tmp_path), "--batch"])

    assert code == 0
    assert len(concept_calls) == 2
    
    # First PDF should get no existing concepts
    assert concept_calls[0]['existing'] is None
    
    # Second PDF should get concepts from first PDF
    assert concept_calls[1]['existing'] is not None
    assert "concept_from_first.pdf" in concept_calls[1]['existing']


def test_dedupe_concepts_for_run_handles_parenthetical_alias_variants():
    existing = ["One-Time Pad (OTP)"]
    new_concepts = ["OTP (One-Time Pad)", "AES"]

    kept, dropped = concept_quality.dedupe_concepts_for_run(new_concepts, existing_concepts=existing)

    assert kept == ["AES"]
    assert dropped == [("OTP (One-Time Pad)", "One-Time Pad (OTP)")]


def test_dedupe_concepts_for_run_handles_symmetric_key_naming_variants():
    existing = ["Symmetric Key Cryptography"]
    new_concepts = ["Symmetric-Key Encryption", "Asymmetric-Key Encryption"]

    kept, dropped = concept_quality.dedupe_concepts_for_run(new_concepts, existing_concepts=existing)

    assert "Symmetric-Key Encryption" not in kept
    assert ("Symmetric-Key Encryption", "Symmetric Key Cryptography") in dropped
    assert "Asymmetric-Key Encryption" in kept
