from pathlib import Path

import pdfwiki.main as main


def test_parse_index_extracts_only_concepts_section():
    raw = """
CONCEPTS:
1. Encryption - turning plaintext into ciphertext
2. RSA (Rivest-Shamir-Adleman)
3. One-Time Pad (OTP)
RELATIONSHIPS:
- Encryption -> Confidentiality
""".strip()

    concepts, original = main.parse_index(raw)

    assert concepts == [
        "Encryption",
        "RSA (Rivest-Shamir-Adleman)",
        "One-Time Pad (OTP)",
    ]
    assert original == raw


def test_parse_index_accepts_bullets_and_deduplicates():
    raw = """
CONCEPTS:
- RSA - first mention
- RSA - duplicate mention
* AES - symmetric cipher
RELATIONSHIPS:
- RSA -> AES: contrast
""".strip()

    concepts, _ = main.parse_index(raw)

    assert concepts == ["RSA", "AES"]


def test_parse_index_handles_markdown_heading_and_colon_descriptions():
    raw = """
## CONCEPTS
1) **RSA**: public key cryptosystem
2) **Diffie-Hellman Key Exchange**: shared secret protocol

## RELATIONSHIPS
- RSA -> Diffie-Hellman: both public-key era concepts
""".strip()

    concepts, _ = main.parse_index(raw)

    assert concepts == ["RSA", "Diffie-Hellman Key Exchange"]


def test_parse_index_handles_inline_concepts_without_relationships_section():
    raw = "CONCEPTS: RSA, AES, One-Time Pad (OTP)"

    concepts, _ = main.parse_index(raw)

    assert concepts == ["RSA", "AES", "One-Time Pad (OTP)"]


def test_parse_index_trims_sentence_like_concept_lines():
    raw = """
CONCEPTS:
- RSA depends on difficulties in factoring large composites into their prime factors.
- Diffie-Hellman Key Exchange is based on modular arithmetic.
RELATIONSHIPS:
- RSA -> Diffie-Hellman Key Exchange
""".strip()

    concepts, _ = main.parse_index(raw)

    assert concepts == ["RSA", "Diffie-Hellman Key Exchange"]


def test_parse_index_drops_clause_style_non_concepts():
    raw = """
CONCEPTS:
- If an algorithm is deterministic, it may leak patterns.
- Cipher Block Chaining (CBC) Mode
RELATIONSHIPS:
- CBC -> IND-CPA Security
""".strip()

    concepts, _ = main.parse_index(raw)

    assert concepts == ["Cipher Block Chaining (CBC) Mode"]


def test_parse_index_drops_arrow_relationship_style_candidates():
    raw = """
CONCEPTS:
- RSA → Diffie-Hellman
- OTP → Perfect Secrecy
- AES (Advanced Encryption Standard)
RELATIONSHIPS:
- RSA -> Diffie-Hellman
""".strip()

    concepts, _ = main.parse_index(raw)

    assert concepts == ["AES (Advanced Encryption Standard)"]


def test_filter_concepts_with_evidence_removes_hallucinated_item():
    chunks = [
        "CBC mode uses a random IV and improves security over ECB.",
        "Collision resistance is a property of cryptographic hash functions.",
    ]
    concepts = ["Collision Resistance", "Chaocipher"]

    kept, dropped = main._filter_concepts_with_evidence(concepts, chunks)

    assert "Collision Resistance" in kept
    assert "Chaocipher" in dropped


def test_normalize_concept_handles_underscored_titles():
    assert main._normalize_concept("AES_(Advanced_Encryption_Standard)") == "aes"


def test_dedupe_concepts_for_run_collapses_near_duplicates():
    concepts = [
        "AES (Advanced Encryption Standard)",
        "AES_(Advanced_Encryption_Standard)",
    ]

    kept, dropped = main._dedupe_concepts_for_run(concepts)

    assert kept == ["AES (Advanced Encryption Standard)"]
    assert dropped == [("AES_(Advanced_Encryption_Standard)", "AES (Advanced Encryption Standard)")]


def test_dedupe_concepts_for_run_keeps_symmetric_and_asymmetric_distinct():
    concepts = [
        "Symmetric Encryption",
        "Asymmetric Encryption",
    ]

    kept, dropped = main._dedupe_concepts_for_run(concepts)

    assert kept == concepts
    assert dropped == []


def test_parse_wiki_page_uses_filename_header_when_present():
    raw = """FILENAME: RSA (Rivest-Shamir-Adleman)
## RSA (Rivest-Shamir-Adleman)
Body"""

    filename, content = main.parse_wiki_page(raw)

    assert filename == "RSA (Rivest-Shamir-Adleman)"
    assert content.startswith("## RSA")


def test_fix_wikilinks_corrects_aliases_and_close_matches():
    concepts = [
        "RSA (Rivest-Shamir-Adleman)",
        "Diffie-Hellman Key Exchange",
        "Kasiski Test",
    ]
    text = (
        "Use [[RSA]] for public-key systems. "
        "See also [[Diffie-Hellman]] and [[Kasisky Test]]."
    )

    fixed = main.fix_wikilinks(text, concepts)

    assert "[[RSA (Rivest-Shamir-Adleman)]]" in fixed
    assert "[[Diffie-Hellman Key Exchange]]" in fixed
    assert "[[Kasiski Test]]" in fixed


def test_find_near_duplicate_detects_normalized_and_prefix_matches():
    vault_pages = [
        "IND-CPA (Indistinguishability under Chosen Plaintext Attack)",
        "Kasiski Test",
    ]

    assert (
        main.find_near_duplicate("IND-CPA Security", vault_pages)
        == "IND-CPA (Indistinguishability under Chosen Plaintext Attack)"
    )
    assert main.find_near_duplicate("Kasisky Test", vault_pages) == "Kasiski Test"


def test_postprocess_generated_content_normalizes_setext_headings():
    raw = """Frequency Analysis
================

Overview text.

Practical Limitations
---------------------

Some detail.
"""

    fixed = main._postprocess_generated_content(raw)

    assert "## Frequency Analysis" in fixed
    assert "## Practical Limitations" in fixed
    assert "================" not in fixed
    assert "---------------------" not in fixed


def test_postprocess_generated_content_normalizes_related_section():
    raw = """## Topic

Notes.

Related Concepts
-----------------

### RSA
Public key cryptosystem.

### AES
Symmetric block cipher.
"""

    fixed = main._postprocess_generated_content(raw)

    assert "### Related" in fixed
    assert "- [[RSA]]" in fixed
    assert "- [[AES]]" in fixed


# Currently uses cryptography terms; will add more as needed
def test_find_near_duplicate_avoids_unrelated_terms():
    vault_pages = [
        "RSA Algorithm (Rivest-Shamir-Adleman)",
        "Diffie-Hellman Key Exchange",
    ]

    assert main.find_near_duplicate("AES (Advanced Encryption Standard)", vault_pages) is None


def test_detect_subject_prefers_readable_filename_without_ai_call():
    subject = main.detect_subject("Cryptography_pptx", concepts=[])
    assert subject == "Cryptography"

    subject = main.detect_subject("Unit3_Spanish_pdf", concepts=[])
    assert subject == "Spanish"

    subject = main.detect_subject("European_History_docx", concepts=[])
    assert subject == "European History"

    subject = main.detect_subject("Week_4_Calculus_docx", concepts=[])
    assert subject == "Calculus"



def test_detect_subject_uses_ai_for_generic_names(monkeypatch):
    monkeypatch.setattr(main, "query", lambda *args, **kwargs: "Network Security")

    subject = main.detect_subject("Week_3", concepts=["Firewall", "TLS"], batch_mode=True)

    assert subject == "Network Security"


def test_detect_subject_batch_falls_back_unsorted_on_inconclusive_ai(monkeypatch):
    monkeypatch.setattr(main, "query", lambda *args, **kwargs: "")

    subject = main.detect_subject("Week_3", concepts=["Firewall", "TLS"], batch_mode=True)

    assert subject == "Unsorted"


def test_chapter_summary_text_skips_empty_chunk_lists(monkeypatch):
    monkeypatch.setattr(
        main,
        "chunk_text",
        lambda text, max_chars=6000, overlap=200: [] if text == "" else [f"summary:{text}"],
    )

    summary = main._chapter_summary_text([
        {"content": "first"},
        {"content": ""},
        {"content": "second"},
    ])

    assert summary == "summary:first\n\n---\n\nsummary:second"


def test_run_cli_applies_provider_override(monkeypatch):
    called = {}

    monkeypatch.setattr(main, "set_provider", lambda value: called.setdefault("provider", value))
    monkeypatch.setattr(main, "get_provider", lambda: called.get("provider", "claude"))
    monkeypatch.setattr(main, "process_pdf", lambda *args, **kwargs: None)

    exit_code = main.run_cli([
        "sample.pdf",
        "--vault",
        "./vault",
        "--batch",
        "--provider",
        "ollama",
    ])

    assert exit_code == 0
    assert called["provider"] == "ollama"
