from pathlib import Path

import pdf_to_obsidian.main as main


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


def test_detect_subject_prefers_readable_filename_without_ai_call():
    subject = main.detect_subject("Cryptography_pptx", concepts=[])
    assert subject == "Cryptography"


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
