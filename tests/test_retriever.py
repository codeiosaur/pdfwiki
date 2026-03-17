import pdfwiki.retriever as retriever


def test_keywords_from_concept_extracts_words_abbreviations_and_phrase():
    keywords = retriever._keywords_from_concept("One-Time Pad (OTP)")

    assert "otp" in keywords
    assert "pad" in keywords
    assert "one-time pad" in keywords


def test_retrieve_chunks_returns_top_scored_chunks_with_separator():
    chunks = [
        "Symmetric encryption uses shared keys.",
        "RSA is a public key cryptosystem based on factoring.",
        "Diffie-Hellman supports key exchange.",
    ]

    selected = retriever.retrieve_chunks(
        chunks,
        concept="RSA",
        related_concepts=["Public Key Cryptography"],
        top_k=2,
        max_chars=2000,
    )

    assert "RSA is a public key cryptosystem" in selected
    assert "---" in selected


def test_retrieve_chunks_falls_back_when_no_scores():
    chunks = [
        "Topic A has no overlap.",
        "Topic B has no overlap.",
    ]

    selected = retriever.retrieve_chunks(chunks, concept="Quantum", top_k=2)

    assert selected == chunks[0]
