from pathlib import Path

import pdfwiki.main as main


INDEX_RUN_1 = """CONCEPTS:
1. Symmetric Encryption - first concept
2. Public Key Cryptography - second concept
RELATIONSHIPS:
- Symmetric Encryption -> Public Key Cryptography: related
""".strip()

INDEX_RUN_2 = """CONCEPTS:
1. Symmetric Encryption - first concept
2. Public Key Cryptography - second concept
3. Hash Function - third concept
RELATIONSHIPS:
- Public Key Cryptography -> Hash Function: extends
""".strip()

INDEX_RUN_ALT_SUBJECT = """CONCEPTS:
1. Role-Based Access Control - authorization model
2. Principle of Least Privilege - risk-reduction principle
RELATIONSHIPS:
- Principle of Least Privilege -> Role-Based Access Control: supports policy design
""".strip()

INDEX_GOLDEN = """CONCEPTS:
1. Concept Alpha - alpha concept
RELATIONSHIPS:
- Concept Alpha -> Concept Alpha: self
""".strip()


class OfflineQueryStub:
    def __init__(self, index_outputs):
        self.index_outputs = list(index_outputs)

    def __call__(self, prompt: str, system: str = "", max_tokens: int = 4096, quality: bool = False) -> str:
        if prompt.startswith("INDEX_PROMPT"):
            return self.index_outputs.pop(0)

        if prompt.startswith("WIKI_PROMPT"):
            concept = prompt.split("::", 2)[1]
            if concept == "Concept Alpha":
                return (
                    "FILENAME: Concept Alpha\n"
                    "## Concept Alpha\n"
                    "This concept explains alpha.\n"
                    "### Related\n"
                    "- [[Concept Alpha]]: self link"
                )
            return (
                f"FILENAME: {concept}\n"
                f"## {concept}\n"
                f"Definition for {concept}.\n"
                "### Related\n"
                f"- [[{concept}]]: related"
            )

        if prompt.startswith("MERGE_PROMPT"):
            return "NO_UPDATE"

        if prompt.startswith("MOC_PROMPT"):
            return "FILENAME: MOC\n## Offline MOC\nBody"

        if prompt.startswith("FLASHCARDS_PROMPT"):
            return "Q: test\nA: answer"

        if prompt.startswith("CHEATSHEET_PROMPT"):
            return "## Offline Cheat Sheet"

        raise AssertionError(f"Unexpected prompt: {prompt[:80]}")


def _mock_pipeline_dependencies(monkeypatch, index_outputs):
    monkeypatch.setattr(main, "extract_text", lambda pdf_path: "fake extracted text")
    monkeypatch.setattr(
        main,
        "split_into_chapters",
        lambda full_text: [{"title": "T", "content": "chapter content", "start_page": 1}],
    )
    monkeypatch.setattr(main, "chunk_by_page", lambda full_text, pages_per_chunk=2: ["chunk 1", "chunk 2"])
    monkeypatch.setattr(main, "chunk_text", lambda text, max_chars=6000, overlap=200: ["summary chunk"])
    monkeypatch.setattr(
        main,
        "retrieve_chunks",
        lambda chunks, concept, related_concepts=None, top_k=3, max_chars=4000: f"relevant text for {concept}",
    )

    def fake_load_prompt(name: str) -> str:
        markers = {
            "index": "INDEX_PROMPT::{text}",
            "wiki": "WIKI_PROMPT::{concept}::{text}",
            "merge": "MERGE_PROMPT::{concept}::{new_content}",
            "moc": "MOC_PROMPT::{subject}",
            "flashcards": "FLASHCARDS_PROMPT::{text}",
            "cheatsheet": "CHEATSHEET_PROMPT::{text}",
        }
        return markers[name]

    monkeypatch.setattr(main, "load_prompt", fake_load_prompt)
    monkeypatch.setattr(main, "query", OfflineQueryStub(index_outputs))


def test_incremental_second_run_adds_only_new_concepts(monkeypatch, tmp_path):
    _mock_pipeline_dependencies(monkeypatch, [INDEX_RUN_1, INDEX_RUN_2])

    vault_dir = tmp_path / "vault"
    main.process_pdf("lecture1.pdf", str(vault_dir), subject_override="Cryptography", batch_mode=True)

    a_path = vault_dir / "Cryptography" / "Symmetric Encryption.md"
    b_path = vault_dir / "Cryptography" / "Public Key Cryptography.md"
    assert a_path.exists()
    assert b_path.exists()
    before_a = a_path.read_text(encoding="utf-8")

    main.process_pdf("lecture2.pdf", str(vault_dir), subject_override="Cryptography", batch_mode=True)

    c_path = vault_dir / "Cryptography" / "Hash Function.md"
    assert c_path.exists()
    after_a = a_path.read_text(encoding="utf-8")
    assert before_a == after_a

    concept_pages = [
        p for p in (vault_dir / "Cryptography").glob("*.md")
        if "MOC" not in p.name
    ]
    assert len(concept_pages) == 3


def test_golden_output_regression_for_single_concept(monkeypatch, tmp_path):
    _mock_pipeline_dependencies(monkeypatch, [INDEX_GOLDEN])

    vault_dir = tmp_path / "vault"
    main.process_pdf("golden.pdf", str(vault_dir), subject_override="Cryptography", batch_mode=True)

    actual = (vault_dir / "Cryptography" / "Concept Alpha.md").read_text(encoding="utf-8")
    expected = (
        (Path(__file__).parent / "golden" / "concept_alpha.md")
        .read_text(encoding="utf-8")
    )

    assert actual == expected


def test_multi_pdf_processing_offline(monkeypatch, tmp_path):
    _mock_pipeline_dependencies(monkeypatch, [INDEX_RUN_1, INDEX_RUN_ALT_SUBJECT])

    vault_dir = tmp_path / "vault"
    main.process_pdf("week1.pdf", str(vault_dir), subject_override="Cryptography", batch_mode=True)
    main.process_pdf("week2.pdf", str(vault_dir), subject_override="Access Control", batch_mode=True)

    assert (vault_dir / "Cryptography" / "Symmetric Encryption.md").exists()
    assert (vault_dir / "Access Control" / "Role-Based Access Control.md").exists()
    assert (vault_dir / "Access Control" / "Access Control - MOC.md").exists()


def test_process_pdf_fails_fast_when_index_has_no_parseable_concepts(monkeypatch, tmp_path):
    _mock_pipeline_dependencies(monkeypatch, ["CONCEPTS:\nRELATIONSHIPS:"])

    vault_dir = tmp_path / "vault"

    try:
        main.process_pdf("empty.pdf", str(vault_dir), subject_override="Cryptography", batch_mode=True)
    except ValueError as error:
        assert "No concepts were parsed" in str(error)
    else:
        raise AssertionError("Expected process_pdf to fail when index parsing returns no concepts")
