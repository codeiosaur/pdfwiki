from pathlib import Path

import pdf_to_obsidian.vault as vault
import pdf_to_obsidian.writer as writer


def test_sanitize_filename_removes_invalid_characters():
    unsafe = 'RSA: Intro/Overview?*'
    assert writer.sanitize_filename(unsafe) == "RSA IntroOverview"


def test_sanitize_filename_falls_back_to_untitled_when_empty():
    assert writer.sanitize_filename('<<:/?*>') == "Untitled"


def test_write_wiki_writes_files_to_subject_directory(tmp_path):
    pages = {"RSA (Rivest-Shamir-Adleman)": "# RSA"}

    written = writer.write_wiki(str(tmp_path), pages, subject="Cryptography")

    assert len(written) == 1
    output_file = Path(written[0])
    assert output_file.exists()
    assert output_file.parent.name == "Cryptography"


def test_extract_aliases_from_frontmatter():
    content = """---
aliases: [\"RSA\", \"Rivest Shamir Adleman\"]
source: \"Cryptography\"
---
# RSA
"""

    aliases = vault._extract_aliases(content)

    assert aliases == ["RSA", "Rivest Shamir Adleman"]


def test_find_existing_page_checks_subject_and_aliases():
    state = {
        "subjects": ["Cryptography"],
        "pages": {
            "Cryptography": {
                "RSA (Rivest-Shamir-Adleman)": "/vault/Cryptography/RSA (Rivest-Shamir-Adleman).md"
            }
        },
        "aliases": {"rsa": "RSA (Rivest-Shamir-Adleman)"},
    }

    exact = vault.find_existing_page("RSA (Rivest-Shamir-Adleman)", "Cryptography", state)
    via_alias = vault.find_existing_page("RSA", "Cryptography", state)

    assert exact is not None
    assert via_alias is not None


def test_has_new_concepts_detects_when_any_is_missing():
    state = {
        "subjects": ["Cryptography"],
        "pages": {"Cryptography": {"AES": "/vault/Cryptography/AES.md"}},
        "aliases": {"aes": "AES"},
    }

    assert vault.has_new_concepts(["AES", "RSA"], "Cryptography", state) is True
    assert vault.has_new_concepts(["AES"], "Cryptography", state) is False
