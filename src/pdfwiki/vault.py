"""
Vault state management.
Reads the existing Obsidian vault to understand what's already been generated,
enabling incremental updates without duplicating content.
"""

import re
from pathlib import Path


def load_vault_state(vault_dir: str) -> dict:
    """
    Scan an existing vault and return its current state.

    Returns:
    {
        "subjects": ["Cryptography", "Access Control", ...],
        "pages": {
            "Cryptography": {
                "RSA (Rivest-Shamir-Adleman)": "/path/to/file.md",
                "Vigenere Cipher": "/path/to/file.md",
                ...
            },
            ...
        },
        "aliases": {
            "rsa": "RSA (Rivest-Shamir-Adleman)",   # lowercase → canonical name
            "otp": "One-Time Pad (OTP)",
            ...
        }
    }
    """
    vault_path = Path(vault_dir)
    state = {"subjects": [], "pages": {}, "aliases": {}}

    if not vault_path.exists():
        return state

    for subject_dir in sorted(vault_path.iterdir()):
        if not subject_dir.is_dir():
            continue
        # Skip hidden dirs and output dirs
        if subject_dir.name.startswith('.'):
            continue
        if subject_dir.name in ("flashcards", "cheatsheets", "Unsorted"):
            continue

        subject = subject_dir.name
        state["subjects"].append(subject)
        state["pages"][subject] = {}

        for md_file in subject_dir.glob("*.md"):
            # Skip MOC/TOC files
            if "MOC" in md_file.stem or "TOC" in md_file.stem:
                continue

            page_name = md_file.stem
            state["pages"][subject][page_name] = str(md_file)

            # Extract aliases from frontmatter
            content = md_file.read_text(encoding="utf-8")
            aliases = _extract_aliases(content)
            for alias in aliases:
                state["aliases"][alias.lower()] = page_name

            # Also register the page name itself as an alias
            state["aliases"][page_name.lower()] = page_name

    return state


def _extract_aliases(content: str) -> list[str]:
    """Extract alias list from YAML frontmatter."""
    aliases = []
    fm_match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if not fm_match:
        return aliases

    fm = fm_match.group(1)
    alias_match = re.search(r'aliases:\s*\[([^\]]+)\]', fm)
    if alias_match:
        raw = alias_match.group(1)
        aliases = [a.strip().strip('"\'') for a in raw.split(',')]

    return aliases


def find_existing_page(concept: str, subject: str,
                       vault_state: dict) -> str | None:
    """
    Check if a concept already has a page in the vault.
    Checks the subject folder first, then aliases across all subjects.

    Returns the file path if found, None otherwise.
    """
    # Check exact match in subject folder
    subject_pages = vault_state["pages"].get(subject, {})
    if concept in subject_pages:
        return subject_pages[concept]

    # Check aliases in subject folder
    canonical = vault_state["aliases"].get(concept.lower())
    if canonical and canonical in subject_pages:
        return subject_pages[canonical]

    # Check other subjects (concept might exist in a related domain)
    for other_subject, pages in vault_state["pages"].items():
        if other_subject == subject:
            continue
        if concept in pages:
            return pages[concept]
        canonical = vault_state["aliases"].get(concept.lower())
        if canonical and canonical in pages:
            return pages[canonical]

    return None


def has_new_concepts(new_concepts: list[str], subject: str,
                     vault_state: dict) -> bool:
    """
    Return True if any concepts in new_concepts don't have existing pages.
    Used to decide whether to regenerate the MOC.
    """
    for concept in new_concepts:
        if find_existing_page(concept, subject, vault_state) is None:
            return True
    return False