"""
Output writer module.
Handles writing wiki pages, flashcards, and cheat sheets to disk.
"""

from pathlib import Path
import re


def sanitize_filename(name: str) -> str:
    """Remove characters that are invalid in filenames."""
    cleaned = re.sub(r'[<>:"/\\|?*]', "", name).strip().rstrip('.')
    return cleaned or "Untitled"


def write_wiki(output_dir: str, pages: dict[str, str],
               subject: str = "") -> list[str]:
    """
    Write Obsidian wiki markdown files into a subject subfolder.
    pages: dict of {filename: markdown_content}
    subject: subfolder name within the vault (e.g. "Cryptography")
    Returns list of files written.
    """
    # Pages go into vault/Subject/ — no extra "wiki" subfolder
    wiki_dir = Path(output_dir) / subject if subject else Path(output_dir)
    wiki_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for filename, content in pages.items():
        safe_name = sanitize_filename(filename)
        filepath = wiki_dir / f"{safe_name}.md"
        filepath.write_text(content, encoding="utf-8")
        written.append(str(filepath))
        print(f"  Written: {filepath.name}")

    return written


def write_flashcards(output_dir: str, subject: str, content: str) -> str:
    """
    Write flashcards as a markdown file (Anki-importable format).
    Returns path of file written.
    """
    cards_dir = Path(output_dir) / "flashcards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    safe_name = sanitize_filename(subject)
    filepath = cards_dir / f"{safe_name}_flashcards.md"
    filepath.write_text(content, encoding="utf-8")
    print(f"  Written: {filepath.name}")
    return str(filepath)


def write_cheatsheet(output_dir: str, subject: str, content: str) -> str:
    """
    Write cheat sheet as a markdown file.
    Returns path of file written.
    """
    sheets_dir = Path(output_dir) / "cheatsheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    safe_name = sanitize_filename(subject)
    filepath = sheets_dir / f"{safe_name}_cheatsheet.md"
    filepath.write_text(content, encoding="utf-8")
    print(f"  Written: {filepath.name}")
    return str(filepath)