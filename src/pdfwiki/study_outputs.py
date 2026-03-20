"""Study output generation services (MOC, flashcards, cheatsheet)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import re
import hashlib


SOURCE_HASH_RE = re.compile(r"<!--\s*study_source_hash:\s*([0-9a-f]{12,64})\s*-->")


def _study_source_hash(subject: str, summary_text: str) -> str:
    payload = f"{subject}\n{summary_text}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def _extract_existing_study_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8")
    match = SOURCE_HASH_RE.search(content)
    if match:
        return match.group(1)
    return None


def _with_study_hash_marker(content: str, source_hash: str) -> str:
    marker = f"<!-- study_source_hash: {source_hash} -->"
    if SOURCE_HASH_RE.search(content):
        return SOURCE_HASH_RE.sub(marker, content, count=1)
    return marker + "\n\n" + content


@dataclass(frozen=True)
class StudyOutputDeps:
    load_vault_state: Callable[[str], dict]
    load_prompt: Callable[[str], str]
    query: Callable[..., str]
    parse_wiki_page: Callable[[str], tuple[str, str]]
    add_frontmatter: Callable[..., str]
    write_flashcards: Callable[[str, str, str], str]
    write_cheatsheet: Callable[[str, str, str], str]


def maybe_regenerate_moc(
    *,
    added_new: bool,
    output_dir: str,
    subject: str,
    concepts: list[str],
    index_text: str,
    deps: StudyOutputDeps,
) -> None:
    """Regenerate subject MOC only when new concepts were created in this run."""
    if not added_new:
        print("\n[5/6] Skipping MOC regeneration (no new concepts added)")
        return

    print("\n[5/6] Regenerating Map of Contents (new concepts added)...")
    all_subject_pages = list(deps.load_vault_state(output_dir)["pages"].get(subject, {}).keys())
    all_concepts_for_moc = list(dict.fromkeys(concepts + all_subject_pages))

    concept_list = "\n".join(f"- {c}" for c in all_concepts_for_moc)
    relationships = ""
    if "RELATIONSHIPS:" in index_text:
        relationships = index_text.split("RELATIONSHIPS:")[1].strip()

    moc_prompt = (
        deps.load_prompt("moc")
        .replace("{subject}", subject)
        .replace("{concept_list}", concept_list)
        .replace("{relationships}", relationships[:2000])
    )

    moc_raw = deps.query(moc_prompt, task="cheap", max_tokens=2000)
    _, moc_content = deps.parse_wiki_page(moc_raw)
    moc_content = deps.add_frontmatter(
        subject,
        moc_content,
        all_concepts_for_moc,
        subject=subject,
        tags=["moc", subject.lower().replace(" ", "-")],
    )

    safe_subject = re.sub(r'[<>:"/\\|?*]', '', subject).strip()
    moc_path = Path(output_dir) / safe_subject / f"{safe_subject} - MOC.md"
    moc_path.parent.mkdir(parents=True, exist_ok=True)
    moc_path.write_text(moc_content, encoding="utf-8")
    print(f"  Written: {moc_path.name}")


def generate_study_aids(
    *,
    output_dir: str,
    subject: str,
    summary_text: str,
    flashcards_max_tokens: int = 1200,
    cheatsheet_max_tokens: int = 1200,
    summary_max_chars: int = 10000,
    deps: StudyOutputDeps,
) -> None:
    """Generate flashcards and cheatsheet artifacts from summary text."""
    print("\n[6/6] Generating flashcards and cheat sheet...")

    trimmed_summary = summary_text[:summary_max_chars]
    source_hash = _study_source_hash(subject, trimmed_summary)
    safe_subject = re.sub(r'[<>:"/\\|?*]', '', subject).strip() or "Untitled"
    cards_path = Path(output_dir) / "flashcards" / f"{safe_subject}_flashcards.md"
    sheet_path = Path(output_dir) / "cheatsheets" / f"{safe_subject}_cheatsheet.md"

    existing_cards_hash = _extract_existing_study_hash(cards_path)
    existing_sheet_hash = _extract_existing_study_hash(sheet_path)
    if existing_cards_hash == source_hash and existing_sheet_hash == source_hash:
        print("  Skipped: flashcards + cheatsheet unchanged (source hash match)")
        return

    if existing_cards_hash != source_hash:
        cards_raw = deps.query(
            deps.load_prompt("flashcards").replace("{text}", trimmed_summary),
            task="cheap",
            max_tokens=flashcards_max_tokens,
        )
        cards_raw = _with_study_hash_marker(cards_raw, source_hash)
        deps.write_flashcards(output_dir, subject, cards_raw)
    else:
        print("  Skipped: flashcards unchanged")

    if existing_sheet_hash != source_hash:
        sheet_raw = deps.query(
            deps.load_prompt("cheatsheet").replace("{text}", trimmed_summary),
            task="cheap",
            max_tokens=cheatsheet_max_tokens,
        )
        sheet_raw = _with_study_hash_marker(sheet_raw, source_hash)
        deps.write_cheatsheet(output_dir, subject, sheet_raw)
    else:
        print("  Skipped: cheatsheet unchanged")
