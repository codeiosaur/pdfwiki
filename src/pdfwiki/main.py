"""
pdf_to_notes — main entry point
Usage: python main.py <path_to_pdf> [output_dir]

Pipeline:
  1. Extract text from PDF
  2. Split into chapters/sections
  3. Chunk each section into context-sized pieces
  4. Build concept index (Pass 1 — one API call)
  5. For each concept, retrieve relevant chunks + generate wiki page (Pass 2)
  6. Generate flashcards and cheat sheet from full text
"""

import re
import argparse
from pathlib import Path
from difflib import get_close_matches

from pdfwiki.extractor import extract_text, split_into_chapters, chunk_text, chunk_by_page
from pdfwiki.retriever import retrieve_chunks
from pdfwiki.ai_client import query
from pdfwiki.writer import write_wiki, write_flashcards, write_cheatsheet
from pdfwiki.vault import load_vault_state, find_existing_page


PROMPTS_DIR = Path(__file__).parent / "prompts"

def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


# --- Parsers ---

def parse_index(raw: str) -> tuple[list[str], str]:
    """
    Extract concept names from index output.
    Returns (concept_list, raw_index_text) — raw text preserved for pass 2 context.
    """
    concepts = []
    seen = set()
    in_concepts = False

    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("CONCEPTS:"):
            in_concepts = True
            continue
        if line.startswith("RELATIONSHIPS:"):
            break
        if not in_concepts or not line:
            continue

        match = re.match(r'^(?:\d+\.|[-*])\s+(.+?)(?:\s+[—–-]\s+.+)?$', line)
        if match:
            concept = match.group(1).strip()
            if concept not in seen:
                concepts.append(concept)
                seen.add(concept)

    return concepts, raw


def parse_wiki_page(raw: str) -> tuple[str, str]:
    """Parse a single wiki page response into (filename, content)."""
    match = re.search(r'^\*{0,2}FILENAME:\*{0,2}\s*(.+?)\*{0,2}$', raw, re.MULTILINE)
    if match:
        filename = match.group(1).strip()
        content = raw[match.end():].strip()
        return filename, content

    heading = re.search(r'^##\s+(.+)$', raw, re.MULTILINE)
    if heading:
        return heading.group(1).strip(), raw

    return "Untitled", raw


# --- Post-processing ---

def build_alias_map(concepts: list[str]) -> dict[str, str]:
    """
    Build a map of likely aliases → canonical concept names.
    e.g. "RSA" → "RSA (Rivest-Shamir-Adleman)"
         "Diffie-Hellman" → "Diffie-Hellman Key Exchange"
    """
    alias_map = {}
    for concept in concepts:
        # The concept itself maps to itself
        alias_map[concept.lower()] = concept
        # Parenthetical: "RSA (Rivest-Shamir-Adleman)" → also map "RSA"
        # Also captures abbreviations: "One-Time Pad (OTP)" → also map "OTP"
        paren_match = re.match(r'^(.+?)\s*\((.+?)\)', concept)
        if paren_match:
            short = paren_match.group(1).strip()
            abbrev = paren_match.group(2).strip()
            alias_map[short.lower()] = concept
            alias_map[abbrev.lower()] = concept
        # Last word stripped: "Diffie-Hellman Key Exchange" → "Diffie-Hellman"
        words = concept.split()
        if len(words) > 1:
            alias_map[words[0].lower()] = concept
    return alias_map


def fix_wikilinks(content: str, concepts: list[str],
                  vault_pages: list[str] | None = None) -> str:
    """
    Find all [[wikilinks]] in a page and correct any that don't match
    a known concept name, using fuzzy matching as fallback.

    concepts: names from the current run's index
    vault_pages: existing page names from the vault (for cross-run matching)
    """
    # Combine current concepts + existing vault pages as valid link targets
    all_known = list(dict.fromkeys(concepts + (vault_pages or [])))
    alias_map = build_alias_map(all_known)
    concept_set = set(all_known)

    def fix_link(match):
        link_text = match.group(1).strip()
        # Already correct
        if link_text in concept_set:
            return f'[[{link_text}]]'
        # Check alias map (case-insensitive)
        canonical = alias_map.get(link_text.lower())
        if canonical:
            return f'[[{canonical}]]'
        # Fuzzy match against all known names — catches cross-run spelling variants
        # Higher cutoff (0.82) to avoid false positives on short concept names
        close = get_close_matches(link_text, all_known, n=1, cutoff=0.82)
        if close:
            return f'[[{close[0]}]]'
        # No match found — leave as-is
        return match.group(0)

    fixed = re.sub(r'\[\[([^\]]+)\]\]', fix_link, content)
    return fixed


def _normalize_concept(name: str) -> str:
    """Strip parentheticals and normalize for fuzzy comparison.
    'IND-CPA (Indistinguishability...)' → 'ind-cpa'
    'RSA (Rivest-Shamir-Adleman)' → 'rsa'
    """
    return re.sub(r'\s*\([^)]+\)', '', name).strip().lower()


def find_near_duplicate(concept: str, vault_pages: list[str],
                        cutoff: float = 0.82) -> str | None:
    """
    Check if a concept name is a near-duplicate of an existing vault page.
    Uses two-pass matching:
    1. Raw string similarity (catches 'Kasiski' vs 'Kasisky')
    2. Normalized similarity — strips parentheticals before comparing
       (catches 'IND-CPA Security' vs 'IND-CPA (Indistinguishability...)')
    Returns the existing page name if found, None otherwise.
    """
    # Pass 1: raw match
    close = get_close_matches(concept, vault_pages, n=1, cutoff=cutoff)
    if close:
        return close[0]

    # Pass 2: normalized match with lower cutoff
    norm_concept = _normalize_concept(concept)
    norm_pages = {_normalize_concept(p): p for p in vault_pages}
    close_norm = get_close_matches(norm_concept, list(norm_pages.keys()),
                                   n=1, cutoff=0.75)
    if close_norm:
        return norm_pages[close_norm[0]]

    # Pass 3: prefix match — catches 'IND-CPA Security' vs 'IND-CPA (...)'
    # If the normalized concept is a prefix of a normalized vault page
    # (or vice versa) and the overlap is >= 6 chars, treat as duplicate
    for norm_page, original_page in norm_pages.items():
        shorter = min(norm_concept, norm_page, key=len)
        longer  = max(norm_concept, norm_page, key=len)
        if len(shorter) >= 6 and longer.startswith(shorter):
            return original_page

    return None


def add_frontmatter(filename: str, content: str, concepts: list[str],
                    subject: str = "", tags: list[str] | None = None) -> str:
    """
    Add Obsidian YAML frontmatter with:
    - aliases: short forms so [[RSA]] resolves to full page name
    - tags: subject domain + auto-detected topic tags
    - subject: source PDF name
    - dataview fields: enables TABLE queries across the vault
    """
    aliases = []
    paren_match = re.match(r'^(.+?)\s*\(', filename)
    if paren_match:
        short = paren_match.group(1).strip()
        aliases.append(short)
        # Also grab abbreviation inside parens
        abbrev_match = re.match(r'^.+?\((.+?)\)', filename)
        if abbrev_match:
            aliases.append(abbrev_match.group(1).strip())
    words = filename.split()
    if len(words) > 2:
        aliases.append(words[0])
        aliases.append(" ".join(words[:2]))
    # Deduplicate, remove if same as filename
    aliases = list(dict.fromkeys(a for a in aliases if a != filename))

    # Auto-detect difficulty from content keywords
    content_lower = content.lower()
    if any(w in content_lower for w in ["euler", "modular", "discrete log", "polynomial", "complexity"]):
        difficulty = "advanced"
    elif any(w in content_lower for w in ["algorithm", "protocol", "attack", "encrypt"]):
        difficulty = "intermediate"
    else:
        difficulty = "foundational"

    # Build tag list — deduplicate
    all_tags = list(tags or [])
    subject_tag = subject.lower().replace(" ", "-") if subject else ""
    if subject_tag and subject_tag not in all_tags:
        all_tags.append(subject_tag)
    difficulty_tag = f"difficulty/{difficulty}"
    if difficulty_tag not in all_tags:
        all_tags.append(difficulty_tag)

    # Compose frontmatter
    lines = ["---"]
    if aliases:
        alias_yaml = ", ".join(f'"{a}"' for a in aliases)
        lines.append(f"aliases: [{alias_yaml}]")
    if subject:
        lines.append(f'source: "{subject}"')
    lines.append(f"difficulty: {difficulty}")
    if all_tags:
        tag_yaml = ", ".join(f'"{t}"' for t in all_tags)
        lines.append(f"tags: [{tag_yaml}]")
    lines.append("---")
    lines.append("")

    return "\n".join(lines) + "\n" + content


# --- Subject detection ---

# Filenames that tell us nothing about the subject
GENERIC_NAMES = {
    "lecture", "notes", "slides", "week", "chapter", "unit",
    "class", "module", "part", "doc", "file", "reading", "handout",
    "worksheet", "assignment", "homework", "lab", "tutorial"
}


def _is_readable_word(s: str) -> bool:
    """
    Heuristic: does this string look like a real word/phrase?
    Rejects garbage like 'asdfoiuasydf' or UUIDs.
    Logic: real words have vowels, and consonant runs rarely exceed 4.
    """
    s = s.lower().replace(" ", "")
    if not s:
        return False
    vowels = sum(1 for c in s if c in "aeiou")
    # Too few vowels relative to length → likely garbage
    if len(s) > 4 and vowels / len(s) < 0.15:
        return False
    # UUID/hash pattern: hex chars with no vowels
    if re.match(r'^[0-9a-f]{8,}$', s):
        return False
    # Max consonant run > 5 → likely garbage
    consonant_run = max(
        (len(m.group()) for m in re.finditer(r'[bcdfghjklmnpqrstvwxyz]+', s)),
        default=0
    )
    if consonant_run > 5:
        return False
    return True


def detect_subject(raw_stem: str, concepts: list[str] | None = None,
                   batch_mode: bool = False) -> str:
    """
    Hybrid subject detection:
    1. Clean the filename — if result is meaningful and readable, use it
    2. If filename is generic or unreadable, infer from concept list via AI
    3. If AI is uncertain:
       - Interactive mode: prompt the user
       - Batch mode: fall back to "Unsorted" with a warning (never blocks)
    """
    # Step 1: clean filename
    cleaned = re.sub(
        r'[_-](pptx|pdf|optimized|compressed|final|v\d+)$',
        '', raw_stem, flags=re.IGNORECASE
    ).replace('_', ' ').replace('-', ' ').strip()

    first_word = cleaned.split()[0].lower() if cleaned.split() else ""
    is_generic = (
        first_word in GENERIC_NAMES
        or cleaned.lower() in GENERIC_NAMES
        or re.match(r'^\d+$', cleaned)
        or re.match(r'^week\s*\d+', cleaned, re.IGNORECASE)
        or len(cleaned) <= 2
        or not _is_readable_word(cleaned)   # catches garbage filenames
    )

    if not is_generic:
        print(f"  Subject detected from filename: \"{cleaned}\"")
        return cleaned

    # Step 2: AI inference from concept list
    if concepts:
        reason = "unreadable filename" if not _is_readable_word(cleaned) else "generic filename"
        print(f"  {reason.capitalize()} — inferring subject from content...")
        concept_sample = ", ".join(concepts[:15])
        prompt = (
            f"Given these academic concepts: {concept_sample}\n\n"
            f"What is the most likely university course subject or topic area "
            f"these concepts belong to? Reply with ONLY the subject name, "
            f"2-4 words maximum. Examples: \"Cryptography\", \"Operating Systems\", "
            f"\"Network Security\", \"Linear Algebra\""
        )
        inferred = query(prompt, max_tokens=20).strip().strip('"\'\n')
        if inferred and len(inferred.split()) <= 5 and '.' not in inferred:
            print(f"  Subject inferred from content: \"{inferred}\"")
            return inferred
        print("  AI inference inconclusive.")

    # Step 3: prompt or fallback
    if batch_mode:
        print(f"  WARNING: could not determine subject for '{raw_stem}'.")
        print(f"  Files will be written to \'Unsorted/' — review and move manually.")
        return "Unsorted"

    print("  Could not auto-detect subject.")
    user_input = input("  Enter subject name for this PDF (e.g. \'Cryptography\'): ").strip()
    return user_input or cleaned or "Unsorted"


def _chapter_summary_text(chapters: list[dict]) -> str:
    summaries = []
    for chapter in chapters:
        chunks = chunk_text(chapter["content"], max_chars=2000)
        if chunks:
            summaries.append(chunks[0])
    return "\n\n---\n\n".join(summaries)


# --- Main pipeline ---

def process_pdf(pdf_path: str, output_dir: str, subject_override: str = "", batch_mode: bool = False):
    raw_stem = Path(pdf_path).stem
    print(f"\n{'='*50}")
    print(f"Processing: {raw_stem}")
    print(f"{'='*50}")

    # 1. Extract text
    print("\n[1/6] Extracting text from PDF...")
    full_text = extract_text(pdf_path)

    # 2. Split into chapters and chunk each one
    print("\n[2/6] Splitting into chapters and chunking...")
    chapters = split_into_chapters(full_text)
    # Use page-based chunking for slide PDFs — each page is a natural topic boundary
    # Falls back gracefully on textbooks (pages will just be paragraphs)
    all_chunks = chunk_by_page(full_text, pages_per_chunk=2)
    print(f"  Total chunks: {len(all_chunks)}")

    # For flashcards/cheatsheet: compressed summary (first chunk per chapter)
    summary_text = _chapter_summary_text(chapters)

    # For index: full text so no concepts get dropped
    # One API call, so cost is low even with more tokens
    index_text_input = "\n\n---\n\n".join(ch["content"] for ch in chapters)

    # 3. Build concept index — Pass 1 (one API call on full text)
    print("\n[3/6] Building concept index (Pass 1)...")
    index_prompt = load_prompt("index").replace("{text}", index_text_input)
    index_raw = query(index_prompt, max_tokens=4000)  # Haiku
    concepts, index_text = parse_index(index_raw)
    if not concepts:
        raise ValueError(
            "No concepts were parsed from the index response. "
            "Check the prompt/output format before continuing."
        )
    print(f"  Found {len(concepts)} concepts: {', '.join(concepts)}")

    # Detect subject — uses filename first, AI inference second, prompt third
    if subject_override:
        subject = subject_override
        print(f"  Subject (from --subject flag): \"{subject}\"")
    else:
        subject = detect_subject(raw_stem, concepts, batch_mode=batch_mode)

    # 4. Generate wiki pages — Pass 2 (incremental: new / merge / skip)
    print(f"\n[4/6] Processing {len(concepts)} concepts (Pass 2)...")
    vault_state = load_vault_state(output_dir)
    # Flat list of all existing page names across all subjects — used for
    # cross-run fuzzy matching (catches "Kasiski" vs "Kasisky" etc.)
    all_vault_pages = [
        page
        for pages in vault_state["pages"].values()
        for page in pages.keys()
    ]
    wiki_pages = {}         # new pages to write
    merged_pages = {}       # existing pages that were updated
    skipped = []            # concepts with no new content
    concept_names = "\n".join(f"- {c}" for c in concepts)
    wiki_prompt_template = load_prompt("wiki")
    merge_prompt_template = load_prompt("merge")

    for i, concept in enumerate(concepts):
        other_concepts = [c for c in concepts if c != concept]
        relevant_text = retrieve_chunks(
            all_chunks,
            concept=concept,
            related_concepts=other_concepts[:5],
            top_k=2,       # reduced from 3 — diminishing returns beyond 2 chunks
            max_chars=4000 # reduced from 8000 — sweet spot for quality vs cost
        )

        existing_path = find_existing_page(concept, subject, vault_state)

        # Cross-run near-duplicate check — catches spelling variants like
        # "Kasiski Test" matching existing "Kasisky Test"
        if existing_path is None and all_vault_pages:
            near_dup = find_near_duplicate(concept, all_vault_pages)
            if near_dup:
                # Treat as a merge with the near-duplicate page
                subject_pages = vault_state["pages"].get(subject, {})
                if near_dup in subject_pages:
                    existing_path = subject_pages[near_dup]
                    print(f"  [{i+1}/{len(concepts)}] NEAR-DUP: "
                          f"{concept} ≈ {near_dup}")

        if existing_path is None:
            # NEW PAGE — generate from scratch
            print(f"  [{i+1}/{len(concepts)}] NEW: {concept} "
                  f"({len(relevant_text):,} chars)...")
            # Send concept names only as index — relationships add tokens without
            # improving wikilink accuracy (concept_names already covers that)
            page_prompt = (wiki_prompt_template
                           .replace("{concept}", concept)
                           .replace("{index}", concept_names)
                           .replace("{concept_names}", concept_names)
                           .replace("{text}", relevant_text))
            page_raw = query(page_prompt, quality=True, max_tokens=1500)
            filename, page_content = parse_wiki_page(page_raw)
            page_content = fix_wikilinks(page_content, concepts,
                                         vault_pages=all_vault_pages)
            page_content = add_frontmatter(filename, page_content, concepts,
                                           subject=subject)
            wiki_pages[filename] = page_content

        else:
            # EXISTING PAGE — diff-aware merge
            print(f"  [{i+1}/{len(concepts)}] MERGE: {concept} "
                  f"(existing: {Path(existing_path).name})...")
            existing_content = Path(existing_path).read_text(encoding="utf-8")
            merge_prompt = (merge_prompt_template
                            .replace("{existing_content}", existing_content)
                            .replace("{concept}", concept)
                            .replace("{new_content}", relevant_text)
                            .replace("{source}", subject)
                            .replace("{concept_names}", concept_names))
            merge_raw = query(merge_prompt, quality=True, max_tokens=800)

            if merge_raw.strip() == "NO_UPDATE":
                print(f"    → No new content, skipping")
                skipped.append(concept)
            else:
                merge_raw = fix_wikilinks(merge_raw, concepts,
                                          vault_pages=all_vault_pages)
                merged_pages[Path(existing_path).stem] = (existing_path, merge_raw)
                print(f"    → Merged new content")

    # Write new pages
    if wiki_pages:
        write_wiki(output_dir, wiki_pages, subject=subject)

    # Write merged pages (overwrite existing files with updated content)
    for stem, (path, content_) in merged_pages.items():
        Path(path).write_text(content_, encoding="utf-8")
        print(f"  Updated: {Path(path).name}")

    # Summary
    print(f"\n  Summary: {len(wiki_pages)} new, "
          f"{len(merged_pages)} merged, {len(skipped)} skipped")
    added_new = len(wiki_pages) > 0

    # 5. Generate MOC page (only if new concepts were added)
    if added_new:
        print("\n[5/6] Regenerating Map of Contents (new concepts added)...")
        # Collect ALL concepts in this subject folder for a complete MOC
        all_subject_pages = list(load_vault_state(output_dir)["pages"]
                                 .get(subject, {}).keys())
        all_concepts_for_moc = list(dict.fromkeys(concepts + all_subject_pages))

        concept_list = "\n".join(f"- {c}" for c in all_concepts_for_moc)
        relationships = ""
        if "RELATIONSHIPS:" in index_text:
            relationships = index_text.split("RELATIONSHIPS:")[1].strip()

        moc_prompt = (load_prompt("moc")
                      .replace("{subject}", subject)
                      .replace("{concept_list}", concept_list)
                      .replace("{relationships}", relationships[:2000]))

        moc_raw = query(moc_prompt, quality=False, max_tokens=2000)  # Haiku
        _, moc_content = parse_wiki_page(moc_raw)
        moc_content = add_frontmatter(subject, moc_content, all_concepts_for_moc,
                                      subject=subject,
                                      tags=["moc", subject.lower().replace(" ", "-")])
        safe_subject = re.sub(r'[<>:"/\\|?*]', '', subject).strip()
        moc_path = Path(output_dir) / safe_subject / f"{safe_subject} - MOC.md"
        moc_path.parent.mkdir(parents=True, exist_ok=True)
        moc_path.write_text(moc_content, encoding="utf-8")
        print(f"  Written: {moc_path.name}")
    else:
        print("\n[5/6] Skipping MOC regeneration (no new concepts added)")

    # 6. Flashcards + cheatsheet on summary text
    print("\n[6/6] Generating flashcards and cheat sheet...")
    cards_raw = query(load_prompt("flashcards").replace("{text}", summary_text))  # Haiku
    write_flashcards(output_dir, subject, cards_raw)

    sheet_raw = query(load_prompt("cheatsheet").replace("{text}", summary_text))  # Haiku
    write_cheatsheet(output_dir, subject, sheet_raw)

    print(f"\nDone! Output written to: {output_dir}/")
    print(f"Wiki pages: {len(wiki_pages)}")


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Obsidian wiki, flashcards, and cheatsheet from PDFs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single PDF, auto-detect subject
  python main.py Cryptography_pptx.pdf --vault ./vault

  # Single PDF, override subject
  python main.py Lecture_3.pdf --vault ./vault --subject Cryptography

  # Multiple PDFs (auto batch mode)
  python main.py Cryptography.pdf AccessControl.pdf --vault ./vault

  # Glob (shell expands *.pdf before passing to argparse)
  python main.py *.pdf --vault ./vault
        """
    )
    parser.add_argument("pdfs", nargs="+", help="One or more PDF files to process")
    parser.add_argument("--vault", "-v", default=None,
                        help="Obsidian vault root. Defaults to a 'vault/' folder "
                             "next to the first PDF if not specified.")
    parser.add_argument("--subject", "-s", default=None,
                        help="Subject override. Single PDF only.")
    parser.add_argument("--batch", action="store_true",
                        help="Never prompt. Unknown subjects go to Unsorted/.")
    args = parser.parse_args(argv)

    if args.subject and len(args.pdfs) > 1:
        parser.error("--subject can only be used with a single PDF")

    # Default vault: sibling folder next to the first PDF
    # e.g. ~/Documents/Cryptography.pdf → ~/Documents/vault/
    vault = args.vault or str(Path(args.pdfs[0]).parent / "vault")
    if not args.vault:
        print(f"No --vault specified. Using: {vault}")

    batch = args.batch or len(args.pdfs) > 1
    failed = []

    for pdf_path in args.pdfs:
        try:
            process_pdf(pdf_path, vault,
                        subject_override=args.subject if len(args.pdfs) == 1 else "",
                        batch_mode=batch)
        except Exception as e:
            print(f"\nERROR processing {pdf_path}: {e}")
            failed.append((pdf_path, str(e)))

    if failed:
        print(f"\n{'='*50}")
        print(f"FAILED ({len(failed)}/{len(args.pdfs)} PDFs):")
        for path, err in failed:
            print(f"  {path}: {err}")
        return 1
    elif len(args.pdfs) > 1:
        print(f"\n{'='*50}")
        print(f"All {len(args.pdfs)} PDFs processed successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())