"""
pdf_to_notes — main entry point
Usage: python main.py <path_to_pdf> [output_dir]

Pipeline:
  1. Extract text from PDF
  2. Split into chapters/sections
  3. Chunk each section into context-sized pieces
  4. Build concept index (Pass 1 — one API call)
    5. For each concept, retrieve relevant chunks -> extract facts -> generate wiki page (Pass 2)
  6. Generate flashcards and cheat sheet from full text
"""

import re
import os
import argparse
import sys
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from difflib import get_close_matches, SequenceMatcher
import time

# Allow running this file directly: `python src/pdfwiki/main.py ...`
# by ensuring `src/` is on sys.path for `import pdfwiki...`.
if __package__ in (None, ""):
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from pdfwiki.extractor import extract_text, split_into_chapters, chunk_text, chunk_by_page, smart_chunk
from pdfwiki.retriever import retrieve_chunks, retrieve_ranked_chunks, retrieve_ranked_chunks_with_scores, limit_context, find_related_concepts, build_concept_graph, _compute_adaptive_context_size
from pdfwiki.ai_client import query, extract_facts, set_provider, get_provider
from pdfwiki.writer import write_wiki, write_flashcards, write_cheatsheet
from pdfwiki.vault import load_vault_state, find_existing_page


PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
if not PROMPTS_DIR.exists():
    PROMPTS_DIR = Path(__file__).parent / "prompts"

# --- Precompiled regex patterns for performance (avoid recompilation) ---
REGEX_BULLET = re.compile(r'^\s*(?:[-*•]|\d+[.)])\s+')
REGEX_FILENAME = re.compile(r'^\*{0,2}FILENAME:\*{0,2}\s*(.+?)\*{0,2}$', re.MULTILINE)
REGEX_HEADING = re.compile(r'^##\s+(.+)$', re.MULTILINE)
REGEX_WIKILINK = re.compile(r'\[\[([^\]]+)\]\]')
REGEX_CONCEPTS_HEADING = re.compile(r'^(concepts?|key concepts?|topics?)\s*:?.*$', re.IGNORECASE)
REGEX_RELATIONSHIPS_HEADING = re.compile(r'^(relationships?|links?)\s*:?.*$', re.IGNORECASE)
REGEX_PAREN = re.compile(r'^(.+?)\s*\((.+?)\)')
REGEX_VERB = re.compile(
    r'\b(depends on|is|are|means|refers to|involves|uses|requires|allows|provides|ensures)\b',
    re.IGNORECASE,
)
REGEX_MARKDOWN_LINK = re.compile(r'\[([^\]]+)\]\([^\)]+\)')
REGEX_SOURCE_HASH = re.compile(r'<!--\s*source_context_hash:\s*([0-9a-f]{12,40})\s*-->')


RUN_PROFILE_ALIASES = {
    "balanced": "hybrid",
}

RUN_PROFILE_SETTINGS: dict[str, dict[str, int]] = {
    # Fastest local iteration for most machines.
    "speed": {
        "context_max_chars": 2200,
        "extract_max_tokens": 280,
        "write_max_tokens": 800,
        "merge_max_tokens": 500,
        "retrieve_top_k": 1,
        "default_max_workers": 4,
    },
    # Default tradeoff profile.
    "hybrid": {
        "context_max_chars": 3000,
        "extract_max_tokens": 360,
        "write_max_tokens": 1000,
        "merge_max_tokens": 700,
        "retrieve_top_k": 2,
        "default_max_workers": 3,
    },
    # Highest quality, slower throughput.
    "quality": {
        "context_max_chars": 4800,
        "extract_max_tokens": 550,
        "write_max_tokens": 1400,
        "merge_max_tokens": 900,
        "retrieve_top_k": 3,
        "default_max_workers": 2,
    },
}

def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


# --- Parsers ---

def parse_index(raw: str) -> tuple[list[str], str]:
    """
    Extract concept names from index output.
    Returns (concept_list, raw_index_text) — raw text preserved for pass 2 context.
    """
    concepts: list[str] = []
    seen: set[str] = set()

    def add_concept(candidate: str) -> None:
        concept = _clean_concept_candidate(candidate)
        if concept and concept not in seen:
            concepts.append(concept)
            seen.add(concept)

    lines = raw.splitlines()
    in_concepts = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        heading = _normalize_heading(stripped)
        if _is_concepts_heading(heading):
            in_concepts = True
            # Handle inline style: "CONCEPTS: RSA, AES, ..."
            if ":" in heading:
                inline = heading.split(":", 1)[1].strip()
                for part in re.split(r"[,;|]", inline):
                    add_concept(part)
            continue

        if in_concepts and _is_relationships_heading(heading):
            break

        if in_concepts:
            add_concept(stripped)

    # Fallback: parse all bullet/numbered lines if no explicit concepts section
    if not concepts:
        for line in lines:
            if REGEX_BULLET.match(line):
                add_concept(line)

    return concepts, raw


def _normalize_heading(line: str) -> str:
    """Normalize markdown-heavy heading lines for robust section matching."""
    normalized = re.sub(r"^[>#\-*\s]+", "", line.strip())
    return normalized.strip("*").strip()


def _is_concepts_heading(line: str) -> bool:
    return bool(REGEX_CONCEPTS_HEADING.match(line))


def _is_relationships_heading(line: str) -> bool:
    return bool(REGEX_RELATIONSHIPS_HEADING.match(line))


def _clean_concept_candidate(text: str) -> str:
    """Extract a concept name from a noisy list item line."""
    candidate = text.strip()
    candidate = REGEX_BULLET.sub("", candidate)
    candidate = candidate.strip().strip("*").strip("`")
    candidate = candidate.replace("**", "").replace("__", "")
    candidate = REGEX_MARKDOWN_LINK.sub(r"\1", candidate)

    # Skip relationships or obvious non-concept lines.
    if "->" in candidate or "http://" in candidate or "https://" in candidate:
        return ""

    # Keep only leading concept name before descriptions.
    candidate = re.split(r"\s+[—–-]\s+|:\s+", candidate, maxsplit=1)[0].strip()

    # Some model outputs include full explanatory sentences in the concept list.
    # Trim these to the leading concept phrase when possible.
    sentence_verb = REGEX_VERB.search(candidate)
    if sentence_verb and len(candidate.split()) >= 4:
        leading = candidate[:sentence_verb.start()].strip(" ,;:-")
        if leading and len(leading.split()) <= 6:
            candidate = leading
        else:
            return ""

    candidate = candidate.strip().rstrip(".;")

    if not candidate:
        return ""
    if len(candidate) > 90 or len(candidate.split()) > 12:
        return ""
    if re.fullmatch(r"[\W_0-9]+", candidate):
        return ""
    return candidate


def parse_wiki_page(raw: str) -> tuple[str, str]:
    """Parse a single wiki page response into (filename, content)."""
    match = REGEX_FILENAME.search(raw)
    if match:
        filename = match.group(1).strip()
        content = raw[match.end():].strip()
        return filename, content

    heading = REGEX_HEADING.search(raw)
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

    fixed = REGEX_WIKILINK.sub(fix_link, content)
    return fixed


def _normalize_concept(name: str) -> str:
    """Strip parentheticals and normalize for fuzzy comparison.
    'IND-CPA (Indistinguishability...)' → 'ind-cpa'
    """
    return re.sub(r'\s*\([^)]+\)', '', name).strip().lower()


def _concept_tokens(name: str) -> set[str]:
    """Tokenize a concept for overlap checks used in duplicate guards."""
    return set(re.findall(r"[a-z0-9]+", _normalize_concept(name)))


def _concept_aliases(name: str) -> set[str]:
    """Generate likely aliases (short name, abbreviation, normalized form)."""
    aliases = {_normalize_concept(name)}
    paren_match = re.match(r'^(.+?)\s*\((.+?)\)', name)
    if paren_match:
        aliases.add(paren_match.group(1).strip().lower())
        aliases.add(paren_match.group(2).strip().lower())
    return {a for a in aliases if a}


def _is_safe_near_duplicate(candidate: str, existing: str) -> bool:
    """Conservative guard to prevent merging distinct concepts."""
    norm_candidate = _normalize_concept(candidate)
    norm_existing = _normalize_concept(existing)

    if not norm_candidate or not norm_existing:
        return False
    if norm_candidate == norm_existing:
        return True

    alias_overlap = _concept_aliases(candidate) & _concept_aliases(existing)
    if alias_overlap:
        return True

    candidate_tokens = _concept_tokens(candidate)
    existing_tokens = _concept_tokens(existing)
    overlap = candidate_tokens & existing_tokens
    if not overlap:
        return False

    token_union = candidate_tokens | existing_tokens
    token_jaccard = len(overlap) / max(len(token_union), 1)
    norm_similarity = SequenceMatcher(None, norm_candidate, norm_existing).ratio()

    # Prefix matches are allowed only for strong stems (e.g. "ind cpa").
    if norm_candidate.startswith(norm_existing) or norm_existing.startswith(norm_candidate):
        return len(min(norm_candidate, norm_existing, key=len)) >= 6

    return norm_similarity >= 0.84 and token_jaccard >= 0.25


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
    if close and _is_safe_near_duplicate(concept, close[0]):
        return close[0]

    # Pass 2: normalized match with lower cutoff
    norm_concept = _normalize_concept(concept)
    norm_pages = {_normalize_concept(p): p for p in vault_pages}
    close_norm = get_close_matches(norm_concept, list(norm_pages.keys()),
                                   n=1, cutoff=0.75)
    if close_norm:
        candidate = norm_pages[close_norm[0]]
        if _is_safe_near_duplicate(concept, candidate):
            return candidate

    # Pass 3: prefix match — catches 'IND-CPA Security' vs 'IND-CPA (...)'
    # If the normalized concept is a prefix of a normalized vault page
    # (or vice versa) and the overlap is >= 6 chars, treat as duplicate
    for norm_page, original_page in norm_pages.items():
        shorter = min(norm_concept, norm_page, key=len)
        longer  = max(norm_concept, norm_page, key=len)
        if len(shorter) >= 6 and longer.startswith(shorter) and _is_safe_near_duplicate(concept, original_page):
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


def inject_active_wikilinks(content: str, related_concepts: list[str],
                           all_concepts: list[str]) -> str:
    """
    Inject wikilinks to related concepts that appear in the content.
    This enables the "system actively decides what should be linked" behavior.
    
    Only links concepts that:
    - Appear as whole words or phrases in the content
    - Are not already wikilinked
    - Are in the all_concepts list (to avoid linking outside the knowledge base)
    
    Args:
        content: Wiki page content to augment with links
        related_concepts: Concepts that should be linked (from concept graph)
        all_concepts: All known concepts (validates link targets)
    """
    linked_content = content
    concept_set = set(all_concepts)
    
    for related in related_concepts:
        if related not in concept_set:
            continue
        
        # Skip if already wikilinked
        if f'[[{related}]]' in linked_content:
            continue
        
        # Build pattern: match concept as whole words/phrases
        escaped = re.escape(related)
        # Match with word boundaries (handles punctuation around the concept)
        pattern = rf'\b{escaped}\b'
        
        # Replace first occurrence only (conservative: avoid over-linking)
        linked_content = re.sub(
            pattern,
            f'[[{related}]]',
            linked_content,
            count=1,
            flags=re.IGNORECASE
        )
    
    return linked_content


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
        r'[_-](pptx|pdf|docx|doc|txt|md|optimized|compressed|final|v\d+)$',
        '', raw_stem, flags=re.IGNORECASE
    ).replace('_', ' ').replace('-', ' ').strip()

    # Drop leading structural prefixes like "Unit3", "Week 4", "Chapter 2".
    cleaned = re.sub(
        r'^(?:week|unit|chapter|module|class|lecture|part)\s*\d+\s*',
        '',
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

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
        inferred = query(prompt, task="cheap", max_tokens=20).strip().strip('"\'\\n')
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


def _context_hash(text: str) -> str:
    """Short stable hash used to skip unchanged merge generations."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _extract_source_hash(content: str) -> str | None:
    match = REGEX_SOURCE_HASH.search(content)
    if match:
        return match.group(1)
    return None


def _upsert_source_hash_marker(content: str, source_hash: str) -> str:
    marker = f"<!-- source_context_hash: {source_hash} -->"
    if REGEX_SOURCE_HASH.search(content):
        return REGEX_SOURCE_HASH.sub(marker, content, count=1)

    if content.startswith("---\n"):
        closing = content.find("\n---\n", 4)
        if closing != -1:
            insert_at = closing + len("\n---\n")
            return content[:insert_at] + marker + "\n\n" + content[insert_at:].lstrip("\n")

    return marker + "\n\n" + content


def _split_adjacent_wikilink_bullets(content: str) -> str:
    """Split '- [[A]][[B]]' into separate bullets for each wikilink."""
    fixed_lines: list[str] = []
    for line in content.splitlines():
        stripped = line.lstrip()
        is_bullet = stripped.startswith("- ") or stripped.startswith("* ")
        line_has_adjacent = is_bullet and bool(re.search(r"\]\]\s*\[\[", line))

        if not line_has_adjacent:
            fixed_lines.append(line)
            continue

        indent_len = len(line) - len(stripped)
        bullet = stripped[:2] if len(stripped) >= 2 else "- "
        indent = " " * indent_len
        links = REGEX_WIKILINK.findall(line)
        if len(links) >= 2:
            for link in links:
                fixed_lines.append(f"{indent}{bullet}[[{link}]]")
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


def _postprocess_generated_content(content: str) -> str:
    """Normalize common model artifacts in generated markdown."""
    fixed = _split_adjacent_wikilink_bullets(content)
    fixed = re.sub(r"\]\]\s*\[\[", "]] [[", fixed)
    fixed = re.sub(r"\bsmall encryption keys\b", "small key spaces", fixed, flags=re.IGNORECASE)
    fixed = re.sub(r"\bsize of (?:the )?encryption key(?:s)?\b", "size of the key space", fixed, flags=re.IGNORECASE)
    return fixed


def _build_index(chapters: list[dict]) -> tuple[list[str], str]:
    """Run pass-1 indexing and return parsed concepts and raw index text."""
    index_text_input = "\n\n---\n\n".join(ch["content"] for ch in chapters)
    index_prompt = load_prompt("index").replace("{text}", index_text_input)
    index_raw = query(index_prompt, task="cheap", max_tokens=4000)
    concepts, index_text = parse_index(index_raw)
    if not concepts:
        raise ValueError(
            "No concepts were parsed from the index response. "
            "Check the prompt/output format before continuing."
        )
    return concepts, index_text


def _get_subject(raw_stem: str, concepts: list[str], subject_override: str, batch_mode: bool) -> str:
    if subject_override:
        print(f"  Subject (from --subject flag): \"{subject_override}\"")
        return subject_override
    return detect_subject(raw_stem, concepts, batch_mode=batch_mode)


def _collect_vault_pages(vault_state: dict) -> list[str]:
    return [
        page
        for pages in vault_state["pages"].values()
        for page in pages.keys()
    ]


def _distill_concept_context(
    all_chunks: list[str],
    concept: str,
    concepts: list[str],
    max_chars: int = 4000,
    retrieve_top_k: int = 2,
    extract_max_tokens: int = 450,
    skip_extract_min_chars: int = 1400,
) -> tuple[str, str, bool]:
    """Retrieve -> dedupe/rank -> adaptive sizing -> distill facts for one concept."""
    other_concepts = [c for c in concepts if c != concept]
    
    # Get ranked chunks with scores for intelligent context sizing
    ranked_with_scores = retrieve_ranked_chunks_with_scores(
        all_chunks,
        concept=concept,
        related_concepts=other_concepts[:5],
        top_k=retrieve_top_k,
    )
    
    # Calculate average relevance score to adjust context size
    if ranked_with_scores:
        avg_score = sum(score for _, score in ranked_with_scores) / len(ranked_with_scores)
        adaptive_max_chars = _compute_adaptive_context_size(
            avg_score=avg_score,
            base_max_chars=max_chars,
        )
    else:
        adaptive_max_chars = max_chars
    
    ranked_chunks = [chunk for chunk, _ in ranked_with_scores]
    relevant_text = limit_context(ranked_chunks, max_chars=adaptive_max_chars)

    # Fast path: short contexts don't benefit much from another model pass.
    if len(relevant_text) < skip_extract_min_chars:
        return relevant_text, relevant_text, True

    facts_text = extract_facts(concept, relevant_text, max_tokens=extract_max_tokens).strip()
    distilled = facts_text or relevant_text
    return distilled, relevant_text, not bool(facts_text)


# --- Main pipeline ---

def _resolve_run_profile(profile: str | None) -> tuple[str, dict[str, int]]:
    """Resolve run profile from arg/env/default and return profile name + settings."""
    raw = (profile or os.environ.get("PDF_TO_NOTES_PROFILE", "hybrid")).strip().lower()
    normalized = RUN_PROFILE_ALIASES.get(raw, raw)
    if normalized not in RUN_PROFILE_SETTINGS:
        print(f"  [warn] invalid profile {raw!r}; using 'hybrid'", flush=True)
        normalized = "hybrid"
    return normalized, RUN_PROFILE_SETTINGS[normalized]


def _resolve_max_workers(max_workers: int | None, default_workers: int) -> int:
    """Resolve concept-processing worker count from arg/env/default."""
    if max_workers is not None:
        return max(1, int(max_workers))

    env_raw = os.environ.get("PDF_TO_NOTES_MAX_WORKERS", "").strip()
    if env_raw:
        try:
            return max(1, int(env_raw))
        except ValueError:
            print(f"  [warn] invalid PDF_TO_NOTES_MAX_WORKERS={env_raw!r}; using default", flush=True)

    return max(1, default_workers)


def process_pdf(
    pdf_path: str,
    output_dir: str,
    subject_override: str = "",
    batch_mode: bool = False,
    max_workers: int | None = None,
    profile: str | None = None,
):
    raw_stem = Path(pdf_path).stem
    print(f"\n{'='*50}", flush=True)
    print(f"Processing: {raw_stem}", flush=True)
    print(f"{'='*50}", flush=True)

    # 1. Extract text
    print("\n[1/6] Extracting text from PDF...", flush=True)
    full_text = extract_text(pdf_path)

    profile_name, profile_settings = _resolve_run_profile(profile)
    print(
        f"  Run profile: {profile_name} "
        f"(ctx={profile_settings['context_max_chars']}, "
        f"extract_tokens={profile_settings['extract_max_tokens']}, "
        f"write_tokens={profile_settings['write_max_tokens']}, "
        f"merge_tokens={profile_settings['merge_max_tokens']}, "
        f"top_k={profile_settings['retrieve_top_k']})",
        flush=True,
    )

    # 2. Split into chapters and chunk each one
    print("\n[2/6] Splitting into chapters and chunking...", flush=True)
    chapters = split_into_chapters(full_text)
    # Smart chunking auto-selects page/headings/paragraph/size strategy.
    all_chunks = smart_chunk(full_text, pages_per_chunk=2)
    print(f"  Total chunks: {len(all_chunks)}", flush=True)

    # For flashcards/cheatsheet: compressed summary (first chunk per chapter)
    summary_text = _chapter_summary_text(chapters)

    # 3. Build concept index — Pass 1
    print("\n[3/6] Building concept index (Pass 1)...", flush=True)
    concepts, index_text = _build_index(chapters)
    print(f"  Found {len(concepts)} concepts: {', '.join(concepts)}", flush=True)

    subject = _get_subject(raw_stem, concepts, subject_override, batch_mode)

    # Build concept graph for active linking (system decides what should link)
    print("\n[Pass 1.5] Building concept relationship graph...", flush=True)
    concept_graph = build_concept_graph(concepts, all_chunks)
    print(f"  Concept relationships mapped", flush=True)

    # 4. Generate wiki pages — Pass 2 (incremental: new / merge / skip)
    print(f"\n[4/6] Processing {len(concepts)} concepts (Pass 2)...", flush=True)
    vault_state = load_vault_state(output_dir)
    all_vault_pages = _collect_vault_pages(vault_state)
    wiki_pages = {}         # new pages to write
    merged_pages = {}       # existing pages that were updated
    skipped = []            # concepts with no new content
    concept_names = "\n".join(f"- {c}" for c in concepts)
    wiki_prompt_template = load_prompt("wiki")
    merge_prompt_template = load_prompt("merge")

    def _process_single_concept(i: int, concept: str) -> dict:
        """Run pass-2 processing for one concept in isolation."""
        distilled_facts, retrieved_context, extraction_skipped = _distill_concept_context(
            all_chunks,
            concept,
            concepts,
            max_chars=profile_settings["context_max_chars"],
            retrieve_top_k=profile_settings["retrieve_top_k"],
            extract_max_tokens=profile_settings["extract_max_tokens"],
        )
        source_hash = _context_hash(retrieved_context)
        existing_path = find_existing_page(concept, subject, vault_state)
        near_dup = None

        # Cross-run near-duplicate check — catches spelling variants like
        # "Kasiski Test" matching existing "Kasisky Test"
        if existing_path is None and all_vault_pages:
            near_dup = find_near_duplicate(concept, all_vault_pages)
            if near_dup:
                subject_pages = vault_state["pages"].get(subject, {})
                if near_dup in subject_pages:
                    existing_path = subject_pages[near_dup]

        if existing_path is None:
            page_prompt = (wiki_prompt_template
                           .replace("{concept}", concept)
                           .replace("{index}", concept_names)
                           .replace("{concept_names}", concept_names)
                           .replace("{facts}", distilled_facts)
                           .replace("{text}", distilled_facts))
            page_raw = query(
                page_prompt,
                task="write",
                max_tokens=profile_settings["write_max_tokens"],
            )
            filename, page_content = parse_wiki_page(page_raw)
            page_content = fix_wikilinks(page_content, concepts,
                                         vault_pages=all_vault_pages)
            graph_related = set(concept_graph.get(concept, set()))
            content_related = set(find_related_concepts(page_content, concepts))
            active_related = sorted((graph_related | content_related) - {concept})
            page_content = inject_active_wikilinks(page_content, active_related, concepts)
            page_content = _postprocess_generated_content(page_content)
            page_content = add_frontmatter(filename, page_content, concepts,
                                           subject=subject)
            page_content = _upsert_source_hash_marker(page_content, source_hash)
            return {
                "index": i,
                "concept": concept,
                "kind": "new",
                "near_dup": near_dup,
                "distilled_len": len(distilled_facts),
                "filename": filename,
                "content": page_content,
                "reason": "extract-skipped" if extraction_skipped else "generated",
            }

        existing_content = Path(existing_path).read_text(encoding="utf-8")
        existing_hash = _extract_source_hash(existing_content)
        if existing_hash == source_hash:
            return {
                "index": i,
                "concept": concept,
                "kind": "skip",
                "near_dup": near_dup,
                "existing_path": existing_path,
                "reason": "source-hash-match",
            }

        merge_prompt = (merge_prompt_template
                        .replace("{existing_content}", existing_content)
                        .replace("{concept}", concept)
                        .replace("{facts}", distilled_facts)
                        .replace("{new_content}", distilled_facts)
                        .replace("{source}", subject)
                        .replace("{concept_names}", concept_names))
        merge_raw = query(
            merge_prompt,
            task="write",
            max_tokens=profile_settings["merge_max_tokens"],
        )

        if merge_raw.strip() == "NO_UPDATE":
            return {
                "index": i,
                "concept": concept,
                "kind": "skip",
                "near_dup": near_dup,
                "existing_path": existing_path,
                "reason": "no-update",
            }

        merge_raw = fix_wikilinks(merge_raw, concepts,
                                  vault_pages=all_vault_pages)
        graph_related = set(concept_graph.get(concept, set()))
        content_related = set(find_related_concepts(merge_raw, concepts))
        active_related = sorted((graph_related | content_related) - {concept})
        merge_raw = inject_active_wikilinks(merge_raw, active_related, concepts)
        merge_raw = _postprocess_generated_content(merge_raw)
        merge_raw = _upsert_source_hash_marker(merge_raw, source_hash)
        return {
            "index": i,
            "concept": concept,
            "kind": "merge",
            "near_dup": near_dup,
            "existing_path": existing_path,
            "stem": Path(existing_path).stem,
            "content": merge_raw,
            "reason": "extract-skipped" if extraction_skipped else "merged",
        }

    concept_workers = _resolve_max_workers(
        max_workers,
        default_workers=profile_settings["default_max_workers"],
    )
    print(f"  Parallel concept workers: {concept_workers}", flush=True)

    results: list[dict] = []
    concept_errors: list[tuple[int, str, str]] = []
    total_concepts = len(concepts)
    completed = 0
    with ThreadPoolExecutor(max_workers=concept_workers) as executor:
        futures = {}
        for i, concept in enumerate(concepts):
            print(f"  [{i+1}/{total_concepts}] QUEUED: {concept}", flush=True)
            future = executor.submit(_process_single_concept, i, concept)
            futures[future] = (i, concept)

        for future in as_completed(futures):
            i, concept = futures[future]
            try:
                item = future.result()
                results.append(item)
                completed += 1
                kind = item.get("kind", "unknown").upper()
                near_dup = item.get("near_dup")
                if near_dup:
                    print(f"  [{i+1}/{total_concepts}] DONE ({completed}/{total_concepts}) {kind}: {concept} (near-dup: {near_dup})", flush=True)
                else:
                    print(f"  [{i+1}/{total_concepts}] DONE ({completed}/{total_concepts}) {kind}: {concept}", flush=True)
            except Exception as exc:
                concept_errors.append((i, concept, str(exc)))
                completed += 1
                print(f"  [{i+1}/{total_concepts}] FAILED ({completed}/{total_concepts}): {concept}", flush=True)

    # Apply results in original concept order for deterministic output.
    for item in sorted(results, key=lambda x: x["index"]):
        kind = item["kind"]
        if kind == "new":
            wiki_pages[item["filename"]] = item["content"]
        elif kind == "merge":
            merged_pages[item["stem"]] = (item["existing_path"], item["content"])
        else:
            skipped.append(item["concept"])

    if concept_errors:
        print(f"  [warn] {len(concept_errors)} concepts failed during parallel processing:", flush=True)
        for i, concept, err in sorted(concept_errors, key=lambda x: x[0]):
            print(f"    - [{i+1}/{len(concepts)}] {concept}: {err}", flush=True)

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

        moc_raw = query(moc_prompt, task="cheap", max_tokens=2000)
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
    cards_raw = query(load_prompt("flashcards").replace("{text}", summary_text), task="cheap")
    write_flashcards(output_dir, subject, cards_raw)

    sheet_raw = query(load_prompt("cheatsheet").replace("{text}", summary_text), task="cheap")
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
    parser.add_argument(
        "--provider",
        choices=["anthropic", "ollama"],
        default=None,
        help=(
            "Model provider override for this run. "
            "Defaults to PDF_TO_NOTES_PROVIDER from environment."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=(
            "Maximum parallel workers for concept pass (Pass 2). "
            "Defaults to PDF_TO_NOTES_MAX_WORKERS or an auto value."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=["speed", "quality", "hybrid"],
        default=None,
        help=(
            "Performance/quality profile. "
            "Also configurable via PDF_TO_NOTES_PROFILE."
            "'balanced' is an alias for 'hybrid'."
        ),
    )
    args = parser.parse_args(argv)

    if args.subject and len(args.pdfs) > 1:
        parser.error("--subject can only be used with a single PDF")

    if args.provider:
        set_provider(args.provider)
        print(f"Provider override: {get_provider()}")

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
                        batch_mode=batch,
                        max_workers=args.max_workers,
                        profile=args.profile)
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