"""Parsing and markdown content utilities for the PDF wiki pipeline."""

from __future__ import annotations

import hashlib
import re
from difflib import get_close_matches
from typing import Callable


# --- Precompiled regex patterns for performance (avoid recompilation) ---
REGEX_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
REGEX_FILENAME = re.compile(r"^\*{0,2}FILENAME:\*{0,2}\s*(.+?)\*{0,2}$", re.MULTILINE)
REGEX_HEADING = re.compile(r"^##\s+(.+)$", re.MULTILINE)
REGEX_WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")
REGEX_CONCEPTS_HEADING = re.compile(r"^(concepts?|key concepts?|topics?)\s*:?.*$", re.IGNORECASE)
REGEX_RELATIONSHIPS_HEADING = re.compile(r"^(relationships?|links?)\s*:?.*$", re.IGNORECASE)
REGEX_VERB = re.compile(
    r"\b(depends on|is|are|means|refers to|involves|uses|requires|allows|provides|ensures)\b",
    re.IGNORECASE,
)
REGEX_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
REGEX_SOURCE_HASH = re.compile(r"<!--\s*source_context_hash:\s*([0-9a-f]{12,40})\s*-->")
REGEX_BAD_CONCEPT_START = re.compile(
    r"^(if|when|while|because|although|though|however|therefore|thus|whereas|unless|until)\b",
    re.IGNORECASE,
)
REGEX_SETEXT_RULE = re.compile(r"^[=\-]{3,}\s*$")


GENERIC_NAMES = {
    "lecture",
    "notes",
    "slides",
    "week",
    "chapter",
    "unit",
    "class",
    "module",
    "part",
    "doc",
    "file",
    "reading",
    "handout",
    "worksheet",
    "assignment",
    "homework",
    "lab",
    "tutorial",
}


def parse_index(raw: str) -> tuple[list[str], str]:
    """Extract concept names from index output."""
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
            if ":" in heading:
                inline = heading.split(":", 1)[1].strip()
                for part in re.split(r"[,;|]", inline):
                    add_concept(part)
            continue

        if in_concepts and _is_relationships_heading(heading):
            break

        if in_concepts:
            add_concept(stripped)

    if not concepts:
        for line in lines:
            if REGEX_BULLET.match(line):
                add_concept(line)

    return concepts, raw


def _normalize_heading(line: str) -> str:
    normalized = re.sub(r"^[>#\-*\s]+", "", line.strip())
    return normalized.strip("*").strip()


def _is_concepts_heading(line: str) -> bool:
    return bool(REGEX_CONCEPTS_HEADING.match(line))


def _is_relationships_heading(line: str) -> bool:
    return bool(REGEX_RELATIONSHIPS_HEADING.match(line))


def _clean_concept_candidate(text: str) -> str:
    candidate = text.strip()
    candidate = REGEX_BULLET.sub("", candidate)
    candidate = candidate.strip().strip("*").strip("`")
    candidate = candidate.replace("**", "").replace("__", "")
    candidate = REGEX_MARKDOWN_LINK.sub(r"\1", candidate)

    if re.search(r"->|=>|→|⇒|<-|←", candidate) or "http://" in candidate or "https://" in candidate:
        return ""

    candidate = re.split(r"\s+[—–-]\s+|:\s+", candidate, maxsplit=1)[0].strip()

    sentence_verb = REGEX_VERB.search(candidate)
    if sentence_verb and len(candidate.split()) >= 4:
        leading = candidate[:sentence_verb.start()].strip(" ,;:-")
        if leading and len(leading.split()) <= 6:
            candidate = leading
        else:
            return ""

    candidate = candidate.strip().rstrip(".;")
    if REGEX_BAD_CONCEPT_START.match(candidate):
        return ""

    lower = candidate.lower()
    if lower.startswith(("if an ", "if a ", "if the ", "when an ", "when a ", "when the ")):
        return ""

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
        content = raw[match.end() :].strip()
        return filename, content

    heading = REGEX_HEADING.search(raw)
    if heading:
        return heading.group(1).strip(), raw

    return "Untitled", raw


def build_alias_map(concepts: list[str]) -> dict[str, str]:
    alias_map = {}
    for concept in concepts:
        alias_map[concept.lower()] = concept
        paren_match = re.match(r"^(.+?)\s*\((.+?)\)", concept)
        if paren_match:
            short = paren_match.group(1).strip()
            abbrev = paren_match.group(2).strip()
            alias_map[short.lower()] = concept
            alias_map[abbrev.lower()] = concept
        words = concept.split()
        if len(words) > 1:
            alias_map[words[0].lower()] = concept
    return alias_map


def fix_wikilinks(content: str, concepts: list[str], vault_pages: list[str] | None = None) -> str:
    all_known = list(dict.fromkeys(concepts + (vault_pages or [])))
    alias_map = build_alias_map(all_known)
    concept_set = set(all_known)

    def fix_link(match):
        link_text = match.group(1).strip()
        if link_text in concept_set:
            return f"[[{link_text}]]"
        canonical = alias_map.get(link_text.lower())
        if canonical:
            return f"[[{canonical}]]"
        close = get_close_matches(link_text, all_known, n=1, cutoff=0.82)
        if close:
            return f"[[{close[0]}]]"
        return match.group(0)

    return REGEX_WIKILINK.sub(fix_link, content)


def add_frontmatter(filename: str, content: str, concepts: list[str], subject: str = "", tags: list[str] | None = None) -> str:
    aliases = []
    paren_match = re.match(r"^(.+?)\s*\(", filename)
    if paren_match:
        short = paren_match.group(1).strip()
        aliases.append(short)
        abbrev_match = re.match(r"^.+?\((.+?)\)", filename)
        if abbrev_match:
            aliases.append(abbrev_match.group(1).strip())
    words = filename.split()
    if len(words) > 2:
        aliases.append(words[0])
        aliases.append(" ".join(words[:2]))
    aliases = list(dict.fromkeys(a for a in aliases if a != filename))

    content_lower = content.lower()
    if any(w in content_lower for w in ["euler", "modular", "discrete log", "polynomial", "complexity"]):
        difficulty = "advanced"
    elif any(w in content_lower for w in ["algorithm", "protocol", "attack", "encrypt"]):
        difficulty = "intermediate"
    else:
        difficulty = "foundational"

    all_tags = list(tags or [])
    subject_tag = subject.lower().replace(" ", "-") if subject else ""
    if subject_tag and subject_tag not in all_tags:
        all_tags.append(subject_tag)
    difficulty_tag = f"difficulty/{difficulty}"
    if difficulty_tag not in all_tags:
        all_tags.append(difficulty_tag)

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


def inject_active_wikilinks(content: str, related_concepts: list[str], all_concepts: list[str]) -> str:
    linked_content = content
    concept_set = set(all_concepts)

    for related in related_concepts:
        if related not in concept_set:
            continue
        if f"[[{related}]]" in linked_content:
            continue
        escaped = re.escape(related)
        pattern = rf"\b{escaped}\b"
        linked_content = re.sub(pattern, f"[[{related}]]", linked_content, count=1, flags=re.IGNORECASE)

    return linked_content


def _is_readable_word(s: str) -> bool:
    s = s.lower().replace(" ", "")
    if not s:
        return False
    vowels = sum(1 for c in s if c in "aeiou")
    if len(s) > 4 and vowels / len(s) < 0.15:
        return False
    if re.match(r"^[0-9a-f]{8,}$", s):
        return False
    consonant_run = max((len(m.group()) for m in re.finditer(r"[bcdfghjklmnpqrstvwxyz]+", s)), default=0)
    if consonant_run > 5:
        return False
    return True


def detect_subject(
    raw_stem: str,
    concepts: list[str] | None = None,
    batch_mode: bool = False,
    query_fn: Callable[..., str] | None = None,
) -> str:
    cleaned = re.sub(
        r"[_-](pptx|pdf|docx|doc|txt|md|optimized|compressed|final|v\d+)$",
        "",
        raw_stem,
        flags=re.IGNORECASE,
    ).replace("_", " ").replace("-", " ").strip()

    cleaned = re.sub(
        r"^(?:week|unit|chapter|module|class|lecture|part)\s*\d+\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    first_word = cleaned.split()[0].lower() if cleaned.split() else ""
    is_generic = (
        first_word in GENERIC_NAMES
        or cleaned.lower() in GENERIC_NAMES
        or re.match(r"^\d+$", cleaned)
        or re.match(r"^week\s*\d+", cleaned, re.IGNORECASE)
        or len(cleaned) <= 2
        or not _is_readable_word(cleaned)
    )

    if not is_generic:
        print(f'  Subject detected from filename: "{cleaned}"')
        return cleaned

    if concepts and query_fn:
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
        inferred = query_fn(prompt, task="cheap", max_tokens=20).strip().strip('"\'\n')
        if inferred and len(inferred.split()) <= 5 and "." not in inferred:
            print(f'  Subject inferred from content: "{inferred}"')
            return inferred
        print("  AI inference inconclusive.")

    if batch_mode:
        print(f"  WARNING: could not determine subject for '{raw_stem}'.")
        print("  Files will be written to 'Unsorted/' — review and move manually.")
        return "Unsorted"

    print("  Could not auto-detect subject.")
    user_input = input("  Enter subject name for this PDF (e.g. 'Cryptography'): ").strip()
    return user_input or cleaned or "Unsorted"


def chapter_summary_text(chapters: list[dict], chunk_text_fn: Callable[..., list[str]]) -> str:
    summaries = []
    for chapter in chapters:
        chunks = chunk_text_fn(chapter["content"], max_chars=2000)
        if chunks:
            summaries.append(chunks[0])
    return "\n\n---\n\n".join(summaries)


def context_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def extract_source_hash(content: str) -> str | None:
    match = REGEX_SOURCE_HASH.search(content)
    if match:
        return match.group(1)
    return None


def upsert_source_hash_marker(content: str, source_hash: str) -> str:
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


def _normalize_setext_headings(content: str) -> str:
    """Convert setext headings to ATX style for consistent markdown structure."""
    lines = content.splitlines()
    fixed: list[str] = []
    i = 0
    while i < len(lines):
        current = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        if nxt is not None and REGEX_SETEXT_RULE.match(nxt.strip()) and current.strip():
            fixed.append(f"## {current.strip()}")
            i += 2
            continue
        fixed.append(current)
        i += 1
    return "\n".join(fixed)


def _normalize_related_section(content: str) -> str:
    """Normalize related sections to: '### Related' + bullet list with wikilinks."""
    lines = content.splitlines()
    related_idx = -1
    for idx, line in enumerate(lines):
        normalized = line.strip().lstrip("#").strip().lower()
        if normalized in {"related", "related concept", "related concepts"}:
            related_idx = idx
            break

    if related_idx == -1:
        return content

    before = "\n".join(lines[:related_idx]).rstrip()
    tail = lines[related_idx + 1 :]
    bullets: list[str] = []
    seen: set[str] = set()

    for line in tail:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("#"):
            candidate = stripped.lstrip("#").strip()
            if candidate and candidate.lower() not in {"related", "related concept", "related concepts"}:
                wikilink = f"[[{candidate}]]"
                if wikilink.lower() not in seen:
                    bullets.append(f"- {wikilink}")
                    seen.add(wikilink.lower())
            continue

        body = stripped[2:].strip() if stripped.startswith(("- ", "* ")) else stripped
        if body.startswith("[["):
            concept = body.split("]]", 1)[0].strip("[] ")
            reason = body.split(":", 1)[1].strip() if ":" in body else ""
        else:
            concept = re.split(r"\s*:\s*|\s+-\s+", body, maxsplit=1)[0].strip()
            reason = body.split(":", 1)[1].strip() if ":" in body else ""

        if not concept:
            continue

        wikilink = f"[[{concept}]]"
        key = wikilink.lower()
        if key in seen:
            continue
        seen.add(key)
        bullets.append(f"- {wikilink}: {reason}" if reason else f"- {wikilink}")

    if not bullets:
        return before + "\n\n### Related\n"
    return before + "\n\n### Related\n" + "\n".join(bullets)


def standardize_wiki_markdown(content: str) -> str:
    """Enforce a consistent, Wikipedia-style markdown structure."""
    fixed = _normalize_setext_headings(content)
    fixed = _normalize_related_section(fixed)
    fixed = re.sub(r"\n{3,}", "\n\n", fixed)
    return fixed.strip() + "\n"


def postprocess_generated_content(content: str) -> str:
    fixed = _split_adjacent_wikilink_bullets(content)
    fixed = re.sub(r"\]\]\s*\[\[", "]] [[", fixed)
    fixed = re.sub(r"\bsmall encryption keys\b", "small key spaces", fixed, flags=re.IGNORECASE)
    fixed = re.sub(r"\bsize of (?:the )?encryption key(?:s)?\b", "size of the key space", fixed, flags=re.IGNORECASE)
    fixed = standardize_wiki_markdown(fixed)
    return fixed


def looks_incomplete_output(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if REGEX_WIKILINK.search(stripped) and stripped.count("[[") != stripped.count("]]"):
        return True
    if stripped.count("```") % 2 != 0:
        return True
    if stripped.endswith(("[[", "[", "(`", "(", " -", " :", " Al")):
        return True
    return False


def query_with_quality_retry(
    prompt: str,
    max_tokens: int,
    task: str,
    query_fn: Callable[..., str],
) -> str:
    first = query_fn(prompt, task=task, max_tokens=max_tokens)
    if not looks_incomplete_output(first):
        return first

    retry_prompt = prompt + "\n\nIMPORTANT: Return COMPLETE markdown only. Close all [[wikilinks]] and code fences."
    return query_fn(retry_prompt, task=task, max_tokens=max(int(max_tokens * 1.35), max_tokens + 200))
