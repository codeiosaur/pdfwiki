"""
Fact extraction from text chunks using a pluggable LLM backend.

Two-pass pipeline:
  Pass 1: Extract raw factual statements (no concept naming)
  Pass 2: Assign concept names from a seed list

Legacy single-pass functions are kept for backward compatibility.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import json
import time
import uuid

if TYPE_CHECKING:
    from ingest.pdf_loader import Chunk
    from backend.base import LLMBackend


@dataclass
class Fact:
    id: str
    concept: str
    content: str
    source_chunk_id: str


# ---------------------------------------------------------------------------
# Built-in seed fallback — used only when auto-generation fails and no
# --seeds file is provided.  Domain-specific data lives in seeds/*.json,
# not in Python source.
# ---------------------------------------------------------------------------
_BUILTIN_SEEDS_FILE = Path(__file__).parent.parent.parent / "seeds" / "accounting.json"


def load_builtin_seeds() -> List[str]:
    """Load the built-in accounting seed list from seeds/accounting.json."""
    try:
        return load_seeds_from_file(str(_BUILTIN_SEEDS_FILE))
    except Exception as exc:
        print(f"  [seeds] Warning: could not load built-in seeds ({exc}); no fallback available")
        return []


# ---------------------------------------------------------------------------
# Seed concept utilities
# ---------------------------------------------------------------------------

def load_seeds_from_file(path: str) -> List[str]:
    """Load a seed concept list from a JSON file (array of strings)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Seeds file must be a JSON array, got {type(data).__name__}")
    seeds = [s.strip() for s in data if isinstance(s, str) and s.strip()]
    if not seeds:
        raise ValueError("Seeds file contained no valid strings")
    return seeds


def derive_seed_concepts(
    statements: List[dict],
    backend: "LLMBackend",
    target_count: int = 40,
    sample_size: int = 120,
) -> List[str]:
    """
    Pass 1.5: Derive seed concept names from extracted statements.

    Asks the LLM to identify the key concepts covered by the statements,
    producing a domain-specific seed list for Pass 2 concept assignment.
    Returns an empty list on failure so the caller can fall back to SEED_CONCEPTS.
    """
    if not statements:
        return []

    sample = statements[:sample_size]
    statement_block = "\n".join(
        f"  {i + 1}. {s['statement']}" for i, s in enumerate(sample)
    )

    prompt = f"""The following statements were extracted from a document.
Identify the {target_count} most important distinct concept names a reader would need to know.

Rules:
- Short noun phrases only (1-4 words)
- Use standard domain terminology
- Each name must be meaningfully distinct
- Do NOT use vague names like "Overview", "Impact", "Effects", "Goals", "Management"
- Do NOT repeat the same concept under different phrasings
- Prefer concepts the document teaches in depth over concepts it merely mentions as examples or context. A concept worth a page is one a reader must understand to master the subject — not one that appears only to illustrate a point.

Statements:
{statement_block}

Output ONLY a JSON array of concept name strings:
["Concept Name", "Another Concept", ...]"""

    try:
        raw = backend.generate(prompt)
    except Exception as exc:
        print(f"  [pass1.5] Failed to derive seeds: {exc}")
        return []

    parsed = _parse_json_array(raw)
    if not isinstance(parsed, list):
        preview = (raw[:200] + "...") if len(raw) > 200 else raw
        print(f"  [pass1.5] Could not parse seed list from response: {preview!r}")
        return []

    seeds = [s.strip() for s in parsed if isinstance(s, str) and s.strip()]
    return seeds


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences (```json ... ```)."""
    text = text.strip()
    if text.startswith("```"):
        # Strip opening fence line
        newline = text.find("\n")
        if newline != -1:
            text = text[newline + 1:]
    if text.endswith("```"):
        text = text[: text.rfind("```")].rstrip()
    return text.strip()


def _parse_json_array(raw_content: str):
    """Parse a JSON array from raw model output.

    Handles three common model output patterns:
    - Bare array:              [{"a": 1}, ...]
    - Markdown-fenced:         ```json\n[...]\n```
    - Object-wrapped array:    {"items": [...]} or {"statements": [...]}
      (produced by providers using json_object mode, which enforces a dict root)
    """
    cleaned = _strip_markdown_fences(raw_content)
    try:
        parsed = json.loads(cleaned)
    except Exception:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except Exception:
            return None

    if isinstance(parsed, list):
        return parsed

    # json_object mode wraps the array in a dict (e.g. {"statements": [...]}).
    # Take the longest list value — that's the payload, not an empty error list.
    if isinstance(parsed, dict):
        lists = [v for v in parsed.values() if isinstance(v, list)]
        if lists:
            return max(lists, key=len)

    return None


def _classify_json_failure(raw_content: str) -> str:
    """Return a short label describing why JSON parsing failed."""
    cleaned = _strip_markdown_fences(raw_content).strip()
    if not cleaned:
        return "empty response"
    if cleaned.startswith("[") and not cleaned.endswith("]"):
        return "truncated (array started but not closed — increase num_ctx or reduce batch size)"
    if cleaned.startswith("{"):
        try:
            keys = list(json.loads(cleaned).keys())
            return f"wrong shape (object instead of array) — keys: {keys}"
        except Exception:
            pass
        return "wrong shape (object instead of array)"
    return "malformed (unexpected format)"


def _parse_json_object(raw_content: str):
    """Parse a JSON object from raw model output."""
    try:
        return json.loads(raw_content)
    except Exception:
        start = raw_content.find("{")
        end = raw_content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(raw_content[start : end + 1])
        except Exception:
            return None


# ===================================================================
# PASS 1: Extract raw atomic statements (no concept assignment)
# ===================================================================

def extract_raw_statements_batched(
    chunks: List["Chunk"],
    backend: "LLMBackend",
    batch_size: int = 4,
) -> List[dict]:
    """
    Pass 1 (batched): Extract raw factual statements from chunks.

    The model only extracts facts — it does NOT assign concept names.
    This is a much simpler task for small models and produces
    cleaner output.

    batch_size: number of chunks sent to the LLM in a single call (default 4).
    Callers can pass a smaller value (e.g. 1 or 2) for models with a tight
    context window, or a larger value to reduce total API round-trips.
    """
    if not chunks:
        return []
    if batch_size < 1:
        batch_size = 1

    all_statements: List[dict] = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        allowed_chunk_ids = {chunk.id for chunk in batch}

        combined_sections: List[str] = []
        for chunk in batch:
            combined_sections.append(f"[CHUNK_ID={chunk.id}]\n{chunk.text}")
        combined_text = "\n\n".join(combined_sections)

        prompt = f"""Extract factual statements from the text sections below.

Rules:
- Be thorough -- extract 10-20 statements per text section.
- Each statement must be ONE complete, self-contained factual claim.
- Write clear sentences a student could study from.
- Include definitions, relationships between concepts, rules, exceptions, and comparisons.
- When a formula or mathematical relationship is present, extract it as a concise symbolic statement (e.g. 'X Ratio = A ÷ B') AND as a plain-English description. These are two separate statements.
- Include brief self-contained worked examples that show how a concept applies (e.g. 'Under method X, if input is Y, output is Z'). Omit multi-step worksheet calculations that merely repeat the same arithmetic.
- Skip instructions ("calculate...", "determine...", "prepare...").
- Skip figure/table references and page numbers.
- Skip statements where a concept is only referenced as a passing example or analogy. Extract statements that explain how or why something works, not merely that something exists.
- Include the source_chunk_id from the [CHUNK_ID=...] marker.

Allowed source_chunk_id values: {sorted(allowed_chunk_ids)}

Output ONLY a JSON array:
[{{"statement": "...", "source_chunk_id": "..."}}]

Text:
{combined_text}"""

        # JSON schema for structured output (enforced by OpenRouter, best-effort elsewhere)
        statement_extraction_schema = {
            "name": "extracted_statements",
            "schema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "statement": {
                            "type": "string",
                            "description": "A single factual claim extracted from the text",
                        },
                        "source_chunk_id": {
                            "type": "string",
                            "description": "The chunk ID from the [CHUNK_ID=...] marker",
                        },
                    },
                    "required": ["statement", "source_chunk_id"],
                    "additionalProperties": False,
                },
            },
        }

        batch_num = i // batch_size + 1
        parsed = None
        _MAX_JSON_RETRIES = 2
        for attempt in range(_MAX_JSON_RETRIES):
            t0 = time.monotonic()
            try:
                raw_content = backend.generate(
                    prompt, json_schema=statement_extraction_schema,
                    context=f"batch {batch_num}",
                )
            except Exception as exc:
                print(f"  [pass1] Batch {batch_num}: LLM call failed: {exc}")
                break
            elapsed = time.monotonic() - t0
            parsed = _parse_json_array(raw_content)
            if isinstance(parsed, list):
                break
            reason = _classify_json_failure(raw_content)
            retry_msg = f", retrying ({attempt + 2}/{_MAX_JSON_RETRIES})..." if attempt + 1 < _MAX_JSON_RETRIES else ""
            _lbl = backend.last_used_label() if hasattr(backend, "last_used_label") else backend.label
            print(f"  [{_lbl}] JSON parse failed — {reason}{retry_msg}")

        if not isinstance(parsed, list):
            continue

        # Build a mapping from chunk id -> source filename so we can attach
        # a human-friendly filename to each extracted statement. The LLM
        # returns only the chunk id; we know each chunk's source locally.
        chunk_id_to_source = {chunk.id: getattr(chunk, "source", "") for chunk in batch}

        for item in parsed:
            if not isinstance(item, dict):
                continue
            statement = item.get("statement", "")
            source_chunk_id = item.get("source_chunk_id", "")
            source_name = ""
            if isinstance(source_chunk_id, str) and source_chunk_id in chunk_id_to_source:
                source_name = chunk_id_to_source.get(source_chunk_id, "") or ""

            if (
                isinstance(statement, str)
                and len(statement.strip()) > 10
                and isinstance(source_chunk_id, str)
                and source_chunk_id in allowed_chunk_ids
            ):
                all_statements.append({
                    "statement": statement.strip(),
                    "chunk_id": source_chunk_id,
                    "source": source_name,
                })

    return all_statements


# ===================================================================
# PASS 2: Assign concept names from seed list (or create new ones)
# ===================================================================

def assign_concepts_to_statements(
    statements: List[dict],
    backend: "LLMBackend",
    seed_concepts: Optional[List[str]] = None,
    batch_size: int = 16,
    strict_seeds: bool = False,
    raise_on_json_failure: bool = False,
) -> List[Fact]:
    """
    Pass 2: Given raw statements, assign each to a concept name.

    Uses the seed concept list to anchor naming. The model can also
    propose NEW concept names if a statement doesn't fit any seed,
    but the seed list keeps most names consistent.

    batch_size: number of statements sent to the LLM in a single call (default 16).
    strict_seeds: when True, the model must pick from the seed list (no invention).
                  Use this with OpenRouter where the JSON schema enum is enforced
                  server-side. Removes the "create a new name" instruction and adds
                  an enum constraint to the output schema.
    """
    if not statements:
        return []

    seeds = seed_concepts if seed_concepts is not None else load_builtin_seeds()
    seed_block = "\n".join(f"  - {c}" for c in seeds)

    # Build the system message once — it contains the static instructions and
    # seed list, which are identical across all batches in this call.  Sending
    # it as a system message lets providers that support prompt caching
    # (OpenAI, Anthropic via OpenRouter, DeepSeek) serve it from cache on
    # every batch after the first, cutting input token cost significantly.
    if strict_seeds:
        p2_system_prompt = f"""Assign each statement to exactly ONE concept name from the approved list.

APPROVED concept names — you MUST use one of these, no exceptions:
{seed_block}

Rules:
- Every statement must be assigned to the single most relevant name from the list above.
- If a statement fits multiple concepts from the list, assign it to the more specific one.
- Do NOT invent new concept names.

Output ONLY a JSON array with one entry per statement:
[{{"index": 1, "concept": "Concept Name"}}, {{"index": 2, "concept": "Concept Name"}}]"""
    else:
        p2_system_prompt = f"""Assign each statement to exactly ONE concept name.

PREFERRED concept names (use one of these when the statement is about that topic):
{seed_block}

Rules:
- Use a preferred name from the list above whenever possible.
- If no preferred name fits, create a SHORT noun phrase (1-4 words).
- Use standard textbook terminology for new names.
- Do NOT use vague names like "Overview", "Impact", "Method", "Management".
- Every statement must get exactly one concept name.
- If a statement fits multiple concepts from the list, assign it to the more specific one.

Output ONLY a JSON array with one entry per statement:
[{{"index": 1, "concept": "Concept Name"}}, {{"index": 2, "concept": "Concept Name"}}]"""

    # JSON schema for structured output (enforced by OpenRouter, best-effort elsewhere).
    # When strict_seeds=True, add an enum constraint so OpenRouter enforces seed adherence
    # server-side — the model cannot output a concept not in the approved list.
    concept_property: dict = {
        "type": "string",
        "description": "The assigned concept name (1-4 word noun phrase)",
    }
    if strict_seeds and seeds:
        concept_property["enum"] = seeds

    concept_assignment_schema = {
        "name": "concept_assignments",
        "schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "The statement number (1-based)",
                    },
                    "concept": concept_property,
                },
                "required": ["index", "concept"],
                "additionalProperties": False,
            },
        },
    }

    all_facts: List[Fact] = []

    for i in range(0, len(statements), batch_size):
        batch = statements[i : i + batch_size]

        numbered_statements = "\n".join(
            f"  {j+1}. {s['statement']}" for j, s in enumerate(batch)
        )
        # User message: just the variable statements for this batch.
        prompt = f"Statements:\n{numbered_statements}"

        batch_num = i // batch_size + 1
        parsed = None
        _MAX_JSON_RETRIES = 2
        for attempt in range(_MAX_JSON_RETRIES):
            try:
                raw_content = backend.generate(
                    prompt, json_schema=concept_assignment_schema,
                    context=f"batch {batch_num}",
                    system_prompt=p2_system_prompt,
                )
            except Exception as exc:
                print(f"  [pass2] Batch {batch_num}: LLM call failed: {exc}")
                break
            parsed = _parse_json_array(raw_content)
            if isinstance(parsed, list):
                break
            reason = _classify_json_failure(raw_content)
            retry_msg = f", retrying ({attempt + 2}/{_MAX_JSON_RETRIES})..." if attempt + 1 < _MAX_JSON_RETRIES else ""
            _lbl = backend.last_used_label() if hasattr(backend, "last_used_label") else backend.label
            print(f"  [{_lbl}] JSON parse failed — {reason}{retry_msg}")

        if not isinstance(parsed, list):
            if raise_on_json_failure:
                from backend.base import LLMBackendError
                raise LLMBackendError(
                    f"JSON parse failed for batch {batch_num} after {_MAX_JSON_RETRIES} retries"
                )
            continue

        concept_map: dict[int, str] = {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            concept = item.get("concept", "")
            if isinstance(idx, int) and isinstance(concept, str) and concept.strip():
                concept_map[idx] = concept.strip()

        matched = 0
        for j, stmt in enumerate(batch):
            concept = concept_map.get(j + 1)
            if not concept:
                continue
            matched += 1
            # Preserve a human-readable source (filename) alongside the
            # chunk id by composing "filename::chunk_id" when available.
            chunk_id = stmt.get("chunk_id", "")
            source_name = stmt.get("source", "")
            composed_source = f"{source_name}::{chunk_id}" if source_name else chunk_id
            all_facts.append(
                Fact(
                    id=str(uuid.uuid4()),
                    concept=concept,
                    content=stmt["statement"],
                    source_chunk_id=composed_source,
                )
            )

        if matched == 0 and parsed:
            # Model returned JSON but none of it mapped — show what we got
            print(f"  [pass2] Batch {batch_num}: Parsed {len(parsed)} items but 0 matched expected format.")
            print(f"           First item: {parsed[0] if parsed else 'empty'}")

    return all_facts


# ===================================================================
# Legacy single-pass interfaces (kept for backward compatibility)
# ===================================================================

def extract_facts(
    chunk_text: str,
    chunk_id: str,
    backend: "LLMBackend",
) -> List[Fact]:
    """Legacy single-pass extraction. Prefer the two-pass pipeline."""
    prompt = f"""
    Extract atomic facts from the text.

    Rules:

    Output ONLY valid JSON (no prose, no markdown).
    JSON format: [{{"concept":"...","content":"..."}}]
    Use short canonical concept names (1-4 words), noun phrases only.
    Use standard textbook terms; prefer common names.
    Do not invent facts or concepts; skip unclear/underspecified items.
    Include all explicitly defined concepts, even if they appear minor.
    Use either acronym or full term (not both) for the same concept.
    Exclude meta labels (e.g., terminology, example, note).
    Exclude implementation-specific details (e.g., browsers, OS).
    Reject vague concept names ending with: objectives, impact, effects, goals, overview, status.
    Avoid malformed possessive fragments (e.g., "X s ..."); normalize to a clean noun phrase.
    Text:
    {chunk_text}
    """

    try:
        raw_content = backend.generate(prompt)
    except Exception:
        return []

    parsed_data = _parse_json_array(raw_content)
    if parsed_data is None or not isinstance(parsed_data, list):
        return []

    facts: List[Fact] = []
    for item in parsed_data:
        if not isinstance(item, dict):
            continue
        concept = item.get("concept")
        content = item.get("content")
        if not isinstance(concept, str) or not isinstance(content, str):
            continue
        facts.append(
            Fact(
                id=str(uuid.uuid4()),
                concept=concept,
                content=content,
                source_chunk_id=chunk_id,
            )
        )
    return facts


def extract_facts_batched(
    chunks: List["Chunk"],
    backend: "LLMBackend",
    batch_size: int = 3,
) -> List[Fact]:
    """Legacy batched extraction. Prefer the two-pass pipeline."""
    if not chunks:
        return []
    if batch_size < 1:
        batch_size = 1

    all_facts: List[Fact] = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        allowed_chunk_ids = {chunk.id for chunk in batch}

        combined_sections: List[str] = []
        for chunk in batch:
            combined_sections.append(f"[CHUNK_ID={chunk.id}]\n{chunk.text}")
        combined_text = "\n\n".join(combined_sections)

        prompt = f"""
        Extract atomic facts from the batched chunk texts.

        For each fact:
            - Assign a concept name that is SHORT and CANONICAL (1-4 words)
            - Use standard textbook terminology
            - Use noun-phrase style names (not sentence fragments)
            - Prefer commonly accepted names
            - DO NOT invent new concepts
            - Prefer canonical forms: output either acronym OR full term, not both
            - If the concept is underspecified or unclear, SKIP it
            - DO NOT include meta concepts like "terminology", "example", "note"
            - DO NOT include implementation details (e.g., browsers, OS, etc.)
            - DO NOT output vague labels ending with generic terms like
              "objectives", "impact", "effects", "goals", "overview", "status"
            - DO NOT output malformed possessive fragments like "X s ..."

        IMPORTANT:
            - Each fact MUST include the source_chunk_id from the chunk marker.
            - source_chunk_id MUST be exactly one of:
              {sorted(allowed_chunk_ids)}

        Return ONLY a JSON array:
        [
          {{"concept": "...", "content": "...", "source_chunk_id": "..."}}
        ]

        Batched Text:
        {combined_text}
        """

        try:
            raw_content = backend.generate(prompt)
        except Exception:
            continue

        parsed_data = _parse_json_array(raw_content)
        if not isinstance(parsed_data, list):
            continue

        for item in parsed_data:
            if not isinstance(item, dict):
                continue
            concept = item.get("concept")
            content = item.get("content")
            source_chunk_id = item.get("source_chunk_id")
            if not isinstance(concept, str) or not isinstance(content, str):
                continue
            if not isinstance(source_chunk_id, str) or source_chunk_id not in allowed_chunk_ids:
                continue
            all_facts.append(
                Fact(
                    id=str(uuid.uuid4()),
                    concept=concept,
                    content=content,
                    source_chunk_id=source_chunk_id,
                )
            )

    return all_facts