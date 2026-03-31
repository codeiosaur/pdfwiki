"""
Fact extraction from text chunks using a pluggable LLM backend.

Two-pass pipeline:
  Pass 1: Extract raw factual statements (no concept naming)
  Pass 2: Assign concept names from a seed list

Legacy single-pass functions are kept for backward compatibility.
"""

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

import json
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
# Seed concept list — canonical names the model should map facts to.
# Extend this list per-domain or load from a JSON/YAML file.
# ---------------------------------------------------------------------------
SEED_CONCEPTS: List[str] = [
    # Inventory methods & systems
    "First In First Out",
    "Last In First Out",
    "Weighted Average Cost",
    "Specific Identification",
    "Perpetual Inventory System",
    "Periodic Inventory System",
    "Cost of Goods Sold",
    "Cost Allocation",
    # Inventory measurement & reporting
    "Lower of Cost or Market",
    "Inventory Turnover Ratio",
    "Days Sales in Inventory",
    "Gross Profit",
    "Gross Margin",
    "Gross Profit Method",
    "Retail Inventory Method",
    "Beginning Inventory",
    "Ending Inventory",
    "Goods Available for Sale",
    # Inventory issues
    "Inventory Fraud",
    "Inventory Errors",
    "Inventory Shrinkage",
    "Inventory Obsolescence",
    "Consignment",
    # Shipping & ownership
    "FOB Shipping Point",
    "FOB Destination",
    # Technology
    "UPC Barcode",
    "Electronic Product Code",
    # Financial statements & ratios (cross-chapter)
    "Income Statement",
    "Balance Sheet",
    "Net Income",
    "Revenue Recognition",
    "Expense Recognition",
]


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _parse_json_array(raw_content: str):
    """Parse a JSON array from raw model output."""
    try:
        return json.loads(raw_content)
    except Exception:
        start = raw_content.find("[")
        end = raw_content.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(raw_content[start : end + 1])
        except Exception:
            return None


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
    batch_size: int = 2,
) -> List[dict]:
    """
    Pass 1 (batched): Extract raw factual statements from chunks.

    The model only extracts facts — it does NOT assign concept names.
    This is a much simpler task for small models and produces
    cleaner output.
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
- Each statement must be ONE complete, self-contained factual claim.
- Write clear sentences a student could study from.
- Skip worked examples with specific dollar amounts.
- Skip instructions ("calculate...", "determine...", "prepare...").
- Skip figure/table references and page numbers.
- Keep definitions, descriptions, comparisons, and formulas.
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

        try:
            raw_content = backend.generate(
                prompt, json_schema=statement_extraction_schema,
            )
        except Exception as exc:
            print(f"  [pass1] Batch {i//batch_size + 1}: LLM call failed: {exc}")
            continue

        parsed = _parse_json_array(raw_content)
        if not isinstance(parsed, list):
            preview = (raw_content[:200] + "...") if len(raw_content) > 200 else raw_content
            print(f"  [pass1] Batch {i//batch_size + 1}: Failed to parse JSON from response:")
            print(f"           {preview}")
            continue

        for item in parsed:
            if not isinstance(item, dict):
                continue
            statement = item.get("statement", "")
            source_chunk_id = item.get("source_chunk_id", "")
            if (
                isinstance(statement, str)
                and len(statement.strip()) > 10
                and isinstance(source_chunk_id, str)
                and source_chunk_id in allowed_chunk_ids
            ):
                all_statements.append({
                    "statement": statement.strip(),
                    "chunk_id": source_chunk_id,
                })

    return all_statements


# ===================================================================
# PASS 2: Assign concept names from seed list (or create new ones)
# ===================================================================

def assign_concepts_to_statements(
    statements: List[dict],
    backend: "LLMBackend",
    seed_concepts: Optional[List[str]] = None,
    batch_size: int = 12,
) -> List[Fact]:
    """
    Pass 2: Given raw statements, assign each to a concept name.

    Uses the seed concept list to anchor naming. The model can also
    propose NEW concept names if a statement doesn't fit any seed,
    but the seed list keeps most names consistent.
    """
    if not statements:
        return []

    seeds = seed_concepts if seed_concepts is not None else SEED_CONCEPTS
    seed_block = "\n".join(f"  - {c}" for c in seeds)

    all_facts: List[Fact] = []

    for i in range(0, len(statements), batch_size):
        batch = statements[i : i + batch_size]

        numbered_statements = "\n".join(
            f"  {j+1}. {s['statement']}" for j, s in enumerate(batch)
        )

        prompt = f"""Assign each statement below to exactly ONE concept name.

PREFERRED concept names (use one of these when the statement is about that topic):
{seed_block}

Rules:
- Use a preferred name from the list above whenever possible.
- If no preferred name fits, create a SHORT noun phrase (1-4 words).
- Use standard textbook terminology for new names.
- Do NOT use vague names like "Overview", "Impact", "Method", "Management".
- Every statement must get exactly one concept name.

Statements:
{numbered_statements}

Output ONLY a JSON array with one entry per statement:
[{{"index": 1, "concept": "Concept Name"}}, {{"index": 2, "concept": "Concept Name"}}]"""

        # JSON schema for structured output (enforced by OpenRouter, best-effort elsewhere)
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
                        "concept": {
                            "type": "string",
                            "description": "The assigned concept name (1-4 word noun phrase)",
                        },
                    },
                    "required": ["index", "concept"],
                    "additionalProperties": False,
                },
            },
        }

        try:
            raw_content = backend.generate(
                prompt, json_schema=concept_assignment_schema,
            )
        except Exception as exc:
            print(f"  [pass2] Batch {i//batch_size + 1}: LLM call failed: {exc}")
            continue

        parsed = _parse_json_array(raw_content)
        if not isinstance(parsed, list):
            # Show a preview of what we got back so the user can diagnose
            preview = (raw_content[:200] + "...") if len(raw_content) > 200 else raw_content
            print(f"  [pass2] Batch {i//batch_size + 1}: Failed to parse JSON from response:")
            print(f"           {preview}")
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
            all_facts.append(
                Fact(
                    id=str(uuid.uuid4()),
                    concept=concept,
                    content=stmt["statement"],
                    source_chunk_id=stmt["chunk_id"],
                )
            )

        if matched == 0 and parsed:
            # Model returned JSON but none of it mapped — show what we got
            print(f"  [pass2] Batch {i//batch_size + 1}: Parsed {len(parsed)} items but 0 matched expected format.")
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