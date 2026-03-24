from dataclasses import dataclass
from typing import List, TYPE_CHECKING

import json
import os
import uuid
import openai

if TYPE_CHECKING:
    from ingest.pdf_loader import Chunk


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


class OllamaClient:
    def __init__(self) -> None:
        # OpenAI-compatible client pointed at Ollama.
        self._client = openai.OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    def generate(self, model: str, prompt: str, max_tokens: int = 800):
        return self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
        )


ollama_client = OllamaClient()

@dataclass
class Fact:
    id: str
    concept: str
    content: str
    source_chunk_id: str


def _parse_json_array(raw_content: str):
    """
    Parse a JSON array from raw model output.
    Handles direct JSON and common markdown/prose wrappers.
    """
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


def extract_facts(chunk_text: str, chunk_id: str) -> List[Fact]:
    """
    Extract atomic facts from a chunk of text using an LLM.
    Returns a list of Fact objects.
    """

    # Step 1: Ask the LLM for atomic facts in strict JSON format.
    prompt = f"""
    Extract atomic facts from the text.

    Rules:

    Output ONLY valid JSON (no prose, no markdown).
    JSON format: [{"concept":"...","content":"..."}]
    Use short canonical concept names (1–4 words), noun phrases only.
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
        response = ollama_client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            max_tokens=500,
        )
        raw_content = response.choices[0].message.content or ""
    except Exception:
        return []

    # Step 2: Parse JSON safely. If invalid, return an empty list.
    parsed_data = _parse_json_array(raw_content)
    if parsed_data is None:
        return []

    if not isinstance(parsed_data, list):
        return []

    # Step 3: Convert valid JSON items into Fact objects.
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

    # Step 4: Return facts (or [] if no valid items were provided).
    return facts


def extract_facts_batched(chunks: List["Chunk"], batch_size: int = 3) -> List[Fact]:
    """
    Extract atomic facts from multiple chunks while reducing LLM calls.

    Chunks are grouped into small batches (default 3) and sent in one prompt.
    The model must return source_chunk_id for each fact so traceability is preserved.
    """
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
            response = ollama_client.generate(
                model=OLLAMA_MODEL,
                prompt=prompt,
                max_tokens=900,
            )
            raw_content = response.choices[0].message.content or ""
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