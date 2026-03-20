from dataclasses import dataclass
from typing import List
import json
import os
import uuid
import openai


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


def extract_facts(chunk_text: str, chunk_id: str) -> List[Fact]:
    """
    Extract atomic facts from a chunk of text using an LLM.
    Returns a list of Fact objects.
    """

    # Step 1: Ask the LLM for atomic facts as a strict JSON array.
    prompt = (
        "Analyze the chunk text and extract atomic facts. "
        "Return ONLY a JSON array. "
        "Each item must be an object with exactly two string keys: "
        '"concept" and "content". '
        "Do not include markdown, code fences, or extra text.\n\n"
        f"TEXT:\n{chunk_text}"
    )

    try:
        response = ollama_client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            max_tokens=800,
        )
        raw_content = response.choices[0].message.content or ""
    except Exception:
        return []

    # Step 2: Parse JSON safely. If invalid, return an empty list.
    try:
        parsed = json.loads(raw_content)
        if not isinstance(parsed, list):
            return []
    except Exception:
        return []

    # Step 3: Convert valid JSON items into Fact objects.
    facts: List[Fact] = []
    for item in parsed:
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