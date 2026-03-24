from typing import Optional
from pathlib import Path
from extract.fact_extractor import ollama_client, OLLAMA_MODEL

import json
import re

CANONICAL_CACHE_PATH = Path(__file__).with_name("canonical_cache.json")

def load_canonical_cache() -> dict[str, Optional[str]]:
    if not CANONICAL_CACHE_PATH.exists():
        return {}

    try:
        with CANONICAL_CACHE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    cache: dict[str, Optional[str]] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        cache[key] = value if isinstance(value, str) else None
    return cache


def save_canonical_cache(cache: dict[str, Optional[str]]) -> None:
    with CANONICAL_CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def canonicalize_concepts(concepts: list[str]) -> dict[str, Optional[str]]:
    if not concepts:
        return {}

    # Step 0: Load cache and only request uncached concepts.
    cache = load_canonical_cache()
    missing = [concept for concept in concepts if concept not in cache]

    if not missing:
        return {name: cache.get(name) for name in concepts}

    # Step 1: Send uncached concepts in one batched prompt.
    prompt = (f"""
    You are given concept names extracted from academic material.

    Your task:
    - Fix spelling errors
    - Fix possessives (e.g., Shannon, Euler, Kerckhoffs)
    - Normalize to standard terminology
    - Remove invalid, vague, or underspecified concepts (return null)
        - Keep concept names as clean noun phrases (1-4 words)

    Rules:
    - Do NOT merge different concepts
    - Do NOT invent new concepts
    - Preserve meaning exactly
    - Keep names concise (1–4 words)
        - If a concept is malformed due to dropped apostrophes (e.g., "X S Y"),
            rewrite it to a clean canonical phrase
        - If a concept ends with vague generic labels
            (objectives, impact, effects, goals, overview, status),
            return a clearer canonical name only when clearly supported;
            otherwise return null
    - Normalize to ONE canonical form:
    - acronym OR expansion, never both
    - Remove vague trailing words:
    support, impact, overview, objectives, goals, status, effects
    (if meaning remains clear)

    Return ONLY valid JSON mapping original → fixed (or null):

    {{
      "Concept A": "Fixed Name",
      "Concept B": null
    }}

    Concepts:
    {chr(10).join("- " + c for c in missing)}
    """)

    try:
        response = ollama_client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            max_tokens=600,
        )
        raw_content = response.choices[0].message.content or ""
    except Exception:
        for name in missing:
            cache[name] = None
        save_canonical_cache(cache)
        return {name: cache.get(name) for name in concepts}

    # Step 2: Parse JSON safely.
    try:
        parsed = json.loads(raw_content)
    except Exception:
        start = raw_content.find("{")
        end = raw_content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            for name in missing:
                cache[name] = None
            save_canonical_cache(cache)
            return {name: cache.get(name) for name in concepts}
        try:
            parsed = json.loads(raw_content[start : end + 1])
        except Exception:
            for name in missing:
                cache[name] = None
            save_canonical_cache(cache)
            return {name: cache.get(name) for name in concepts}

    if not isinstance(parsed, dict):
        for name in missing:
            cache[name] = None
        save_canonical_cache(cache)
        return {name: cache.get(name) for name in concepts}

    # Step 3: Update cache with LLM results and save.
    for name in missing:
        value = parsed.get(name)
        cache[name] = value if isinstance(value, str) else None

    save_canonical_cache(cache)
    return {name: cache.get(name) for name in concepts}


def needs_canonicalization(concept: str) -> bool:
    # Rule 1: single-letter token (e.g., "S").
    if any(len(token) == 1 for token in re.findall(r"[A-Za-z]+", concept)):
        return True

    # Rule 2: unusual spacing or punctuation artifacts.
    if concept != concept.strip():
        return True
    if re.search(r"\s{2,}", concept):
        return True
    if re.search(r"[\-_/]{2,}|[()]{2,}|[,:;.]\s*[,:;.]", concept):
        return True

    # Rule 3: repeated words (case-insensitive).
    words = re.findall(r"[A-Za-z]+", concept.lower())
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            return True

    # Rule 4: mixed casing inside a word (e.g., eCb, iNd).
    for token in re.findall(r"[A-Za-z]+", concept):
        if any(c.islower() for c in token) and any(c.isupper() for c in token):
            if not token.istitle():
                return True

    return False