"""
AI client module.
Toggle between Ollama (local/free) and Claude API (cloud/paid) by changing PROVIDER below.

Model routing:
- CLAUDE_FAST_MODEL: used for index, flashcards, cheatsheet (Haiku — cheap)
- CLAUDE_QUALITY_MODEL: used for wiki page generation (Sonnet — best output)
- Ollama always uses OLLAMA_MODEL regardless of task
"""

import os
from dotenv import load_dotenv
import anthropic
from openai import OpenAI

load_dotenv()

# --- CONFIG ---
PROVIDER = os.environ.get("PDF_TO_NOTES_PROVIDER", "claude").strip().lower()

# Claude models — swap either string to change quality/cost tradeoff
CLAUDE_QUALITY_MODEL = os.environ.get(
    "PDF_TO_NOTES_CLAUDE_QUALITY_MODEL",
    "claude-sonnet-4-20250514",
)  # wiki pages — Sonnet 4 (cheaper than 4.6, still high quality)
CLAUDE_FAST_MODEL = os.environ.get(
    "PDF_TO_NOTES_CLAUDE_FAST_MODEL",
    "claude-haiku-4-5-20251001",
)  # index, flashcards, cheatsheet

OLLAMA_BASE_URL = os.environ.get("PDF_TO_NOTES_OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.environ.get("PDF_TO_NOTES_OLLAMA_MODEL", "llama3.1:8b")

CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


def query(
    prompt: str,
    system: str = "",
    max_tokens: int = 4096,
    quality: bool = False   # True = Sonnet, False = Haiku
) -> str:
    """
    Send a prompt to the AI and return the response text.

    quality=True  → use CLAUDE_QUALITY_MODEL (Sonnet) — for wiki pages
    quality=False → use CLAUDE_FAST_MODEL (Haiku)    — for everything else
    """
    if PROVIDER == "claude":
        model = CLAUDE_QUALITY_MODEL if quality else CLAUDE_FAST_MODEL
        return _query_claude(prompt, system, max_tokens, model)
    if PROVIDER == "ollama":
        return _query_ollama(prompt, system, max_tokens)

    raise ValueError(
        f"Unsupported PDF_TO_NOTES_PROVIDER: {PROVIDER}. "
        "Expected 'claude' or 'ollama'."
    )


def _query_claude(prompt: str, system: str, max_tokens: int, model: str) -> str:
    if not CLAUDE_API_KEY:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. "
            "Add it to your .env file: ANTHROPIC_API_KEY=sk-ant-..."
        )

    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    return response.content[0].text.strip()


def _query_ollama(prompt: str, system: str, max_tokens: int) -> str:
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=messages,
        max_tokens=max_tokens,
    )
    
    if not response.choices[0].message.content:
        return ""
    return response.choices[0].message.content.strip()