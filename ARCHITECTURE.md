# Architecture (v2)

## Overview

PDF-to-Wiki is a pipeline that converts PDF documents (textbooks, notes, slides) into interlinked Obsidian wiki pages. It uses a two-pass LLM extraction approach with a pluggable backend that supports local models (Ollama), cloud APIs (OpenRouter, Anthropic), and hybrid configurations.

## Pipeline Flow

```
PDF → Chunks → [Pass 1: Extract Statements] → [Pass 1.5: Derive Seeds] → [Pass 2: Assign Concepts] → Facts
  → Filter → Group → Normalize → Canonicalize → Merge → Cluster → Consolidate
  → [Render Pages] → Markdown Wiki Pages
```

## Data Model

```python
@dataclass
class Chunk:
    id: str          # UUID
    text: str        # 500-700 words of source text
    source: str      # Source filename
    chapter: str | None

@dataclass
class Fact:
    id: str              # UUID
    concept: str         # Canonical concept name (1-4 words)
    content: str         # The factual statement
    source_chunk_id: str # Traces back to source chunk
```

## Two-Pass Extraction

The pipeline splits LLM work into two simpler tasks:

**Pass 1 (Extract):** Ask the LLM to pull factual statements from text. No concept naming. This is easy for small models (8B) — just sentence extraction.

**Pass 1.5 (Derive Seeds):** After extraction, ask the Pass 2 backend to derive 20-30 domain-specific concept names from a sample of the extracted statements. This makes the pipeline domain-agnostic — no hardcoded terminology needed. Priority order: `--seeds FILE` > auto-derived > built-in fallback (`seeds/accounting.json`).

**Pass 2 (Assign):** Given the extracted statements, assign each to a concept name from the seed list. Can use a different (better) model than Pass 1. This is where hybrid mode shines — local extraction + API assignment.

**Why two passes:** Small models (llama3.1:8b) struggle when asked to simultaneously extract facts, name concepts, track chunk IDs, and output JSON. Splitting the task dramatically improves consistency.

## Backend Abstraction (`src/backend/`)

All LLM calls go through a pluggable backend interface:

- `base.py` — Abstract `LLMBackend` class and `BackendConfig` dataclass
- `openai_compat.py` — Handles Ollama, OpenRouter, Runpod, LM Studio, vLLM, Together AI
- `anthropic.py` — Handles the Anthropic Messages API
- `factory.py` — `create_backend()` and `create_pass_backends()` from env vars
- `config.py` — Env var loading, .env file support, API key masking

**OpenRouter features** (activated automatically when base URL is openrouter.ai):
- Structured outputs (`response_format` with JSON schema) — enforces valid JSON
- Response healing plugin — fixes malformed JSON
- Model fallbacks (`PASS2_FALLBACK_MODELS`) — automatic failover between models
- Retry with exponential backoff on 429 rate limits

**Key security rule:** API keys come only from env vars or `.env` file. Never hardcoded, never logged, never in repr().

## Transform Pipeline (`src/transform/`)

After extraction, facts go through deterministic (non-LLM) processing:

1. **filter.py** — Reject invalid concept names (years, countries, verbs, vague descriptors)
2. **grouping.py** — Group facts by normalized concept name
3. **normalize.py** — Deterministic rule-based normalization (title case, singularize, dedupe words)
4. **canonicalize.py** — LLM-assisted canonicalization with persistent cache
5. **merge.py** — Merge exact-token duplicates and strong-overlap concepts
6. **cluster.py** — Cluster related concepts (same head+tail tokens), respecting siblings and antonyms
7. **Consolidation** (in main.py) — One final LLM pass to catch remaining duplicates

**Antonym/sibling awareness:** `matching.py` prevents merging concepts that are intentionally distinct (e.g., FIFO vs LIFO, Periodic vs Perpetual, Inventory Fraud vs Inventory Shrinkage).

## Page Generation (`src/generate/`)

Split into focused modules:

- **classify.py** — Fact classification (definition/key_point/formula/caution/etc.) and definition scoring
- **util.py** — Text normalization, deduplication, formatting helpers
- **titles.py** — Acronym detection, title casing, `normalize_page_title()`
- **related.py** — Related concepts via chunk co-occurrence and token overlap
- **wiki_helpers.py** — Concept-type inference, wikilink injection, fact promotion
- **renderers.py** — Three renderers: standard, enhanced, wiki

**Wiki renderer features:**
- Concept-type templates: ratios get Formula + "What It Tells You", methods/systems get "How It Works"
- Type inference from fact content (domain-agnostic — no hardcoded accounting terms)
- `[[wikilinks]]` injected into body text, nesting-safe
- Related concepts from chunk co-occurrence (not just name similarity)
- Clean source attribution (hides UUID chunk IDs, shows filenames when available)

## Configuration

All config via environment variables (or `.env` file). See `.env.example` for full reference.

**Key variables:**
- `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_MODEL` — Global defaults
- `PASS1_*`, `PASS2_*` — Per-pass overrides for hybrid mode
- `PASS2_FALLBACK_MODELS` — Comma-separated fallback model list
- `TWO_PASS=1` — Enable two-pass extraction (recommended)
- `ENHANCED_PAGE_MODE=1` — Use wiki-style renderer

## Known Limitations

- **Source tracking:** Chunks use UUIDs, not page numbers. The Sources section on wiki pages is hidden until real source references are added to the chunker.
- **Seed concepts:** Pass 1.5 auto-derives domain-specific seeds from the extracted statements, making explicit seed files optional. Supply `--seeds FILE` (a JSON array of concept name strings) to skip auto-generation. Built-in fallback lives in `seeds/accounting.json`.
- **Local model quality:** 8B models produce adequate but imperfect extractions. The hybrid approach (local Pass 1 + API Pass 2) significantly improves quality.