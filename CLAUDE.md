# CLAUDE.md

## Project

PDF-to-Wiki: converts PDF documents into interlinked Obsidian wiki pages using a two-pass LLM pipeline.

## Quick Reference

- **Run:** `python3 src/main.py` (reads config from `.env`)
- **Test:** `pytest tests/ -v`
- **Source:** `src/` with packages: `backend/`, `extract/`, `transform/`, `generate/`, `ingest/`
- **Config:** `.env` file (copy from `.env.example`). See `ARCHITECTURE.md` for all variables.

## Key Architecture Decisions

1. **Two-pass extraction:** Pass 1 extracts raw statements (simple task for local 8B models). Pass 2 assigns concept names from a seed list (can use a better model via API). This split is fundamental — don't merge them back.

2. **Pluggable LLM backend:** All LLM calls go through `backend.LLMBackend`. Supports Ollama, OpenRouter, Anthropic, Runpod, any OpenAI-compatible endpoint. Config is env-var driven. API keys NEVER hardcoded or logged.

3. **Hybrid mode:** Pass 1 and Pass 2 can use different backends. Set `PASS1_*` and `PASS2_*` env vars independently. Typical setup: local Ollama for Pass 1, OpenRouter for Pass 2.

4. **OpenRouter structured outputs:** When the backend detects OpenRouter, it automatically enables JSON schema enforcement, response healing, and model fallbacks. This happens transparently — callers don't need to know.

5. **Domain agnosticism in the renderer:** `generate/` never contains domain-specific terms. Concept types (ratio/method/system) are inferred from fact content. Related concepts use chunk co-occurrence. The seed list in `fact_extractor.py` is the only place domain knowledge lives.

6. **Antonym/sibling awareness:** `transform/matching.py` maintains antonym pairs that prevent merging distinct concepts (FIFO/LIFO, periodic/perpetual, etc.). The clustering and merge logic respects these.

## File Layout

```
src/
  main.py                  # Pipeline orchestration
  backend/                 # LLM backend abstraction
    base.py                #   Abstract interface + config dataclass
    openai_compat.py       #   Ollama, OpenRouter, Runpod, etc.
    anthropic.py           #   Anthropic API
    factory.py             #   create_backend() / create_pass_backends()
    config.py              #   Env var loading, .env support, key masking
  extract/
    fact_extractor.py      # Two-pass extraction + seed concept list
  transform/
    filter.py              # Concept name validation
    grouping.py            # Group facts by concept
    normalize.py           # Deterministic name normalization
    canonicalize.py        # LLM-assisted canonicalization (cached)
    merge.py               # Merge similar concepts
    cluster.py             # Cluster related concepts
    matching.py            # Token matching, siblings, antonyms
  generate/
    classify.py            # Fact classification + definition selection
    util.py                # Text helpers
    titles.py              # Title formatting + acronym map
    related.py             # Related concept discovery + citations
    wiki_helpers.py        # Wikilink injection, type inference
    renderers.py           # Three page renderers (standard/enhanced/wiki)
  ingest/
    pdf_loader.py          # PDF → Chunks
tests/
  conftest.py              # MockBackend + fixtures
  test_backend.py          # Backend config/masking/factory tests
  test_filter.py           # Concept name validation tests
.env.example               # All config options documented
ARCHITECTURE.md            # Detailed architecture doc
RULES.md                   # Code conventions and constraints
```

## Current Branch Structure

- `main` — original V1 (archived)
- `v2` — current stable V2
- Feature branches: `feature/v2-*` branched from `v2`

## What NOT to Do

- Don't merge the two extraction passes back into one
- Don't hardcode domain terms in `generate/`
- Don't call LLM APIs directly — always go through the backend
- Don't log or print API keys, even partially
- Don't silently swallow errors without diagnostic output