# Rules

## Core Principle

This project is a pipeline. Each module does ONE thing.
Do not combine multiple stages in a single file or function.

---

## Pipeline Stages

1. **Ingestion** (`ingest/pdf_loader.py`): PDF → Chunks
2. **Extraction** (`extract/fact_extractor.py`): Chunks → Raw Statements → Facts (two-pass)
3. **Transform** (`transform/`): Facts → Filtered, grouped, normalized, merged concept groups
4. **Generation** (`generate/`): Concept groups → Markdown wiki pages
5. **Orchestration** (`main.py`): Wires the stages together

---

## LLM Usage

**Allowed (via backend abstraction):**
- Pass 1: Fact/statement extraction
- Pass 2: Concept name assignment
- Canonicalization (with cache to avoid repeat calls)
- Post-extraction consolidation (merge detection)

**Not allowed:**
- Deduplication logic (use deterministic matching)
- Control flow decisions
- System orchestration
- Anything that can be done with regex or string comparison

---

## Backend Rules

- All LLM calls go through the `LLMBackend` interface — never call openai/anthropic directly
- API keys come ONLY from environment variables or `.env` — never hardcoded
- Keys must never appear in logs, repr(), error messages, or git history
- The backend auto-detects OpenRouter and enables structured outputs, response healing, and fallbacks
- Local endpoints (Ollama, LM Studio) need no API key

---

## Code Constraints

- Target ~300 lines per file (hard max ~500 for renderers.py)
- Target ~40 lines per function
- Prefer simple functions over classes unless state management is needed
- One responsibility per module — if a file does two things, split it

---

## Domain Agnosticism

- `generate/` must NOT contain domain-specific terms (no "inventory", "FIFO", "accounting")
- Concept-type classification infers from fact content, not hardcoded concept names
- Related concepts use chunk co-occurrence, not hardcoded groups
- Acronym maps are built dynamically from the data
- The seed concept list in `fact_extractor.py` is the ONE place domain knowledge lives

---

## Anti-Patterns (DO NOT DO)

- Large "workflow" functions that do multiple pipeline stages
- Passing giant dependency objects between stages
- Hardcoding domain terms in the renderer or transform logic
- Silently swallowing errors (`except: continue` without logging)
- Overuse of configuration systems — env vars and `.env` are enough
- Complex graph or agent systems
- Premature optimization

---

## Testing Mindset

Each stage should be testable independently with mock data:

- Ingestion → returns chunks from a PDF
- Extraction → returns facts given a MockBackend with canned responses
- Transform → returns grouped/merged concepts given a list of Facts
- Generation → returns markdown given grouped facts
- Backend → config validation, key masking, factory logic (all deterministic)

Test the deterministic logic, mock the LLM calls.

---

## Git Conventions

- One feature branch per feature, branched from `v2`
- Merge into `v2` when tested
- Commit messages: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`
- `.env` is in `.gitignore` — never committed
- `canonical_cache.json` and `evaluation_metrics.json` are in `.gitignore`