# PDF → Obsidian Wiki Generator

Turn any academic PDF into a set of interlinked Obsidian study notes — automatically.

Feed it your course notes, and it produces wiki-style concept pages with definitions, key points, formulas, wikilinks, and related concepts. Drop the output into your Obsidian vault and start studying.

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/pdf-to-wiki.git
cd pdf-to-wiki
pip install -r requirements.txt
```

**With a local model (free, private, no API key needed):**

```bash
# Install Ollama (https://ollama.com), then:
ollama pull llama3.1:8b
python src/main.py your_textbook.pdf
```

**With a cloud model (better quality):**

```bash
cp .env.example .env
# Edit .env with your OpenRouter API key (free tier available)
python src/main.py your_textbook.pdf
```

Output lands in `output/` as markdown files ready for Obsidian.

## What You Get

Each concept becomes a wiki page with:

- **Definition** — what the concept is
- **How It Works** / **Formula** / **What It Tells You** — template chosen automatically based on concept type
- **Key Points** — additional facts extracted from the source
- **Cautions** — common pitfalls or misconceptions
- **Related Concepts** — `[[wikilinks]]` to other pages, based on source co-occurrence
- **Wikilinks in body text** — concept mentions are automatically linked

Example output for an accounting textbook chapter:

```markdown
# First in First Out

**First in First Out**: The FIFO costing assumption tracks inventory items
based on segments or lots of goods, in the order they were acquired.

---

## Definition

The first-in, first-out method (FIFO) assumes that the oldest goods are sold first.

---

## How It Works

- The FIFO costing assumption tracks inventory items in the order they were acquired.
- Using FIFO produces higher [[Ending Inventory]] balances than LIFO in inflationary times.
- The IASB chose to eliminate LIFO from IFRS, arguing FIFO more closely matches goods flow.

---

## Related Concepts
- [[Last in First Out]]
- [[Cost of Goods Sold]]
- [[Lower of Cost or Market]]
- [[Perpetual Inventory]]
```

## How It Works

The pipeline uses a **two-pass LLM approach**:

1. **Pass 1 — Extract:** Pull factual statements from the PDF. This is a simple task that runs well on local 8B models.
2. **Pass 2 — Assign:** Map each statement to a canonical concept name. This can use a better model (cloud API) for higher quality.

After extraction, a deterministic transform pipeline filters, normalizes, deduplicates, and clusters the concepts — no LLM needed for these steps.

```
PDF → Chunks → [Pass 1: Extract] → [Pass 2: Name] → Facts
  → Filter → Normalize → Merge → Cluster → Render → Wiki Pages
```

## Configuration

### Local only (default)

Works out of the box with Ollama. No API key, no cloud, fully private.

### Hybrid mode (recommended for quality)

Run extraction locally, send concept assignment to a cloud model:

```env
PASS1_PROVIDER=openai_compat
PASS1_BASE_URL=http://localhost:11434/v1
PASS1_MODEL=llama3.1:8b

PASS2_PROVIDER=openai_compat
PASS2_BASE_URL=https://openrouter.ai/api/v1
PASS2_MODEL=nvidia/nemotron-3-super-120b-a12b:free
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_ZDR=1  # Optional: request Zero Data Retention on OpenRouter calls

PASS2_FALLBACK_MODELS=meta-llama/llama-4-maverick:free,qwen/qwen3-coder:free
```

Recommended free-tier OpenRouter slugs (examples)
------------------------------------------------
These example slugs are commonly available via OpenRouter and can be used
with `PASS2_MODEL` for Pass 2. Availability and names change over time —
verify the slugs in your OpenRouter dashboard before use.

- `meta-llama/llama-3.3-70b-instruct:free`
- `nvidia/nemotron-3-super-120b-a12b:free`
- `qwen/qwen3-coder:free`
- `nousresearch/hermes-3-llama-3.1-405b:free`

Request Zero Data Retention by setting `OPENROUTER_ZDR=1` and `PASS2_ZDR=1`.

### Cloud only

Point both passes at a cloud provider for maximum quality:

```env
LLM_PROVIDER=openai_compat
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=nvidia/nemotron-3-super-120b-a12b:free
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_ZDR=1  # Optional: request Zero Data Retention on OpenRouter calls
```

See [.env.example](.env.example) for all options.

## Supported Backends

Any OpenAI-compatible endpoint works:

| Provider | Setup | Cost |
|---|---|---|
| **Ollama** (local) | `ollama pull llama3.1:8b` | Free |
| **LM Studio** (local) | Download model in app | Free |
| **OpenRouter** | Get API key at openrouter.ai | Free tier available |
| **Anthropic** | Get API key at console.anthropic.com | Paid |
| **Runpod** (self-hosted) | Deploy model on Runpod | Pay for GPU |
| **vLLM / Together AI** | Point `LLM_BASE_URL` at endpoint | Varies |

## Custom Seed Concepts

By default, the pipeline auto-generates concept names from your document. For better results in a specific domain, provide a seed file:

```bash
python src/main.py textbook.pdf --seeds seeds/accounting.json
```

Seed files are simple JSON arrays of concept names:

```json
["Supply and Demand", "Elasticity", "Market Equilibrium", "Consumer Surplus"]
```

The model uses these as preferred names but can still create new ones for concepts not in the list.

## Project Structure

```
src/
  main.py              # Pipeline orchestration + CLI
  backend/             # Pluggable LLM backend (Ollama, OpenRouter, Anthropic, etc.)
  ingest/              # PDF → text chunks
  extract/             # Two-pass fact extraction
  transform/           # Deterministic concept grouping, normalization, deduplication
  generate/            # Wiki page rendering with templates and wikilinks
tests/                 # pytest suite
.env.example           # All configuration options
ARCHITECTURE.md        # Detailed technical architecture
RULES.md               # Code conventions
```

## Requirements

- Python 3.12+
- For local mode: [Ollama](https://ollama.com) with a downloaded model
- For cloud mode: An API key from [OpenRouter](https://openrouter.ai) (free tier available) or another provider

## Design Philosophy

- **Pipeline, not agent.** Each stage does one thing. No complex graphs or orchestration.
- **LLMs where necessary, deterministic logic everywhere else.** Extraction and naming use LLMs. Filtering, merging, clustering, and rendering are all deterministic.
- **Domain-agnostic rendering.** The page generator infers concept types (ratio, method, system) from content, not hardcoded terms. Works for any subject.
- **Privacy-respecting defaults.** Local-first by default. Cloud is opt-in.

## Status

**v2.0** — Production-ready pipeline with:
- Two-pass extraction (local + cloud hybrid)
- Pluggable LLM backends (Ollama, OpenRouter, Anthropic, any OpenAI-compatible)
- Structured JSON outputs with OpenRouter
- Wiki-style pages with templates, wikilinks, and clean formatting
- Auto-generated or user-provided seed concepts
- CLI interface

## License

This project is licensed under the AGPL-3.0 License.

Copyright (C) 2026 Alexander Nyczkowski

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/> or <https://github.com/codeiosaur/pdfwiki>.

