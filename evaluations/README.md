# Evaluation fixtures

This directory contains the versioned corpus used to compare model-backed runs.
It is deliberately small, text-only, and free of credentials so it can run on a
developer machine or a future self-hosted CI runner.

`fixtures/v1/corpus.json` contains five representative source types: clean
academic prose, slide bullets, noisy OCR, diagram-adjacent text, and noise.
Each case declares expected concepts, accepted aliases, page-content terms, and
concepts that should not be created.

The deterministic scorer in `src/evaluation/scoring.py` measures contracts
rather than exact model wording: source-ID preservation, gold-concept recall,
false positives, forbidden/invalid concepts, page presence, required terms,
headings, and malformed wikilinks.

When the live benchmark runner is added, it should save the model identifier,
runtime details, fixture version, raw pipeline output, and this scorer's JSON
result together. Changing this corpus requires a new version directory; never
silently modify a released corpus, because that would invalidate comparisons.
