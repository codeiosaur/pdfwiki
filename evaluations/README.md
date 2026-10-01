# Evaluation fixtures

This directory contains the versioned corpus used to compare model-backed runs.
It is deliberately small, text-only, and free of credentials so it can run on a
developer machine or a future self-hosted CI runner.

`fixtures/v1/corpus.json` contains five representative source types: clean
academic prose, slide bullets, noisy OCR, diagram-adjacent text, and noise.
Each case declares expected concepts, accepted aliases, page-content terms, and
concepts that should not be created.

The deterministic scorer in `src/evaluation/scoring.py` measures contracts
rather than exact model wording: exact source-ID preservation, gold-concept
recall, explicitly forbidden concepts, page presence, required terms, headings,
and malformed wikilinks. An output concept absent from the short gold list is
*unreviewed*, not automatically a false positive. An explicitly marked
`noise_only` case requires no output at all. Metrics with no applicable
denominator are `null`; suite averages include only applicable cases and
report their denominator under `scored_case_counts`. A heading check is not
full Markdown validation.

When the live benchmark runner is added, it should save the model identifier,
runtime details, fixture version, raw pipeline output, and this scorer's JSON
result together. New benchmark content requires a new version directory; never
silently modify a released corpus, because that would invalidate comparisons.
The v1 noise case was marked `noise_only` as a rubric clarification; its source
text and expected concepts are unchanged.
