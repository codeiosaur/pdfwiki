#!/usr/bin/env python3
"""
Compare quality of synthesized pages by different backends.

Usage:
    python scripts/compare_quality.py <pdf_path> <vault_dir> --backend <backend_name>

This script:
1. Extracts facts from the PDF using the specified backend
2. Groups them by concept (same as pipeline)
3. Loads synthesized pages from the vault
4. Compares pages by different backends on coverage and hallucination risk
5. Prints a console report

Example:
    python scripts/compare_quality.py sample.pdf output/vault-phi-gemma --backend local-extract
"""

import sys
from pathlib import Path
from collections import defaultdict
import re

# Add src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingest.pdf_loader import load_pdf_chunks
from extract.fact_extractor import extract_raw_statements_batched, assign_concepts_to_statements
from transform.grouping import group_facts_by_concept
from transform.normalize import normalize_group_keys
from backend.factory import create_pass_backends_from_config, backends_config_path
from generate.titles import normalize_page_title


def extract_key_terms(text: str) -> set[str]:
    """Extract lowercase words from text for keyword matching."""
    # Simple: split on whitespace/punctuation, filter short words
    words = re.findall(r'\b[a-z]+\b', text.lower())
    # Filter out common stop words and very short words
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'is', 'are', 'was', 'were', 'be', 'been', 'by', 'this', 'that',
        'it', 'as', 'from', 'with', 'have', 'has', 'can', 'will', 'may'
    }
    return {w for w in words if len(w) > 2 and w not in stopwords}


def fuzzy_match_concepts(concept_key: str, page_title: str, threshold: float = 0.5) -> bool:
    """
    Fuzzy match between concept key and page title.
    Returns True if they likely refer to the same concept.
    """
    # Normalize both
    concept_norm = normalize_page_title(concept_key).lower().strip()
    page_norm = page_title.lower().strip()

    # Exact match
    if concept_norm == page_norm:
        return True

    # Word overlap: if at least 70% of words in the shorter string match the longer
    concept_words = set(concept_norm.split())
    page_words = set(page_norm.split())

    if not concept_words or not page_words:
        return False

    overlap = len(concept_words & page_words)
    min_len = min(len(concept_words), len(page_words))

    # If majority of words overlap, consider it a match
    if min_len > 0 and overlap / min_len >= threshold:
        return True

    return False


def measure_structure_quality(page_text: str) -> dict:
    """Measure structural quality metrics (don't require source facts)."""
    metrics = {}

    # Count sections (## headers)
    sections = len(re.findall(r'^##\s+', page_text, re.MULTILINE))
    metrics["section_count"] = sections

    # Count wikilinks [[...]]
    wikilinks = len(re.findall(r'\[\[.+?\]\]', page_text))
    metrics["wikilink_count"] = wikilinks

    # Count code blocks
    code_blocks = len(re.findall(r'```', page_text))
    metrics["code_block_count"] = code_blocks // 2  # pairs of backticks

    # Count lists (- or * at line start)
    lists = len(re.findall(r'^\s*[-*]\s+', page_text, re.MULTILINE))
    metrics["list_count"] = lists

    # Estimate prose vs lists
    total_lines = len(page_text.split('\n'))
    metrics["list_line_ratio"] = lists / total_lines if total_lines > 0 else 0

    # Page length
    metrics["page_length"] = len(page_text)

    # Word count
    word_count = len(page_text.split())
    metrics["word_count"] = word_count

    # Avg words per section
    metrics["words_per_section"] = word_count / sections if sections > 0 else word_count

    return metrics


def measure_coverage(fact_contents: list[str], page_text: str) -> float:
    """
    Measure what % of source facts are covered in the page.
    Returns 0-1 (0 = no coverage, 1 = all facts mentioned).
    """
    if not fact_contents:
        return 0.0

    page_terms = extract_key_terms(page_text)
    covered = 0

    for fact in fact_contents:
        fact_terms = extract_key_terms(fact)
        # Fact is "covered" if at least 50% of its key terms appear in page
        if fact_terms:
            overlap = len(page_terms & fact_terms)
            if overlap / len(fact_terms) >= 0.5:
                covered += 1

    return covered / len(fact_contents)


def estimate_hallucination_risk(page_text: str, fact_contents: list[str], coverage: float) -> str:
    """
    Estimate hallucination risk based on coverage and suspicious patterns.
    Returns 'LOW', 'MEDIUM', or 'HIGH'.

    Key insight: if the page well-covers source facts, elaborations are less risky.
    Only flag HIGH risk when coverage is low AND suspicious patterns exist.
    """
    fact_text = " ".join(fact_contents).lower()
    page_lower = page_text.lower()

    # Primary signal: coverage
    # High coverage = page is grounded in source facts
    if coverage > 0.80:
        base_risk = "LOW"
    elif coverage > 0.60:
        base_risk = "MEDIUM"
    else:
        base_risk = "HIGH"

    # Secondary signal: count actual red flags (not elaborations)
    red_flags = 0

    # Red flag 1: Contradictions (specific claims opposite to facts)
    # E.g., if facts say "X increases" but page says "X decreases"
    contradictions = [
        ("increases", "decreases"),
        ("higher", "lower"),
        ("positive", "negative"),
        ("always", "never"),
        ("required", "optional"),
    ]
    for claim1, claim2 in contradictions:
        has_claim1_fact = claim1 in fact_text
        has_claim2_fact = claim2 in fact_text
        has_claim1_page = claim1 in page_lower
        has_claim2_page = claim2 in page_lower
        # If facts assert one, but page asserts the opposite, that's bad
        if has_claim1_fact and has_claim2_page and not has_claim2_fact:
            red_flags += 1
        if has_claim2_fact and has_claim1_page and not has_claim1_fact:
            red_flags += 1

    # Red flag 2: Vague citations without grounding
    # Only flag if not mentioned in facts AND seems important
    vague_citations = re.findall(r'(some research|many studies|experts agree)', page_lower)
    if vague_citations and coverage < 0.70:
        red_flags += 1

    # Red flag 3: Unsourced specific dates/statistics (only if coverage is low)
    if coverage < 0.60:
        # Only penalize if page has many numbers but facts have few
        page_numbers = len(re.findall(r'\d{4}|\d+%', page_text))
        fact_numbers = len(re.findall(r'\d{4}|\d+%', fact_text))
        if page_numbers > fact_numbers * 2:
            red_flags += 1

    # Adjust base risk based on red flags
    if red_flags >= 2:
        return "HIGH"
    elif red_flags >= 1 and base_risk == "HIGH":
        return "HIGH"
    elif red_flags >= 1:
        return "MEDIUM"
    else:
        return base_risk


def load_vault_pages(vault_dir: Path) -> dict[str, dict[str, str]]:
    """
    Load pages from vault, keyed by title and then by backend.
    Returns {title: {backend_label: page_text}}.
    """
    pages_by_title = defaultdict(dict)

    for md_file in vault_dir.glob("*.md"):
        content = md_file.read_text(encoding="utf-8")

        # Extract title from frontmatter or first # heading
        title = None
        if content.startswith("---"):
            # Has frontmatter
            parts = content.split("---", 2)
            body = parts[2] if len(parts) > 2 else ""
        else:
            body = content

        # Extract title from first # heading
        match = re.search(r'^#\s+(.+)$', body, re.MULTILINE)
        if match:
            title = match.group(1).strip()

        if not title:
            continue

        # Extract backend from frontmatter generated_by_backend
        backend_label = None
        if content.startswith("---"):
            fm_match = re.search(r'generated_by_backend:\s*(\S+)', content)
            if fm_match:
                backend_label = fm_match.group(1).strip()

        pages_by_title[title][backend_label or "unknown"] = content

    return pages_by_title


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare page quality from different backends in a vault"
    )
    parser.add_argument("vault", help="Vault directory with synthesized pages")
    parser.add_argument("--pdf", help="PDF file to extract facts from (optional, for coverage metrics)")
    parser.add_argument("--backend", help="Backend to use for extraction (from backends.yaml)")
    parser.add_argument("--heuristics-only", action="store_true",
                       help="Skip PDF extraction, use only structural metrics")

    args = parser.parse_args()

    vault_dir = Path(args.vault)

    # Validate inputs
    if not vault_dir.exists():
        print(f"Error: Vault directory not found: {vault_dir}")
        sys.exit(1)

    print(f"\n=== QUALITY COMPARISON REPORT ===")
    print(f"Vault: {vault_dir}")

    # Extract facts from PDF (optional)
    final_grouped = {}
    if args.heuristics_only:
        print("Mode: Heuristics-only (structural metrics)")
        print()
    else:
        if not args.pdf or not args.backend:
            print("Error: --pdf and --backend required unless using --heuristics-only")
            sys.exit(1)

        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"Error: PDF not found: {pdf_path}")
            sys.exit(1)

        print(f"PDF: {pdf_path}")
        print(f"Extraction Backend: {args.backend}")
        print()

        # Load backend config
        config_path = backends_config_path()
        if not config_path:
            print("Error: backends.yaml not found")
            sys.exit(1)

        # Create extraction backend
        try:
            pass1_backend, pass2_backend, _ = create_pass_backends_from_config(config_path)
            extraction_backend = pass1_backend  # Use pass1 for extraction
        except Exception as e:
            print(f"Error loading backends: {e}")
            sys.exit(1)

        # Extract facts from PDF
        print("Extracting facts from PDF...")
        chunks = load_pdf_chunks(str(pdf_path))
        print(f"  Loaded {len(chunks)} chunks")

        # Pass 1: Extract raw statements
        print("  Pass 1: Extract raw statements...")
        all_statements = extract_raw_statements_batched(chunks, backend=extraction_backend)
        print(f"  Extracted {len(all_statements)} statements")

        # Pass 2: Assign concepts
        print("  Pass 2: Assign concepts...")
        all_facts = assign_concepts_to_statements(all_statements, backend=pass2_backend)
        print(f"  Assigned concepts to {len(all_facts)} facts")

        # Group facts
        grouped = group_facts_by_concept(all_facts)
        final_grouped = normalize_group_keys(grouped)
        print(f"  Grouped into {len(final_grouped)} concepts")
        print()

    # Load vault pages
    print("Loading vault pages...")
    pages_by_title = load_vault_pages(vault_dir)
    print(f"  Loaded pages for {len(pages_by_title)} concepts")

    # Group pages by backend
    pages_by_backend = defaultdict(list)
    for title, page_text in pages_by_title.items():
        # Extract backend from frontmatter
        backend_label = None
        for backend, content in page_text.items():
            backend_label = backend
            break

        if backend_label:
            pages_by_backend[backend_label].append((title, page_text.get(backend_label, "")))

    if len(pages_by_backend) < 2:
        print("Warning: Found pages from only 1 backend")
        print(f"Backends found: {list(pages_by_backend.keys())}")
        sys.exit(0)

    print(f"  Found {len(pages_by_backend)} backends: {', '.join(sorted(pages_by_backend.keys()))}")
    for backend, pages in pages_by_backend.items():
        print(f"    {backend}: {len(pages)} pages")
    print()

    # Analyze pages
    print("Analyzing pages by backend...")
    backend_stats = defaultdict(lambda: {
        "coverage": [],
        "hallucination_risks": {},
        "page_lengths": [],
        "section_counts": [],
        "wikilink_counts": [],
        "code_block_counts": [],
        "list_counts": [],
        "word_counts": [],
        "list_line_ratios": [],
        "with_source_facts": 0,
        "with_heuristics_only": 0,
    })

    matched_pages = 0
    unmatched_pages = 0
    heuristic_only = 0

    for backend_label, page_text_list in pages_by_backend.items():
        for page_title, page_text in page_text_list:
            # Try to find source facts (skip if heuristics-only mode)
            source_facts = None
            if final_grouped:
                for concept_key, facts in final_grouped.items():
                    if fuzzy_match_concepts(concept_key, page_title, threshold=0.6):
                        source_facts = facts
                        break

            if source_facts:
                matched_pages += 1
                backend_stats[backend_label]["with_source_facts"] += 1
                fact_contents = [f.content for f in source_facts]
                coverage = measure_coverage(fact_contents, page_text)
                hallucination_risk = estimate_hallucination_risk(page_text, fact_contents, coverage)

                backend_stats[backend_label]["coverage"].append(coverage)
                if hallucination_risk not in backend_stats[backend_label]["hallucination_risks"]:
                    backend_stats[backend_label]["hallucination_risks"][hallucination_risk] = 0
                backend_stats[backend_label]["hallucination_risks"][hallucination_risk] += 1
            else:
                if final_grouped:
                    unmatched_pages += 1
                heuristic_only += 1
                backend_stats[backend_label]["with_heuristics_only"] += 1

            # Always collect structural metrics
            page_length = len(page_text)
            backend_stats[backend_label]["page_lengths"].append(page_length)

            structure = measure_structure_quality(page_text)
            backend_stats[backend_label]["section_counts"].append(structure["section_count"])
            backend_stats[backend_label]["wikilink_counts"].append(structure["wikilink_count"])
            backend_stats[backend_label]["code_block_counts"].append(structure["code_block_count"])
            backend_stats[backend_label]["list_counts"].append(structure["list_count"])
            backend_stats[backend_label]["word_counts"].append(structure["word_count"])
            backend_stats[backend_label]["list_line_ratios"].append(structure["list_line_ratio"])

    # Print results
    print(f"  Matched {matched_pages} pages to source facts, {heuristic_only} pages analyzed via structure")
    print()

    if matched_pages == 0 and final_grouped:
        print("Warning: No pages could be matched to source facts")
        print("(Will use structural metrics only)")
        print()

    print("\n=== BACKEND COMPARISON ===\n")

    backend_names = sorted(backend_stats.keys())

    for backend_label in backend_names:
        stats = backend_stats[backend_label]
        total_pages = stats["with_source_facts"] + stats["with_heuristics_only"]

        if total_pages == 0:
            continue

        print(f"{backend_label}  ({total_pages} pages total)")
        print(f"  Analyzed: {stats['with_source_facts']} with source facts, {stats['with_heuristics_only']} via heuristics")

        # Coverage-based metrics (if available)
        if stats["coverage"]:
            avg_coverage = sum(stats["coverage"]) / len(stats["coverage"]) * 100
            print(f"  Average Coverage:    {avg_coverage:.1f}%")
            print(f"  Hallucination Risk Distribution:")
            for risk_level in ["LOW", "MEDIUM", "HIGH"]:
                count = stats["hallucination_risks"].get(risk_level, 0)
                pct = count / len(stats["coverage"]) * 100
                print(f"    {risk_level:6}: {count:3} pages ({pct:5.1f}%)")

        # Structural metrics (always available)
        if stats["page_lengths"]:
            avg_length = sum(stats["page_lengths"]) / len(stats["page_lengths"])
            avg_sections = sum(stats["section_counts"]) / len(stats["section_counts"])
            avg_wikilinks = sum(stats["wikilink_counts"]) / len(stats["wikilink_counts"])
            avg_code_blocks = sum(stats["code_block_counts"]) / len(stats["code_block_counts"])
            avg_lists = sum(stats["list_counts"]) / len(stats["list_counts"])
            avg_words = sum(stats["word_counts"]) / len(stats["word_counts"])
            avg_list_ratio = sum(stats["list_line_ratios"]) / len(stats["list_line_ratios"])

            print(f"  Structure Metrics:")
            print(f"    Average Page Length: {avg_length:.0f} chars ({avg_words:.0f} words)")
            print(f"    Avg Sections:        {avg_sections:.1f}")
            print(f"    Avg Wikilinks:       {avg_wikilinks:.1f}")
            print(f"    Avg Code Blocks:     {avg_code_blocks:.1f}")
            print(f"    Avg Lists:           {avg_lists:.1f}")
            print(f"    List Line Ratio:     {avg_list_ratio:.1%}")

        print()

    # Comparison summary
    if len(backend_names) >= 2:
        print("=== COMPARISON SUMMARY ===\n")

        # Get backends with data
        backends_with_data = [b for b in backend_names if backend_stats[b]["page_lengths"]]

        if len(backends_with_data) >= 2:
            stats_list = [(b, backend_stats[b]) for b in backends_with_data]
            b1_name, b1_stats = stats_list[0]
            b2_name, b2_stats = stats_list[1]

            # Coverage comparison (if available)
            if b1_stats["coverage"] and b2_stats["coverage"]:
                b1_cov = sum(b1_stats["coverage"]) / len(b1_stats["coverage"]) * 100
                b2_cov = sum(b2_stats["coverage"]) / len(b2_stats["coverage"]) * 100

                b1_low_risk = b1_stats["hallucination_risks"].get("LOW", 0) / len(b1_stats["coverage"]) * 100
                b2_low_risk = b2_stats["hallucination_risks"].get("LOW", 0) / len(b2_stats["coverage"]) * 100

                print(f"Coverage: {b1_name} {b1_cov:.1f}% vs {b2_name} {b2_cov:.1f}%", end="")
                if abs(b1_cov - b2_cov) > 2:
                    winner = b1_name if b1_cov > b2_cov else b2_name
                    print(f" → {winner} better (+{abs(b1_cov - b2_cov):.1f}%)")
                else:
                    print(" → comparable")

                print(f"Low Risk:  {b1_name} {b1_low_risk:.1f}% vs {b2_name} {b2_low_risk:.1f}%", end="")
                if abs(b1_low_risk - b2_low_risk) > 10:
                    winner = b1_name if b1_low_risk > b2_low_risk else b2_name
                    print(f" → {winner} better")
                else:
                    print(" → comparable")
                print()

            # Structural comparison (always available)
            b1_len = sum(b1_stats["page_lengths"]) / len(b1_stats["page_lengths"])
            b2_len = sum(b2_stats["page_lengths"]) / len(b2_stats["page_lengths"])

            b1_words = sum(b1_stats["word_counts"]) / len(b1_stats["word_counts"])
            b2_words = sum(b2_stats["word_counts"]) / len(b2_stats["word_counts"])

            b1_sections = sum(b1_stats["section_counts"]) / len(b1_stats["section_counts"])
            b2_sections = sum(b2_stats["section_counts"]) / len(b2_stats["section_counts"])

            b1_links = sum(b1_stats["wikilink_counts"]) / len(b1_stats["wikilink_counts"])
            b2_links = sum(b2_stats["wikilink_counts"]) / len(b2_stats["wikilink_counts"])

            b1_code = sum(b1_stats["code_block_counts"]) / len(b1_stats["code_block_counts"])
            b2_code = sum(b2_stats["code_block_counts"]) / len(b2_stats["code_block_counts"])

            print("Structural Metrics:")
            print(f"  Page Length: {b1_name} {b1_len:.0f}c ({b1_words:.0f}w) vs {b2_name} {b2_len:.0f}c ({b2_words:.0f}w)", end="")
            if b1_len > b2_len * 1.2:
                print(f" → {b2_name} is more concise")
            elif b2_len > b1_len * 1.2:
                print(f" → {b1_name} is more concise")
            else:
                print(" → comparable")

            print(f"  Sections:    {b1_name} {b1_sections:.1f} vs {b2_name} {b2_sections:.1f}", end="")
            if b1_sections > b2_sections + 1:
                print(f" → {b1_name} more detailed")
            elif b2_sections > b1_sections + 1:
                print(f" → {b2_name} more detailed")
            else:
                print(" → comparable")

            print(f"  Wikilinks:   {b1_name} {b1_links:.1f} vs {b2_name} {b2_links:.1f}", end="")
            if b1_links > b2_links * 1.1:
                print(f" → {b1_name} more connected")
            elif b2_links > b1_links * 1.1:
                print(f" → {b2_name} more connected")
            else:
                print(" → comparable")

            print(f"  Code Blocks: {b1_name} {b1_code:.1f} vs {b2_name} {b2_code:.1f}", end="")
            if b2_code > b1_code:
                print(f" → {b2_name} more technical/formal")
            elif b1_code > b2_code:
                print(f" → {b1_name} more technical/formal")
            else:
                print(" → comparable")

        print()


if __name__ == "__main__":
    main()
