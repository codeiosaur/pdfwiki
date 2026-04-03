from __future__ import annotations

from typing import Iterable


def is_question_prompt(text: str) -> bool:
    """Return True when a fact looks like a prompt or review question."""
    stripped = text.strip()
    return stripped.endswith("?")


def bullet_section(title: str, items: Iterable[str]) -> list[str]:
    rendered_items = list(items)
    if not rendered_items:
        return []

    lines = ["", "---", "", f"## {title}", ""]
    for item in rendered_items:
        lines.append(f"- {item}")
    lines.append("")
    return lines


def code_section(title: str, items: Iterable[str]) -> list[str]:
    rendered_items = list(items)
    if not rendered_items:
        return []

    lines = ["", "---", "", f"## {title}", ""]
    for item in rendered_items:
        lines.extend(["```", item, "```", ""])
    return lines


def build_wiki_page_lines(
    display_title: str,
    intro: str,
    definition: str,
    concept_type: str,
    formulas: list[str],
    interpretations: list[str],
    examples: list[str],
    cautions: list[str],
    details: list[str],
    related: list[str],
    see_also: list[str],
    sources: list[str],
    include_examples: bool = False,
) -> list[str]:
    """Build a Wikipedia-like page that still reads like a study note."""
    lines: list[str] = [f"# {display_title}", "", intro, "", "---", "", "## Definition", definition]

    if concept_type == "ratio":
        lines.extend(code_section("Formula", formulas))
        lines.extend(bullet_section("What It Tells You", interpretations or details))
        if details and interpretations:
            combined = [item for item in details if item not in interpretations]
            lines.extend(bullet_section("Key Takeaways", combined))
    elif concept_type in ("method", "system"):
        lines.extend(bullet_section("How It Works", details or interpretations))
        lines.extend(code_section("Formula", formulas))
        if interpretations:
            lines.extend(bullet_section("Key Takeaways", interpretations))
    else:
        lines.extend(bullet_section("Key Takeaways", details + interpretations))

    if include_examples:
        lines.extend(bullet_section("Example", examples))
    lines.extend(bullet_section("Cautions", cautions))

    lines.extend(["", "---", "", "## Related Concepts"])
    if related:
        for item in related:
            lines.append(f"- [[{item}]]")
    else:
        lines.append("- None")

    if see_also:
        lines.extend(["", "---", "", "## See Also"])
        for item in see_also:
            lines.append(f"- [[{item}]]")

    if sources:
        lines.extend(["", "---", "", "## Sources"])
        for src in sources:
            lines.append(f"- {src}")

    return lines