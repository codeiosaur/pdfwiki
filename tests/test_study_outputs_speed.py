from pathlib import Path

from pdfwiki.study_outputs import StudyOutputDeps, generate_study_aids


def _make_deps(tmp_path, query_calls):
    def load_prompt(name: str) -> str:
        return "PROMPT\n{text}"

    def query(prompt: str, task: str = "cheap", max_tokens: int = 4096) -> str:
        query_calls.append((task, max_tokens, prompt))
        return "generated"

    def parse_wiki_page(raw: str):
        return "x", raw

    def add_frontmatter(*args, **kwargs):
        return "fm"

    def write_flashcards(output_dir: str, subject: str, content: str) -> str:
        path = Path(output_dir) / "flashcards"
        path.mkdir(parents=True, exist_ok=True)
        out = path / f"{subject}_flashcards.md"
        out.write_text(content, encoding="utf-8")
        return str(out)

    def write_cheatsheet(output_dir: str, subject: str, content: str) -> str:
        path = Path(output_dir) / "cheatsheets"
        path.mkdir(parents=True, exist_ok=True)
        out = path / f"{subject}_cheatsheet.md"
        out.write_text(content, encoding="utf-8")
        return str(out)

    return StudyOutputDeps(
        load_vault_state=lambda _: {"pages": {}},
        load_prompt=load_prompt,
        query=query,
        parse_wiki_page=parse_wiki_page,
        add_frontmatter=add_frontmatter,
        write_flashcards=write_flashcards,
        write_cheatsheet=write_cheatsheet,
    )


def test_generate_study_aids_skips_when_source_hash_matches(tmp_path):
    query_calls = []
    deps = _make_deps(tmp_path, query_calls)

    summary = "A" * 3000
    generate_study_aids(
        output_dir=str(tmp_path),
        subject="Cryptography",
        summary_text=summary,
        flashcards_max_tokens=500,
        cheatsheet_max_tokens=600,
        summary_max_chars=2000,
        deps=deps,
    )
    assert len(query_calls) == 2

    query_calls.clear()
    generate_study_aids(
        output_dir=str(tmp_path),
        subject="Cryptography",
        summary_text=summary,
        flashcards_max_tokens=500,
        cheatsheet_max_tokens=600,
        summary_max_chars=2000,
        deps=deps,
    )
    assert len(query_calls) == 0


def test_generate_study_aids_regenerates_when_summary_changes(tmp_path):
    query_calls = []
    deps = _make_deps(tmp_path, query_calls)

    generate_study_aids(
        output_dir=str(tmp_path),
        subject="Cryptography",
        summary_text="summary one",
        flashcards_max_tokens=501,
        cheatsheet_max_tokens=602,
        summary_max_chars=5000,
        deps=deps,
    )
    assert len(query_calls) == 2
    assert query_calls[0][1] == 501
    assert query_calls[1][1] == 602

    query_calls.clear()
    generate_study_aids(
        output_dir=str(tmp_path),
        subject="Cryptography",
        summary_text="summary two",
        flashcards_max_tokens=501,
        cheatsheet_max_tokens=602,
        summary_max_chars=5000,
        deps=deps,
    )
    assert len(query_calls) == 2
