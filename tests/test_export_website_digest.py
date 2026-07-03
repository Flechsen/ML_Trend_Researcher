from scripts.export_website_digest import export, first_sentence, parse_brief

BRIEF = """# Semantic Early-Stopping for Iterative LLM Agent Loops

## Metadata
- arXiv ID: 2606.27009
- Authors: Sahil Shrivastava
- Published: 2026-06-25
- arXiv link: https://arxiv.org/abs/2606.27009v1
- PDF link: https://arxiv.org/pdf/2606.27009v1

## Why this matters
Most LLM agent loops (Writer→Critic, self-refinement, etc.) terminate via a fixed integer cap — max_iterations. This is wasteful in both directions.

## Technical idea
Details here.
"""


def write_brief(tmp_path, name="2606.27009-semantic-early-stopping.md", text=BRIEF):
    d = tmp_path / "papers" / "2026"
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(text, encoding="utf-8")
    return p


def test_parse_brief_extracts_fields(tmp_path):
    entry = parse_brief(write_brief(tmp_path), tmp_path / "papers")
    assert entry == {
        "title": "Semantic Early-Stopping for Iterative LLM Agent Loops",
        "date": "2026-06-25",
        "gist": "Most LLM agent loops (Writer→Critic, self-refinement, etc.) terminate via a fixed integer cap — max_iterations.",
        "arxiv": "https://arxiv.org/abs/2606.27009v1",
        "brief": "https://github.com/Flechsen/ML_Trend_Researcher/blob/master/papers/2026/2606.27009-semantic-early-stopping.md",
    }


def test_first_sentence_survives_abbreviations():
    assert first_sentence("Uses e.g. RAG (etc.) heavily. Second sentence.") == \
        "Uses e.g. RAG (etc.) heavily."


def test_parse_brief_missing_published_returns_none(tmp_path, capsys):
    broken = BRIEF.replace("- Published: 2026-06-25\n", "")
    assert parse_brief(write_brief(tmp_path, text=broken), tmp_path / "papers") is None
    assert "skip" in capsys.readouterr().err.lower()


def test_export_sorted_and_deterministic(tmp_path):
    write_brief(tmp_path)
    older = BRIEF.replace("2026-06-25", "2026-04-23").replace("2606.27009", "2604.21725")
    write_brief(tmp_path, name="2604.21725-ael.md", text=older)
    out1, out2 = tmp_path / "a.yaml", tmp_path / "b.yaml"
    export(tmp_path / "papers", out1)
    export(tmp_path / "papers", out2)
    assert out1.read_bytes() == out2.read_bytes()
    text = out1.read_text(encoding="utf-8")
    assert text.startswith("# GENERATED FILE")
    assert text.index("2026-06-25") < text.index("2026-04-23")
    assert "date: '2026-06-25'" in text  # quoted string, not a YAML timestamp
