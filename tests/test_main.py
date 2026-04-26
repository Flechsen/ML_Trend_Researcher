from pathlib import Path
from unittest.mock import MagicMock, patch

from ai_research_agent.main import (
    _slugify, _existing_arxiv_ids, _write_paper_file, _regenerate_index, run,
)


def test_slugify_basic():
    assert _slugify("ReAct: Tool Use") == "react-tool-use"


def test_slugify_caps_at_60_chars():
    long = "x" * 200
    assert len(_slugify(long)) <= 60


def test_slugify_strips_punctuation():
    assert _slugify("Hello, World! v2") == "hello-world-v2"


def test_existing_arxiv_ids_scans_papers_dir(tmp_path):
    (tmp_path / "2026").mkdir()
    (tmp_path / "2026" / "2404.12345-foo.md").write_text("x")
    (tmp_path / "2026" / "2403.99999-bar.md").write_text("y")
    (tmp_path / "INDEX.md").write_text("z")  # should be ignored
    ids = _existing_arxiv_ids(tmp_path)
    assert ids == {"2404.12345", "2403.99999"}


def test_write_paper_file_creates_year_subdir(tmp_path):
    path = _write_paper_file(
        papers_dir=tmp_path,
        arxiv_id="2404.12345",
        title="ReAct: Tool Use",
        published_year=2026,
        markdown="# foo",
    )
    assert path.exists()
    assert path.parent.name == "2026"
    assert "2404.12345" in path.name
    assert path.read_text() == "# foo"


def test_regenerate_index_lists_papers_in_reverse_chrono(tmp_path):
    (tmp_path / "2026").mkdir()
    (tmp_path / "2025").mkdir()
    (tmp_path / "2026" / "2404.12345-foo.md").write_text(
        "# Foo Paper\n\n## Metadata\n- arXiv ID: 2404.12345\n"
    )
    (tmp_path / "2025" / "2312.99999-bar.md").write_text(
        "# Bar Paper\n\n## Metadata\n- arXiv ID: 2312.99999\n"
    )
    _regenerate_index(tmp_path)
    idx = (tmp_path / "INDEX.md").read_text()
    # 2026 (newer) appears before 2025 (older)
    assert idx.find("Foo Paper") < idx.find("Bar Paper")


def test_run_validate_config_returns_zero_on_clean_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GH_TOKEN", "token")
    monkeypatch.setenv("GH_REPO", "foo/bar")
    monkeypatch.setenv("BUDGET_USD_CAP", "3.00")
    interests = tmp_path / "interests.yaml"
    interests.write_text("interests: []\nmvp_constraints:\n  hard_drops: []\n  preferred_signals: []\n")
    papers = tmp_path / "papers"
    papers.mkdir()
    rc = run(["--validate-config", "--interests", str(interests), "--papers-dir", str(papers)])
    assert rc == 0
