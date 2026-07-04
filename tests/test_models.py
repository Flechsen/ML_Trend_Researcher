from datetime import datetime, timezone
from ai_research_agent.models import Paper, RankedCandidate, RepoBundle


def test_paper_is_frozen():
    p = Paper(
        arxiv_id="2404.12345",
        version="v1",
        title="Test",
        authors=["Alice"],
        abstract="abs",
        published=datetime(2026, 4, 20, tzinfo=timezone.utc),
        updated=datetime(2026, 4, 20, tzinfo=timezone.utc),
        categories=["cs.AI"],
        pdf_url="https://arxiv.org/pdf/2404.12345",
        arxiv_url="https://arxiv.org/abs/2404.12345",
    )
    import dataclasses
    assert dataclasses.is_dataclass(p)
    try:
        p.title = "Changed"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("Paper should be frozen")


def test_repo_bundle_kinds():
    rb = RepoBundle(
        repo_url="https://github.com/foo/bar",
        repo_kind="github",
        readme="# foo",
        file_tree=["README.md", "src/main.py"],
        truncated=False,
    )
    assert rb.repo_kind in ("github", "huggingface")


def test_ranked_candidate_score_range():
    rc = RankedCandidate(
        paper=Paper(
            arxiv_id="2404.12345", version="v1", title="t", authors=[], abstract="",
            published=datetime(2026, 4, 20, tzinfo=timezone.utc),
            updated=datetime(2026, 4, 20, tzinfo=timezone.utc),
            categories=[], pdf_url="", arxiv_url="",
        ),
        embedding_score=0.5,
        llm_score=8,
        llm_reasoning="solid fit",
        has_repo_url_in_abstract=True,
    )
    assert 1 <= rc.llm_score <= 10
