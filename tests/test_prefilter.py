from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from ai_research_agent.models import Paper
from ai_research_agent.prefilter import score_by_embedding, _cosine, _interests_to_embedding_text


def make_paper(arxiv_id: str, abstract: str) -> Paper:
    return Paper(
        arxiv_id=arxiv_id, version="v1", title="t", authors=[], abstract=abstract,
        published=datetime(2026, 4, 20, tzinfo=timezone.utc),
        updated=datetime(2026, 4, 20, tzinfo=timezone.utc),
        categories=[], pdf_url="", arxiv_url="",
    )


def test_cosine_identical_vectors_is_one():
    assert abs(_cosine([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) - 1.0) < 1e-9


def test_cosine_orthogonal_is_zero():
    assert abs(_cosine([1.0, 0.0], [0.0, 1.0])) < 1e-9


def test_interests_to_embedding_text_includes_topics_and_examples():
    interests = {
        "interests": [
            {"topic": "agents", "description": "tool use", "examples": ["ReAct"], "anti_examples": []},
        ],
    }
    text = _interests_to_embedding_text(interests)
    assert "agents" in text
    assert "tool use" in text
    assert "ReAct" in text


def test_score_by_embedding_returns_top_n_sorted():
    interests = {"interests": [{"topic": "agents", "description": "tool use", "examples": [], "anti_examples": []}]}
    papers = [
        make_paper("1", "tool-using agent"),
        make_paper("2", "unrelated topic about gardens"),
        make_paper("3", "another agent paper"),
    ]
    interest_resp = MagicMock(
        data=[MagicMock(embedding=[1.0, 0.0])],
        usage=MagicMock(prompt_tokens=10, total_tokens=10),
    )
    paper_resp = MagicMock(
        data=[
            MagicMock(embedding=[1.0, 0.0]),  # paper 1, perfect match
            MagicMock(embedding=[0.0, 1.0]),  # paper 2, orthogonal
            MagicMock(embedding=[0.9, 0.1]),  # paper 3, close
        ],
        usage=MagicMock(prompt_tokens=100, total_tokens=100),
    )

    fake_client = MagicMock()
    fake_client.embeddings.create.side_effect = [interest_resp, paper_resp]

    with patch("ai_research_agent.prefilter._client", return_value=fake_client):
        result = score_by_embedding(papers, interests, top_n=2, budget=None)

    assert len(result) == 2
    assert result[0].paper.arxiv_id == "1"  # best match
    assert result[1].paper.arxiv_id == "3"  # second


def test_score_by_embedding_batches_large_input():
    """Papers > EMBED_BATCH_SIZE must be split into multiple API calls."""
    from ai_research_agent.prefilter import EMBED_BATCH_SIZE

    interests = {"interests": [{"topic": "x", "description": "y", "examples": [], "anti_examples": []}]}
    papers = [make_paper(str(i), f"abstract {i}") for i in range(EMBED_BATCH_SIZE + 50)]

    interest_resp = MagicMock(
        data=[MagicMock(embedding=[1.0, 0.0])],
        usage=MagicMock(prompt_tokens=10),
    )
    batch1 = MagicMock(
        data=[MagicMock(embedding=[1.0, 0.0]) for _ in range(EMBED_BATCH_SIZE)],
        usage=MagicMock(prompt_tokens=50_000),
    )
    batch2 = MagicMock(
        data=[MagicMock(embedding=[1.0, 0.0]) for _ in range(50)],
        usage=MagicMock(prompt_tokens=5_000),
    )

    fake_client = MagicMock()
    fake_client.embeddings.create.side_effect = [interest_resp, batch1, batch2]

    with patch("ai_research_agent.prefilter._client", return_value=fake_client):
        result = score_by_embedding(papers, interests, top_n=10, budget=None)

    assert fake_client.embeddings.create.call_count == 3  # interests + 2 paper batches
    assert len(result) == 10
