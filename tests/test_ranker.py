from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from ai_research_agent.budget import Budget
from ai_research_agent.models import Paper, ScoredPaper
from ai_research_agent.ranker import rank_candidates, _parse_ranking_json


FIXTURE = Path(__file__).parent / "fixtures" / "haiku_rank_response.json"


def make_scored(arxiv_id: str, abstract: str = "abs", score: float = 0.5) -> ScoredPaper:
    return ScoredPaper(
        paper=Paper(
            arxiv_id=arxiv_id, version="v1", title=arxiv_id, authors=[], abstract=abstract,
            published=datetime(2026, 4, 20, tzinfo=timezone.utc),
            updated=datetime(2026, 4, 20, tzinfo=timezone.utc),
            categories=[], pdf_url="", arxiv_url="",
        ),
        embedding_score=score,
    )


def test_parse_ranking_json_strict():
    raw = FIXTURE.read_text()
    parsed = _parse_ranking_json(raw)
    assert len(parsed) == 3
    assert parsed[0]["arxiv_id"] == "2404.12345"
    assert parsed[0]["score"] == 9


def test_parse_ranking_json_handles_markdown_wrapper():
    raw = "Here you go:\n```json\n" + FIXTURE.read_text() + "\n```\nThanks!"
    parsed = _parse_ranking_json(raw)
    assert len(parsed) == 3


def test_parse_ranking_json_drops_malformed_entries():
    raw = '[{"arxiv_id": "x", "score": 8, "reasoning": "ok"}, {"foo": "bar"}]'
    parsed = _parse_ranking_json(raw)
    assert len(parsed) == 1
    assert parsed[0]["arxiv_id"] == "x"


def test_rank_candidates_returns_top_n():
    scored = [
        make_scored("2404.12345", abstract="agent code at github.com/a/b"),
        make_scored("2404.12399", abstract="theory paper"),
        make_scored("2404.12500", abstract="RAG hf.co/foo/bar"),
    ]
    interests = {"interests": [], "mvp_constraints": {"hard_drops": [], "preferred_signals": []}}

    fake_msg = MagicMock()
    fake_msg.content = [MagicMock(text=FIXTURE.read_text())]
    fake_msg.usage = MagicMock(input_tokens=500, output_tokens=200)

    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_msg

    budget = Budget(cap_usd=10.0)
    with patch("ai_research_agent.ranker._client", return_value=fake_client):
        result = rank_candidates(scored, interests, top_n=2, budget=budget)

    assert len(result) == 2
    # Should come back sorted by llm_score desc
    assert result[0].llm_score == 9
    assert result[1].llm_score == 7
    assert result[0].has_repo_url_in_abstract  # github.com in abstract
    assert budget.spent > 0


def test_rank_candidates_empty_input_returns_empty():
    interests = {"interests": [], "mvp_constraints": {"hard_drops": [], "preferred_signals": []}}
    out = rank_candidates([], interests, top_n=5, budget=Budget(cap_usd=1.0))
    assert out == []
