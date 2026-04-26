import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from ai_research_agent.budget import Budget
from ai_research_agent.models import Paper, RepoBundle
from ai_research_agent.synthesizer import synthesize, _parse_synthesis_json


FIXTURE = Path(__file__).parent / "fixtures" / "sonnet_synthesis_response.json"


def make_paper() -> Paper:
    return Paper(
        arxiv_id="2404.12345", version="v1", title="Test Paper",
        authors=["Alice"], abstract="abs",
        published=datetime(2026, 4, 20, tzinfo=timezone.utc),
        updated=datetime(2026, 4, 20, tzinfo=timezone.utc),
        categories=["cs.AI"], pdf_url="http://arxiv.org/pdf/x",
        arxiv_url="http://arxiv.org/abs/x",
    )


def make_repo() -> RepoBundle:
    return RepoBundle(
        repo_url="https://github.com/foo/bar",
        repo_kind="github",
        readme="# foo\nrun python train.py",
        file_tree=["README.md", "train.py"],
        truncated=False,
    )


def test_parse_synthesis_json_strict():
    parsed = _parse_synthesis_json(FIXTURE.read_text())
    assert "why_this_matters" in parsed
    assert "implementation_plan" in parsed


def test_parse_synthesis_json_handles_markdown_wrapper():
    raw = "Here:\n```json\n" + FIXTURE.read_text() + "\n```"
    parsed = _parse_synthesis_json(raw)
    assert parsed["technical_idea"]


def test_synthesize_renders_markdown_with_filled_template():
    fake_msg = MagicMock()
    fake_msg.content = [MagicMock(text=FIXTURE.read_text())]
    fake_msg.usage = MagicMock(input_tokens=2000, output_tokens=500)

    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_msg

    budget = Budget(cap_usd=10.0)
    with patch("ai_research_agent.synthesizer._client", return_value=fake_client):
        md = synthesize(
            paper=make_paper(),
            repo=make_repo(),
            full_pdf_text="full paper body",
            budget=budget,
        )

    assert "# Test Paper" in md
    assert "## Implementation plan" in md
    assert "ReAct-style tool use" in md
    assert "github.com/foo/bar" in md
    assert budget.spent > 0


def test_synthesize_handles_missing_repo():
    fake_msg = MagicMock()
    fake_msg.content = [MagicMock(text=FIXTURE.read_text())]
    fake_msg.usage = MagicMock(input_tokens=1000, output_tokens=500)

    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_msg

    with patch("ai_research_agent.synthesizer._client", return_value=fake_client):
        md = synthesize(
            paper=make_paper(),
            repo=None,
            full_pdf_text="full body",
            budget=Budget(cap_usd=10.0),
        )
    assert "(none found)" in md
