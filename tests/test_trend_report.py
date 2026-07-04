from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from ai_research_agent.budget import Budget
from ai_research_agent.models import TrendItem
from ai_research_agent.trends import report
from ai_research_agent.trends.report import TrendsError, generate

INTERESTS = {"interests": [{"topic": "MCP", "description": "Model Context Protocol"}]}

FAKE_BODY = (
    "## TL;DR\n- MCP servers everywhere\n\n## Themes\n### MCP ecosystem\n"
    "[acme/mcp-router](https://github.com/acme/mcp-router) leads.\n\n"
    "## Notable new tools & repos\n- x\n\n## Notable models & papers\n- y\n\n## Radar\n- z\n"
)


def make_item(source="hackernews", title="Story", score=100) -> TrendItem:
    return TrendItem(
        source=source, title=title, url="https://example.com/x", score=score,
        detail="detail", created_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
    )


def _fake_anthropic(body: str = FAKE_BODY) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=body)]
    msg.usage = MagicMock(input_tokens=5000, output_tokens=1500)
    client = MagicMock()
    client.messages.create.return_value = msg
    return client


def _patch_fetchers(hn=None, gh=None, hf=None, rd=None):
    def wrap(value):
        if isinstance(value, Exception):
            return MagicMock(side_effect=value)
        return MagicMock(return_value=value or [])
    return (
        patch.object(report.hackernews, "fetch", wrap(hn)),
        patch.object(report.github_trending, "fetch", wrap(gh)),
        patch.object(report.huggingface, "fetch", wrap(hf)),
        patch.object(report.reddit, "fetch", wrap(rd)),
    )


def test_generate_writes_dated_report_and_index(tmp_path):
    budget = Budget(cap_usd=3.0)
    p1, p2, p3, p4 = _patch_fetchers(hn=[make_item()], gh=[make_item(source="github")])
    with p1, p2, p3, p4, patch.object(report, "_client", return_value=_fake_anthropic()):
        path = generate(INTERESTS, budget, tmp_path)
    today = datetime.now(timezone.utc).date().isoformat()
    assert path == tmp_path / f"{today}.md"
    text = path.read_text()
    assert text.startswith(f"# ML Trends — Week of {today}")
    assert "## TL;DR" in text
    assert "hackernews: 1" in text and "github: 1" in text
    assert budget.spent > 0
    index = (tmp_path / "INDEX.md").read_text()
    assert f"{today}.md" in index


def test_generate_survives_partial_source_failure(tmp_path):
    p1, p2, p3, p4 = _patch_fetchers(hn=RuntimeError("HN down"), gh=[make_item(source="github")])
    with p1, p2, p3, p4, patch.object(report, "_client", return_value=_fake_anthropic()):
        path = generate(INTERESTS, Budget(cap_usd=3.0), tmp_path)
    text = path.read_text()
    assert "hackernews: 0" in text
    assert "Failed sources: hackernews" in text


def test_generate_raises_when_all_sources_empty(tmp_path):
    p1, p2, p3, p4 = _patch_fetchers()
    with p1, p2, p3, p4:
        with pytest.raises(TrendsError):
            generate(INTERESTS, Budget(cap_usd=3.0), tmp_path)
    assert not list(tmp_path.glob("*.md"))  # nothing written


def test_prompt_contains_items_and_interests():
    items = [make_item(title="Unique MCP Story")]
    prompt = report._build_prompt(INTERESTS, items)
    assert "Unique MCP Story" in prompt
    assert "MCP: Model Context Protocol" in prompt
    assert "## TL;DR" in prompt  # instructs required sections
