import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
import respx

from ai_research_agent.models import TrendItem
from ai_research_agent.trends import github_trending, hackernews
from ai_research_agent.trends._http import get_json, get_text
from ai_research_agent.trends.config import DEFAULT_CONFIG, load_config

FIXTURES = Path(__file__).parent / "fixtures"


def make_item(**kw) -> TrendItem:
    defaults = dict(
        source="hackernews",
        title="Some story",
        url="https://example.com/x",
        score=100,
        detail="42 comments",
        created_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
    )
    defaults.update(kw)
    return TrendItem(**defaults)


def test_trend_item_fields():
    item = make_item()
    assert item.source == "hackernews"
    assert item.score == 100


def test_trend_item_allows_missing_score_and_date():
    item = make_item(score=None, created_at=None)
    assert item.score is None
    assert item.created_at is None


def test_load_config_defaults_when_no_trends_block():
    cfg = load_config({"interests": []})
    assert cfg == DEFAULT_CONFIG
    assert cfg["max_items_per_source"] == 30


def test_load_config_merges_user_overrides():
    cfg = load_config({"trends": {"min_hn_points": 10, "subreddits": ["LocalLLaMA"]}})
    assert cfg["min_hn_points"] == 10
    assert cfg["subreddits"] == ["LocalLLaMA"]
    assert cfg["min_github_stars"] == DEFAULT_CONFIG["min_github_stars"]  # untouched default


def test_load_config_handles_null_trends_block():
    assert load_config({"trends": None}) == DEFAULT_CONFIG


@respx.mock
def test_get_json_retries_on_5xx_then_succeeds():
    route = respx.get("https://api.example.com/data").mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json={"ok": True})]
    )
    assert get_json("https://api.example.com/data", params={}) == {"ok": True}
    assert route.call_count == 2


@respx.mock
def test_get_json_raises_immediately_on_404():
    route = respx.get("https://api.example.com/gone").mock(return_value=httpx.Response(404))
    with pytest.raises(httpx.HTTPStatusError):
        get_json("https://api.example.com/gone", params={})
    assert route.call_count == 1


@respx.mock
def test_get_text_returns_body():
    respx.get("https://feeds.example.com/x").mock(
        return_value=httpx.Response(200, text="<feed/>")
    )
    assert get_text("https://feeds.example.com/x", params={"t": "week"}) == "<feed/>"


def _hn_cfg(**over) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update({"hn_queries": ["MCP", "LLM"], "min_hn_points": 40})
    cfg.update(over)
    return cfg


@respx.mock
def test_hn_fetch_filters_points_and_dedups_across_queries():
    route = respx.get("https://hn.algolia.com/api/v1/search").mock(
        return_value=httpx.Response(
            200, json=json.loads((FIXTURES / "hn_search_response.json").read_text())
        )
    )
    items = hackernews.fetch(_hn_cfg())
    assert route.call_count == 2  # one call per query
    assert len(items) == 2  # 12-point story dropped; duplicates collapsed
    assert items[0].score == 262  # sorted by points desc
    assert all(i.source == "hackernews" for i in items)


@respx.mock
def test_hn_fetch_falls_back_to_hn_permalink_when_story_has_no_url():
    respx.get("https://hn.algolia.com/api/v1/search").mock(
        return_value=httpx.Response(
            200, json=json.loads((FIXTURES / "hn_search_response.json").read_text())
        )
    )
    items = hackernews.fetch(_hn_cfg(hn_queries=["MCP"]))
    launch = next(i for i in items if "Manufact" in i.title)
    assert launch.url == "https://news.ycombinator.com/item?id=48762862"


@respx.mock
def test_hn_fetch_respects_max_items_cap():
    respx.get("https://hn.algolia.com/api/v1/search").mock(
        return_value=httpx.Response(
            200, json=json.loads((FIXTURES / "hn_search_response.json").read_text())
        )
    )
    items = hackernews.fetch(_hn_cfg(max_items_per_source=1))
    assert len(items) == 1


def _gh_cfg(**over) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update({"github_topics": ["mcp"], "github_keywords": ["llm agent"]})
    cfg.update(over)
    return cfg


@respx.mock
def test_github_fetch_queries_topics_and_keywords_and_dedups(monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    route = respx.get("https://api.github.com/search/repositories").mock(
        return_value=httpx.Response(
            200, json=json.loads((FIXTURES / "github_search_response.json").read_text())
        )
    )
    items = github_trending.fetch(_gh_cfg())
    assert route.call_count == 2  # 1 topic + 1 keyword query
    assert len(items) == 2  # same repos in both responses -> deduped
    assert items[0].title == "acme/mcp-router"  # sorted by stars desc
    assert items[0].score == 1200
    assert items[0].source == "github"
    assert "mcp" in items[0].detail  # topics folded into detail
    assert items[1].detail == ""  # null description handled


@respx.mock
def test_github_fetch_sends_auth_header_when_token_present(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "gh-test-token")
    route = respx.get("https://api.github.com/search/repositories").mock(
        return_value=httpx.Response(
            200, json=json.loads((FIXTURES / "github_search_response.json").read_text())
        )
    )
    github_trending.fetch(_gh_cfg(github_keywords=[]))
    assert route.calls[0].request.headers["authorization"] == "Bearer gh-test-token"
