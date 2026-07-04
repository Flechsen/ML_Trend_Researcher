from datetime import datetime, timezone

import httpx
import pytest
import respx

from ai_research_agent.models import TrendItem
from ai_research_agent.trends._http import get_json, get_text
from ai_research_agent.trends.config import DEFAULT_CONFIG, load_config


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
