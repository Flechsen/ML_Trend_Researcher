from datetime import datetime, timezone

from ai_research_agent.models import TrendItem
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
