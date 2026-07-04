"""Hugging Face trend fetcher: community-upvoted daily papers + trending models."""
import logging
from datetime import datetime, timedelta, timezone

from ai_research_agent.models import TrendItem
from ai_research_agent.trends._http import get_json

logger = logging.getLogger(__name__)

HF_PAPERS_URL = "https://huggingface.co/api/daily_papers"
HF_MODELS_URL = "https://huggingface.co/api/models"


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _fetch_papers(cfg: dict, days: int) -> list[TrendItem]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    items: list[TrendItem] = []
    for entry in get_json(HF_PAPERS_URL, params={"limit": 50}):
        paper = entry.get("paper") or {}
        pid = paper.get("id")
        published = _parse_dt(entry.get("publishedAt") or paper.get("publishedAt"))
        if not pid or (published and published < cutoff):
            continue
        items.append(TrendItem(
            source="hf_papers",
            title=(entry.get("title") or paper.get("title") or "").strip(),
            url=f"https://huggingface.co/papers/{pid}",
            score=paper.get("upvotes"),
            detail=(paper.get("summary") or "").strip()[:200],
            created_at=published,
        ))
    items.sort(key=lambda t: t.score or 0, reverse=True)
    return items[: cfg["max_items_per_source"]]


def _fetch_models(cfg: dict) -> list[TrendItem]:
    items: list[TrendItem] = []
    data = get_json(
        HF_MODELS_URL,
        params={"sort": "trendingScore", "limit": cfg["max_items_per_source"]},
    )
    for model in data:
        mid = model.get("id") or model.get("modelId")
        if not mid:
            continue
        task = model.get("pipeline_tag") or "unknown task"
        items.append(TrendItem(
            source="hf_models",
            title=mid,
            url=f"https://huggingface.co/{mid}",
            score=model.get("trendingScore"),
            detail=f"{model.get('likes') or 0} likes · {task}",
            created_at=_parse_dt(model.get("createdAt")),
        ))
    return items


def fetch(cfg: dict, days: int = 7) -> list[TrendItem]:
    papers = _fetch_papers(cfg, days)
    models = _fetch_models(cfg)
    logger.info("huggingface: %d papers, %d trending models", len(papers), len(models))
    return papers + models
