"""Hacker News trend fetcher (Algolia search API, keyless)."""
import logging
from datetime import datetime, timedelta, timezone

from ai_research_agent.models import TrendItem
from ai_research_agent.trends._http import get_json

logger = logging.getLogger(__name__)

HN_API_URL = "https://hn.algolia.com/api/v1/search"
HN_ITEM_URL = "https://news.ycombinator.com/item?id={oid}"


def fetch(cfg: dict, days: int = 7) -> list[TrendItem]:
    cutoff = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    seen: dict[str, TrendItem] = {}
    for query in cfg["hn_queries"]:
        data = get_json(HN_API_URL, params={
            "query": query,
            "tags": "story",
            # NB: only created_at_i is server-filterable; points is not (API 400s).
            "numericFilters": f"created_at_i>{cutoff}",
            "hitsPerPage": 50,
        })
        for hit in data.get("hits", []):
            oid = hit.get("objectID")
            points = hit.get("points") or 0
            if not oid or oid in seen or points < cfg["min_hn_points"]:
                continue
            permalink = HN_ITEM_URL.format(oid=oid)
            created = None
            if hit.get("created_at_i"):
                created = datetime.fromtimestamp(hit["created_at_i"], tz=timezone.utc)
            seen[oid] = TrendItem(
                source="hackernews",
                title=(hit.get("title") or "").strip(),
                url=hit.get("url") or permalink,
                score=points,
                detail=f"{hit.get('num_comments') or 0} comments — {permalink}",
                created_at=created,
            )
    items = sorted(seen.values(), key=lambda t: t.score or 0, reverse=True)
    logger.info("hackernews: %d unique stories >= %d points", len(items), cfg["min_hn_points"])
    return items[: cfg["max_items_per_source"]]
