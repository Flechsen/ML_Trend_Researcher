"""GitHub trending fetcher: new high-star repos via the search API.

The /trending page has no API; searching repos *created* recently with a star
floor is the standard proxy and keeps old mega-repos out of the results.
"""
import logging
import os
from datetime import datetime, timedelta, timezone

from ai_research_agent.models import TrendItem
from ai_research_agent.trends._http import get_json

logger = logging.getLogger(__name__)

GITHUB_API_URL = "https://api.github.com/search/repositories"


def _headers() -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch(cfg: dict) -> list[TrendItem]:
    since = (datetime.now(timezone.utc) - timedelta(days=cfg["github_days"])).date().isoformat()
    floor = f"created:>{since} stars:>{cfg['min_github_stars']}"
    queries = [f"topic:{t} {floor}" for t in cfg["github_topics"]]
    queries += [f"{kw} in:name,description {floor}" for kw in cfg["github_keywords"]]

    seen: dict[str, TrendItem] = {}
    for query in queries:
        data = get_json(
            GITHUB_API_URL,
            params={"q": query, "sort": "stars", "order": "desc", "per_page": 30},
            headers=_headers(),
        )
        for repo in data.get("items", []):
            name = repo.get("full_name")
            if not name or name in seen:
                continue
            detail = (repo.get("description") or "").strip()[:200]
            topics = ", ".join(repo.get("topics") or [])
            if topics:
                detail = f"{detail} [topics: {topics}]".strip()
            created = None
            if repo.get("created_at"):
                created = datetime.fromisoformat(repo["created_at"].replace("Z", "+00:00"))
            seen[name] = TrendItem(
                source="github",
                title=name,
                url=repo.get("html_url") or f"https://github.com/{name}",
                score=repo.get("stargazers_count"),
                detail=detail,
                created_at=created,
            )
    items = sorted(seen.values(), key=lambda t: t.score or 0, reverse=True)
    logger.info("github: %d unique repos created since %s", len(items), since)
    return items[: cfg["max_items_per_source"]]
