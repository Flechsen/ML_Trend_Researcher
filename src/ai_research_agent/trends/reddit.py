"""Reddit trend fetcher via RSS.

Reddit's public JSON endpoints return 403 to non-browser clients (verified
2026-07-04); the Atom RSS feeds still work at low request rates. RSS carries
no vote counts, so score is always None. May also be blocked from cloud IPs —
callers treat this source as best-effort.
"""
import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime

from ai_research_agent.models import TrendItem
from ai_research_agent.trends._http import get_text

logger = logging.getLogger(__name__)

REDDIT_RSS_URL = "https://www.reddit.com/r/{sub}/top.rss"
USER_AGENT = "ml-trend-researcher/0.1 (weekly research digest)"
NS = {"atom": "http://www.w3.org/2005/Atom"}
SUB_DELAY_S = 5.0  # be gentle: rapid successive requests get 429ed


def _parse_feed(xml_text: str, sub: str) -> list[TrendItem]:
    items: list[TrendItem] = []
    root = ET.fromstring(xml_text)
    for entry in root.findall("atom:entry", NS):
        title = (entry.findtext("atom:title", default="", namespaces=NS) or "").strip()
        link_el = entry.find("atom:link", NS)
        url = link_el.attrib.get("href", "") if link_el is not None else ""
        updated = entry.findtext("atom:updated", default="", namespaces=NS) or ""
        created = None
        if updated:
            try:
                created = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            except ValueError:
                created = None
        if not title or not url:
            continue
        items.append(TrendItem(
            source="reddit",
            title=title,
            url=url,
            score=None,
            detail=f"r/{sub} weekly top",
            created_at=created,
        ))
    return items


def fetch(cfg: dict) -> list[TrendItem]:
    items: list[TrendItem] = []
    for i, sub in enumerate(cfg["subreddits"]):
        if i:
            time.sleep(SUB_DELAY_S)
        try:
            xml_text = get_text(
                REDDIT_RSS_URL.format(sub=sub),
                params={"t": "week", "limit": 30},
                headers={"User-Agent": USER_AGENT},
            )
            items.extend(_parse_feed(xml_text, sub))
        except Exception as e:
            logger.warning("reddit: r/%s failed (%s) — continuing with other subs", sub, e)
    logger.info("reddit: %d posts from %d subreddits", len(items), len(cfg["subreddits"]))
    return items[: cfg["max_items_per_source"]]
