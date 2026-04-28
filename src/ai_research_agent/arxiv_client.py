import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ai_research_agent.models import Paper

ARXIV_API_URL = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_ID_RE = re.compile(r"arxiv\.org/abs/(?P<base>\d{4}\.\d{4,5})(?P<ver>v\d+)?")


def _parse_atom(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    papers: list[Paper] = []
    for entry in root.findall("atom:entry", NS):
        id_text = (entry.findtext("atom:id", default="", namespaces=NS) or "").strip()
        m = ARXIV_ID_RE.search(id_text)
        if not m:
            continue
        arxiv_id = m.group("base")
        version = m.group("ver") or "v1"

        title = (entry.findtext("atom:title", default="", namespaces=NS) or "").strip()
        abstract = (entry.findtext("atom:summary", default="", namespaces=NS) or "").strip()
        published = datetime.fromisoformat(
            entry.findtext("atom:published", default="", namespaces=NS).replace("Z", "+00:00")
        )
        updated = datetime.fromisoformat(
            entry.findtext("atom:updated", default="", namespaces=NS).replace("Z", "+00:00")
        )

        authors = [
            (a.findtext("atom:name", default="", namespaces=NS) or "").strip()
            for a in entry.findall("atom:author", NS)
        ]
        categories = [
            c.attrib.get("term", "")
            for c in entry.findall("atom:category", NS)
        ]
        pdf_url = ""
        arxiv_url = ""
        for link in entry.findall("atom:link", NS):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")
            elif link.attrib.get("rel") == "alternate":
                arxiv_url = link.attrib.get("href", "")

        papers.append(Paper(
            arxiv_id=arxiv_id,
            version=version,
            title=title,
            authors=authors,
            abstract=abstract,
            published=published,
            updated=updated,
            categories=categories,
            pdf_url=pdf_url,
            arxiv_url=arxiv_url,
        ))
    return papers


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=10, min=15, max=120),
    reraise=True,
)
def _http_get(url: str, params: dict) -> str:
    resp = httpx.get(url, params=params, timeout=60.0)
    resp.raise_for_status()
    return resp.text


def fetch_recent(
    categories: list[str],
    days: int,
    existing_ids: set[str],
    max_results: int = 2000,
) -> list[Paper]:
    """Query arXiv for papers in `categories` updated within the last `days`,
    drop any whose arxiv_id is already in `existing_ids`."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    cat_clause = " OR ".join(f"cat:{c}" for c in categories)
    date_clause = (
        f"lastUpdatedDate:[{start.strftime('%Y%m%d%H%M')} TO "
        f"{end.strftime('%Y%m%d%H%M')}]"
    )
    params = {
        "search_query": f"({cat_clause}) AND {date_clause}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    xml_text = _http_get(ARXIV_API_URL, params)
    papers = _parse_atom(xml_text)
    return [p for p in papers if p.arxiv_id not in existing_ids]
