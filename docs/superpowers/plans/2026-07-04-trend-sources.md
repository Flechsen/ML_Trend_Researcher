# Weekly ML Trends Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fail-soft weekly trends component (Hacker News, GitHub, Hugging Face, Reddit → one Sonnet synthesis → `trends/YYYY-MM-DD.md`) to the existing arXiv digest pipeline, leaving the paper pipeline unchanged.

**Architecture:** New `src/ai_research_agent/trends/` package: four independent fetchers returning `list[TrendItem]`, a shared private HTTP helper, and `report.py` which orchestrates fetch → single Sonnet call → Jinja2-rendered markdown file + INDEX. `main.py` calls it after the paper stages inside its own try/except.

**Tech Stack:** Python 3.12, httpx + tenacity, anthropic (Sonnet, existing pricing entry), Jinja2, pytest + respx + unittest.mock.

## Global Constraints

- Paper pipeline behavior must be byte-for-byte unchanged unless `--trends-only` is passed.
- `BUDGET_USD_CAP` stays `3.00`; trends synthesis charges `budget` with stage `"trends"`, model `"claude-sonnet-4-6"`.
- No new dependencies, no new secrets. `GH_TOKEN` used only if already present in env.
- Fetcher modules never import each other; they may import `models.py`, `trends/config.py`, `trends/_http.py`.
- ruff line-length 100, target py312. Run tests with `uv run pytest`.
- All new fetchers fail-soft at the `report.py` level: an exception from `fetch()` → warning + 0 items, never a crash.
- Reference spec: `docs/superpowers/specs/2026-07-04-trend-sources-design.md`.

---

### Task 1: TrendItem model + trends package + config loader

**Files:**
- Modify: `src/ai_research_agent/models.py` (append dataclass)
- Create: `src/ai_research_agent/trends/__init__.py` (empty)
- Create: `src/ai_research_agent/trends/config.py`
- Test: `tests/test_trend_sources.py` (new file, first tests)

**Interfaces:**
- Produces: `models.TrendItem(source, title, url, score, detail, created_at)` — frozen dataclass; `source` is `Literal["hackernews", "github", "hf_papers", "hf_models", "reddit"]`; `score: int | None`; `created_at: datetime | None`.
- Produces: `trends.config.load_config(interests: dict) -> dict` — merges optional `interests["trends"]` over `DEFAULT_CONFIG`. Keys: `hn_queries: list[str]`, `min_hn_points: int`, `github_topics: list[str]`, `github_keywords: list[str]`, `min_github_stars: int`, `github_days: int`, `subreddits: list[str]`, `max_items_per_source: int`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trend_sources.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_trend_sources.py -v`
Expected: FAIL with `ImportError: cannot import name 'TrendItem'`

- [ ] **Step 3: Implement**

Append to `src/ai_research_agent/models.py`:

```python
@dataclass(frozen=True)
class TrendItem:
    source: Literal["hackernews", "github", "hf_papers", "hf_models", "reddit"]
    title: str
    url: str
    score: int | None      # points / stars / upvotes / trendingScore; None if unavailable
    detail: str            # short context: description, comment count, subreddit, ...
    created_at: datetime | None
```

Create empty `src/ai_research_agent/trends/__init__.py`.

Create `src/ai_research_agent/trends/config.py`:

```python
DEFAULT_CONFIG: dict = {
    "hn_queries": [
        "LLM", "MCP", "AI agent", "Claude", "GPT", "RAG",
        "fine-tuning", "open model", "autoresearch", "agent skills",
    ],
    "min_hn_points": 40,
    "github_topics": ["mcp", "llm", "ai-agents", "rag", "llm-inference", "fine-tuning"],
    "github_keywords": ["mcp server", "llm agent", "agent skills"],
    "min_github_stars": 100,
    "github_days": 14,
    "subreddits": ["LocalLLaMA", "MachineLearning"],
    "max_items_per_source": 30,
}


def load_config(interests: dict) -> dict:
    """Merge the optional `trends:` block of interests.yaml over DEFAULT_CONFIG."""
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(interests.get("trends") or {})
    return cfg
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_trend_sources.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/ai_research_agent/models.py src/ai_research_agent/trends/ tests/test_trend_sources.py
git commit -m "feat(trends): TrendItem model + trends config loader"
```

---

### Task 2: Shared HTTP helper for trend fetchers

**Files:**
- Create: `src/ai_research_agent/trends/_http.py`
- Test: `tests/test_trend_sources.py` (append)

**Interfaces:**
- Produces: `_http.get_json(url: str, params: dict, headers: dict | None = None) -> dict | list` and `_http.get_text(url, params, headers=None) -> str`. Both retry up to 3 attempts on timeouts/transport errors/429/403/5xx with exponential backoff (2s→60s), `reraise=True`; other 4xx raise `httpx.HTTPStatusError` immediately.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_trend_sources.py`:

```python
import httpx
import pytest
import respx

from ai_research_agent.trends._http import get_json, get_text


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_trend_sources.py -v -k "get_json or get_text"`
Expected: FAIL with `ModuleNotFoundError: ai_research_agent.trends._http`

- [ ] **Step 3: Implement**

Create `src/ai_research_agent/trends/_http.py`:

```python
import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

DEFAULT_TIMEOUT = 30.0


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        # 403 included: GitHub signals secondary rate limits with 403.
        return code in (403, 429) or code >= 500
    return False


_RETRY = dict(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    reraise=True,
)


@retry(**_RETRY)
def get_json(url: str, params: dict, headers: dict | None = None):
    resp = httpx.get(url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


@retry(**_RETRY)
def get_text(url: str, params: dict, headers: dict | None = None) -> str:
    resp = httpx.get(url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_trend_sources.py -v`
Expected: 8 PASS (the 5xx-retry test sleeps ~2s — expected)

- [ ] **Step 5: Commit**

```bash
git add src/ai_research_agent/trends/_http.py tests/test_trend_sources.py
git commit -m "feat(trends): shared HTTP helper with retry policy"
```

---

### Task 3: Hacker News fetcher

**Files:**
- Create: `src/ai_research_agent/trends/hackernews.py`
- Create: `tests/fixtures/hn_search_response.json`
- Test: `tests/test_trend_sources.py` (append)

**Interfaces:**
- Consumes: `TrendItem`, `_http.get_json`, config keys `hn_queries`, `min_hn_points`, `max_items_per_source`.
- Produces: `hackernews.fetch(cfg: dict, days: int = 7) -> list[TrendItem]` — one Algolia query per `hn_queries` entry, server-side date filter (`numericFilters=created_at_i>cutoff` — `points` is NOT server-filterable, verified 2026-07-04), client-side points filter, dedup by `objectID`, sorted by points desc, capped.

- [ ] **Step 1: Create fixture**

Create `tests/fixtures/hn_search_response.json`:

```json
{
  "nbHits": 3,
  "hits": [
    {
      "objectID": "48769639",
      "title": "The Safari MCP server for web developers",
      "url": "https://example.com/safari-mcp",
      "points": 262,
      "num_comments": 148,
      "created_at_i": 1782600000
    },
    {
      "objectID": "48762862",
      "title": "Launch HN: Manufact (YC S25) - MCP Cloud",
      "url": null,
      "points": 108,
      "num_comments": 60,
      "created_at_i": 1782500000
    },
    {
      "objectID": "48700001",
      "title": "Story below the points threshold",
      "url": "https://example.com/low",
      "points": 12,
      "num_comments": 3,
      "created_at_i": 1782400000
    }
  ]
}
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_trend_sources.py`:

```python
import json
from pathlib import Path

from ai_research_agent.trends import hackernews

FIXTURES = Path(__file__).parent / "fixtures"


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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_trend_sources.py -v -k hn_fetch`
Expected: FAIL with `ImportError` (no `hackernews` module)

- [ ] **Step 4: Implement**

Create `src/ai_research_agent/trends/hackernews.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_trend_sources.py -v`
Expected: 11 PASS

- [ ] **Step 6: Commit**

```bash
git add src/ai_research_agent/trends/hackernews.py tests/fixtures/hn_search_response.json tests/test_trend_sources.py
git commit -m "feat(trends): Hacker News fetcher"
```

---

### Task 4: GitHub trending fetcher

**Files:**
- Create: `src/ai_research_agent/trends/github_trending.py`
- Create: `tests/fixtures/github_search_response.json`
- Test: `tests/test_trend_sources.py` (append)

**Interfaces:**
- Consumes: `TrendItem`, `_http.get_json`, config keys `github_topics`, `github_keywords`, `min_github_stars`, `github_days`, `max_items_per_source`; env `GH_TOKEN` (optional).
- Produces: `github_trending.fetch(cfg: dict) -> list[TrendItem]` — one search per topic (`topic:<t> created:><date> stars:><n>`) plus one per keyword (`<kw> in:name,description created:><date> stars:><n>`), dedup by `full_name`, sorted by stars desc, capped.

- [ ] **Step 1: Create fixture**

Create `tests/fixtures/github_search_response.json`:

```json
{
  "total_count": 2,
  "incomplete_results": false,
  "items": [
    {
      "full_name": "acme/mcp-router",
      "html_url": "https://github.com/acme/mcp-router",
      "description": "Route MCP requests across multiple servers",
      "stargazers_count": 1200,
      "topics": ["mcp", "llm"],
      "created_at": "2026-06-25T12:00:00Z"
    },
    {
      "full_name": "acme/agent-skills",
      "html_url": "https://github.com/acme/agent-skills",
      "description": null,
      "stargazers_count": 450,
      "topics": [],
      "created_at": "2026-06-28T09:30:00Z"
    }
  ]
}
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_trend_sources.py`:

```python
from ai_research_agent.trends import github_trending


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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_trend_sources.py -v -k github`
Expected: FAIL with `ImportError` (no `github_trending` module)

- [ ] **Step 4: Implement**

Create `src/ai_research_agent/trends/github_trending.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_trend_sources.py -v`
Expected: 13 PASS

- [ ] **Step 6: Commit**

```bash
git add src/ai_research_agent/trends/github_trending.py tests/fixtures/github_search_response.json tests/test_trend_sources.py
git commit -m "feat(trends): GitHub trending fetcher via search API"
```

---

### Task 5: Hugging Face fetcher (daily papers + trending models)

**Files:**
- Create: `src/ai_research_agent/trends/huggingface.py`
- Test: `tests/test_trend_sources.py` (append; payloads built dynamically because the paper-date filter is client-side)

**Interfaces:**
- Consumes: `TrendItem`, `_http.get_json`, config key `max_items_per_source`.
- Produces: `huggingface.fetch(cfg: dict, days: int = 7) -> list[TrendItem]` — daily papers (source `"hf_papers"`, filtered to last `days`, sorted by upvotes, capped) + trending models (source `"hf_models"`, API already sorted by `trendingScore`, capped by the `limit` param).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_trend_sources.py`:

```python
from datetime import timedelta

from ai_research_agent.trends import huggingface


def _hf_papers_payload(now):
    recent = (now - timedelta(days=2)).isoformat()
    stale = (now - timedelta(days=30)).isoformat()
    return [
        {
            "title": "AutoMem: Automated Learning of Memory",
            "publishedAt": recent,
            "paper": {"id": "2506.11111", "upvotes": 42, "summary": "Memory as a skill."},
        },
        {
            "title": "Old Paper",
            "publishedAt": stale,
            "paper": {"id": "2505.00001", "upvotes": 99, "summary": "Too old."},
        },
        {
            "title": "No-id entry (skipped)",
            "publishedAt": recent,
            "paper": {"upvotes": 7, "summary": "Missing id."},
        },
    ]


HF_MODELS_PAYLOAD = [
    {"id": "org/model-a", "likes": 3354, "trendingScore": 601, "pipeline_tag": "text-generation"},
    {"id": "org/model-b", "likes": 100, "trendingScore": 305, "pipeline_tag": None},
]


@respx.mock
def test_hf_fetch_combines_recent_papers_and_trending_models():
    now = datetime.now(timezone.utc)
    respx.get("https://huggingface.co/api/daily_papers").mock(
        return_value=httpx.Response(200, json=_hf_papers_payload(now))
    )
    respx.get("https://huggingface.co/api/models").mock(
        return_value=httpx.Response(200, json=HF_MODELS_PAYLOAD)
    )
    items = huggingface.fetch(dict(DEFAULT_CONFIG))
    papers = [i for i in items if i.source == "hf_papers"]
    models = [i for i in items if i.source == "hf_models"]
    assert len(papers) == 1  # stale + no-id entries dropped
    assert papers[0].url == "https://huggingface.co/papers/2506.11111"
    assert papers[0].score == 42
    assert len(models) == 2
    assert models[0].title == "org/model-a"
    assert models[0].url == "https://huggingface.co/org/model-a"
    assert "3354 likes" in models[0].detail
    assert "unknown task" in models[1].detail  # None pipeline_tag handled
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_trend_sources.py -v -k hf_fetch`
Expected: FAIL with `ImportError` (no `huggingface` module)

- [ ] **Step 3: Implement**

Create `src/ai_research_agent/trends/huggingface.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_trend_sources.py -v`
Expected: 14 PASS

- [ ] **Step 5: Commit**

```bash
git add src/ai_research_agent/trends/huggingface.py tests/test_trend_sources.py
git commit -m "feat(trends): Hugging Face papers + trending models fetcher"
```

---

### Task 6: Reddit fetcher (RSS)

**Files:**
- Create: `src/ai_research_agent/trends/reddit.py`
- Create: `tests/fixtures/reddit_top.rss`
- Test: `tests/test_trend_sources.py` (append)

**Interfaces:**
- Consumes: `TrendItem`, `_http.get_text`, config keys `subreddits`, `max_items_per_source`.
- Produces: `reddit.fetch(cfg: dict) -> list[TrendItem]` — `top.rss?t=week` per subreddit (Atom), `score=None` (RSS has no scores; JSON endpoints 403, verified 2026-07-04), per-subreddit fail-soft so one blocked sub doesn't lose the other.

- [ ] **Step 1: Create fixture**

Create `tests/fixtures/reddit_top.rss`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>top scoring links : LocalLLaMA</title>
  <entry>
    <title>New 8B model beats 70B on coding benchmarks</title>
    <link href="https://www.reddit.com/r/LocalLLaMA/comments/abc123/new_8b_model/"/>
    <updated>2026-07-01T10:00:00+00:00</updated>
  </entry>
  <entry>
    <title>I built an MCP server for local inference</title>
    <link href="https://www.reddit.com/r/LocalLLaMA/comments/def456/mcp_server/"/>
    <updated>2026-06-30T08:00:00+00:00</updated>
  </entry>
</feed>
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_trend_sources.py`:

```python
from ai_research_agent.trends import reddit


@respx.mock
def test_reddit_fetch_parses_atom_entries():
    respx.get("https://www.reddit.com/r/LocalLLaMA/top.rss").mock(
        return_value=httpx.Response(200, text=(FIXTURES / "reddit_top.rss").read_text())
    )
    cfg = dict(DEFAULT_CONFIG)
    cfg["subreddits"] = ["LocalLLaMA"]
    items = reddit.fetch(cfg)
    assert len(items) == 2
    assert items[0].source == "reddit"
    assert items[0].score is None  # RSS carries no scores
    assert items[0].url.startswith("https://www.reddit.com/r/LocalLLaMA/comments/")
    assert items[0].detail == "r/LocalLLaMA weekly top"


@respx.mock
def test_reddit_fetch_survives_one_blocked_subreddit(monkeypatch):
    monkeypatch.setattr(reddit, "SUB_DELAY_S", 0)  # skip the politeness delay in tests
    respx.get("https://www.reddit.com/r/LocalLLaMA/top.rss").mock(
        return_value=httpx.Response(404)
    )
    respx.get("https://www.reddit.com/r/MachineLearning/top.rss").mock(
        return_value=httpx.Response(200, text=(FIXTURES / "reddit_top.rss").read_text())
    )
    cfg = dict(DEFAULT_CONFIG)
    cfg["subreddits"] = ["LocalLLaMA", "MachineLearning"]
    items = reddit.fetch(cfg)
    assert len(items) == 2  # MachineLearning entries still returned
    assert all(i.detail == "r/MachineLearning weekly top" for i in items)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_trend_sources.py -v -k reddit`
Expected: FAIL with `ImportError` (no `reddit` module)

- [ ] **Step 4: Implement**

Create `src/ai_research_agent/trends/reddit.py`:

```python
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
```

Note for the blocked-subreddit test: monkeypatch the delay to keep tests fast — add `monkeypatch.setattr(reddit, "SUB_DELAY_S", 0)` as the first line of `test_reddit_fetch_survives_one_blocked_subreddit` (and add `monkeypatch` to its signature).

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_trend_sources.py -v`
Expected: 16 PASS

- [ ] **Step 6: Commit**

```bash
git add src/ai_research_agent/trends/reddit.py tests/fixtures/reddit_top.rss tests/test_trend_sources.py
git commit -m "feat(trends): Reddit RSS fetcher with per-subreddit fail-soft"
```

---

### Task 7: Trends report orchestrator + synthesis + template

**Files:**
- Create: `src/ai_research_agent/trends/report.py`
- Create: `src/ai_research_agent/templates/trends.md.j2`
- Test: `tests/test_trend_report.py` (new file)

**Interfaces:**
- Consumes: all four fetcher modules (called as `module.fetch(cfg)` so tests can patch `module.fetch`), `load_config`, `Budget.charge`, `TrendItem`.
- Produces: `report.generate(interests: dict, budget: Budget, trends_dir: Path) -> Path` — fetches all sources fail-soft, raises `report.TrendsError` if every source returned 0 items, otherwise one Sonnet call (model `"claude-sonnet-4-6"`, stage `"trends"`), writes `trends_dir/<YYYY-MM-DD>.md`, regenerates `trends_dir/INDEX.md`, returns the report path.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trend_report.py`:

```python
from datetime import datetime, timezone
from pathlib import Path
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_trend_report.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` (no `report` module)

- [ ] **Step 3: Implement**

Create `src/ai_research_agent/templates/trends.md.j2`:

```jinja
# ML Trends — Week of {{ date }}

{{ body }}

---
_Sources this week: {{ counts_line }}._
{% if failed %}_Failed sources: {{ failed | join(", ") }}._
{% endif %}
```

Create `src/ai_research_agent/trends/report.py`:

```python
"""Weekly trends report: fetch all sources fail-soft, synthesize one markdown digest."""
import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from anthropic import Anthropic
from jinja2 import Environment, FileSystemLoader

from ai_research_agent.budget import Budget
from ai_research_agent.models import TrendItem
from ai_research_agent.trends import github_trending, hackernews, huggingface, reddit
from ai_research_agent.trends.config import load_config

logger = logging.getLogger(__name__)

TRENDS_MODEL = "claude-sonnet-4-6"
TEMPLATE_DIR = Path(__file__).parent.parent / "templates"

# (display name, module) — modules referenced so tests can patch module.fetch
SOURCES = [
    ("hackernews", hackernews),
    ("github", github_trending),
    ("huggingface", huggingface),
    ("reddit", reddit),
]


class TrendsError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _client() -> Anthropic:
    return Anthropic()


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(loader=FileSystemLoader(TEMPLATE_DIR))


def _fetch_all(cfg: dict) -> tuple[list[TrendItem], dict[str, int], list[str]]:
    items: list[TrendItem] = []
    counts: dict[str, int] = {}
    failed: list[str] = []
    for name, module in SOURCES:
        try:
            got = module.fetch(cfg)
        except Exception as e:
            logger.warning("trend source %s failed: %s", name, e)
            got = []
            failed.append(name)
        counts[name] = len(got)
        items.extend(got)
    return items, counts, failed


def _build_prompt(interests: dict, items: list[TrendItem]) -> str:
    topics = "\n".join(
        f"- {i.get('topic', '')}: {' '.join((i.get('description') or '').split())}"
        for i in interests.get("interests", [])
    )
    by_source: dict[str, list[TrendItem]] = {}
    for item in items:
        by_source.setdefault(item.source, []).append(item)
    sections = []
    for source, group in by_source.items():
        lines = [
            f"- {it.title} | score={it.score if it.score is not None else 'n/a'}"
            f" | {it.detail} | {it.url}"
            for it in group
        ]
        sections.append(f"### {source} ({len(group)} items)\n" + "\n".join(lines))
    return (
        "You are writing the body of a weekly ML/AI trends report for one developer.\n"
        "Their interests:\n" + topics + "\n\n"
        "Below are this week's trending items from Hacker News, GitHub, Hugging Face "
        "and Reddit:\n\n" + "\n\n".join(sections) + "\n\n"
        "Write GitHub-flavored markdown with EXACTLY these sections:\n"
        "## TL;DR — at most 5 bullets with the week's biggest takeaways\n"
        "## Themes — one '###' subsection per cross-cutting theme (MCP ecosystem, agents, "
        "autoresearch, ... as the data warrants); explain what is happening and why it "
        "matters, citing items as [title](url)\n"
        "## Notable new tools & repos — bullet list, one line each, linked\n"
        "## Notable models & papers — bullet list, one line each, linked\n"
        "## Radar — 3-5 weak signals worth watching\n\n"
        "Rules: every claim must cite at least one provided item URL as a markdown link; "
        "never invent items or URLs; prioritize items matching the developer's interests; "
        "start your reply directly with '## TL;DR' and output nothing after Radar."
    )


def _synthesize(interests: dict, items: list[TrendItem], budget: Budget) -> str:
    prompt = _build_prompt(interests, items)
    resp = _client().messages.create(
        model=TRENDS_MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    budget.charge(
        "trends",
        TRENDS_MODEL,
        in_tok=resp.usage.input_tokens,
        out_tok=resp.usage.output_tokens,
    )
    return resp.content[0].text.strip()


def _regenerate_index(trends_dir: Path) -> None:
    rows = []
    for md in sorted(trends_dir.glob("*.md"), reverse=True):
        if md.name == "INDEX.md":
            continue
        title = md.stem
        first_line = md.read_text().splitlines()[0] if md.exists() else ""
        if first_line.startswith("# "):
            title = first_line[2:].strip()
        rows.append(f"- [{title}]({md.name})")
    body = "# Trends Index\n\n" + "\n".join(rows) + "\n" if rows else "# Trends Index\n\n_(empty)_\n"
    (trends_dir / "INDEX.md").write_text(body)


def generate(interests: dict, budget: Budget, trends_dir: Path) -> Path:
    """Fetch all trend sources, synthesize the weekly report, write it. Charges `budget`."""
    cfg = load_config(interests)
    items, counts, failed = _fetch_all(cfg)
    if not items:
        raise TrendsError(f"all trend sources returned 0 items (failed: {failed or 'none'})")

    body = _synthesize(interests, items, budget)

    date = datetime.now(timezone.utc).date().isoformat()
    counts_line = " · ".join(f"{name}: {n}" for name, n in counts.items())
    markdown = _env().get_template("trends.md.j2").render(
        date=date, body=body, counts_line=counts_line, failed=failed,
    )
    trends_dir.mkdir(parents=True, exist_ok=True)
    path = trends_dir / f"{date}.md"
    path.write_text(markdown)
    _regenerate_index(trends_dir)
    logger.info("trends: wrote %s (%d items, %d sources failed)", path, len(items), len(failed))
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_trend_report.py tests/test_trend_sources.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/ai_research_agent/trends/report.py src/ai_research_agent/templates/trends.md.j2 tests/test_trend_report.py
git commit -m "feat(trends): report orchestrator, Sonnet synthesis, markdown template"
```

---

### Task 8: Wire into main.py + workflow + config + docs

**Files:**
- Modify: `src/ai_research_agent/main.py`
- Modify: `.github/workflows/weekly.yml`
- Modify: `interests.yaml` (append `trends:` block)
- Modify: `README.md`
- Test: `tests/test_main.py` (append)

**Interfaces:**
- Consumes: `report.generate(interests, budget, trends_dir) -> Path`, `report.TrendsError`, `open_failure_issue(stage, exc, run_url, repo)`.
- Produces: CLI flags `--skip-trends`, `--trends-only`, `--trends-dir` (default `"trends"`); `main._run_trends(interests, budget, trends_dir, repo, run_url) -> str` returning a human-readable note for the step summary.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_main.py`:

```python
def test_run_trends_returns_note_on_success(tmp_path):
    fake_path = tmp_path / "2026-07-04.md"
    with patch("ai_research_agent.main.generate_trends_report", return_value=fake_path) as gen:
        from ai_research_agent.main import _run_trends
        note = _run_trends({"interests": []}, MagicMock(), tmp_path, "", "")
    assert gen.called
    assert "2026-07-04.md" in note


def test_run_trends_failure_opens_issue_and_reports(tmp_path):
    with (
        patch("ai_research_agent.main.generate_trends_report",
              side_effect=RuntimeError("boom")),
        patch("ai_research_agent.main.open_failure_issue") as issue,
    ):
        from ai_research_agent.main import _run_trends
        note = _run_trends({"interests": []}, MagicMock(), tmp_path, "foo/bar", "http://run")
    assert "FAILED" in note
    assert issue.call_count == 1
    assert issue.call_args.kwargs["stage"] == "trends"


def test_trends_only_skips_paper_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    interests = tmp_path / "interests.yaml"
    interests.write_text("interests: []\n")
    report_path = tmp_path / "trends" / "2026-07-04.md"
    with (
        patch("ai_research_agent.main.generate_trends_report", return_value=report_path),
        patch("ai_research_agent.main.fetch_recent") as fetch,
    ):
        rc = run([
            "--trends-only",
            "--interests", str(interests),
            "--papers-dir", str(tmp_path / "papers"),
            "--trends-dir", str(tmp_path / "trends"),
        ])
    assert rc == 0
    assert not fetch.called


def test_trends_only_returns_one_on_failure(tmp_path):
    interests = tmp_path / "interests.yaml"
    interests.write_text("interests: []\n")
    with patch("ai_research_agent.main.generate_trends_report",
               side_effect=RuntimeError("boom")):
        rc = run([
            "--trends-only",
            "--interests", str(interests),
            "--papers-dir", str(tmp_path / "papers"),
            "--trends-dir", str(tmp_path / "trends"),
        ])
    assert rc == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_main.py -v -k trends`
Expected: FAIL with `ImportError` (`generate_trends_report` / `_run_trends` not defined)

- [ ] **Step 3: Implement `main.py` changes**

Add import (after the existing imports):

```python
from ai_research_agent.trends.report import generate as generate_trends_report
```

Add helper (after `_validate_config`):

```python
def _run_trends(
    interests: dict,
    budget: Budget,
    trends_dir: Path,
    repo: str,
    run_url: str,
) -> str:
    """Run the trends stage fail-soft; the paper digest never depends on this."""
    try:
        path = generate_trends_report(interests, budget, trends_dir)
        logger.info("[trends] wrote %s", path)
        return f"wrote {path.name}"
    except Exception as e:
        logger.warning("[trends] failed: %s", e)
        if repo and run_url:
            open_failure_issue(stage="trends", exc=e, run_url=run_url, repo=repo)
        return "FAILED (issue opened)" if repo and run_url else "FAILED"
```

Add CLI flags in `run()` (next to `--dry-run`):

```python
    p.add_argument("--trends-dir", default="trends")
    p.add_argument("--skip-trends", action="store_true",
                   help="Run the paper pipeline only.")
    p.add_argument("--trends-only", action="store_true",
                   help="Skip the paper pipeline; only generate the trends report.")
```

Add the `--trends-only` early branch in `run()`, right after `counters = {...}` is defined (before `try:`):

```python
    if args.trends_only:
        note = _run_trends(interests, budget, Path(args.trends_dir), repo, run_url)
        _step_summary("## Weekly Trends — " + datetime.now(timezone.utc).date().isoformat())
        _step_summary(f"Trends: {note} — spend ${budget.spent:.2f}")
        return 0 if not note.startswith("FAILED") else 1
```

Inside the main `try:`, after `logger.info("[stage 6/6] regenerated INDEX.md")` and before the job-summary block:

```python
        trends_note = "skipped (--skip-trends)"
        if not args.skip_trends:
            trends_note = _run_trends(interests, budget, Path(args.trends_dir), repo, run_url)
```

Add a row to the job-summary table (after the `Total spend` row):

```python
        _step_summary(f"| Trends report | {trends_note} |")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_main.py -v`
Expected: all PASS

- [ ] **Step 5: Update workflow, interests.yaml, README**

`.github/workflows/weekly.yml` — three edits:

1. `name: Weekly arXiv Digest` → `name: Weekly AI Digest`
2. Commit step, replace the porcelain check and add:

```yaml
          if [[ -n "$(git status --porcelain papers/ trends/)" ]]; then
            git add papers/ trends/
            git commit -m "weekly digest: $(date -u +%Y-%m-%d)"
            git push
          else
            echo "No new papers or trends this week."
          fi
```

`interests.yaml` — append:

```yaml
trends:
  hn_queries: ["LLM", "MCP", "AI agent", "Claude", "GPT", "RAG",
               "fine-tuning", "open model", "autoresearch", "agent skills"]
  min_hn_points: 40
  github_topics: [mcp, llm, ai-agents, rag, llm-inference, fine-tuning]
  github_keywords: ["mcp server", "llm agent", "agent skills"]
  min_github_stars: 100
  github_days: 14
  subreddits: [LocalLLaMA, MachineLearning]
  max_items_per_source: 30
```

`README.md` — update the pipeline section to:

```markdown
## Pipeline

```
arXiv ── prefilter (embeddings) ── rank (Haiku) ── resolve repo ── synthesize (Sonnet) ── commit
                                                                                            │
HN ─┬─ GitHub ─┬─ Hugging Face ─┬─ Reddit ──── synthesize trends (Sonnet) ── trends/<date>.md
    └──────────┴────────────────┴── (each source fail-soft; report survives partial outages)
```

Papers land in `papers/<year>/<arxiv-id>-<slug>.md`; the weekly trends report lands in
`trends/<YYYY-MM-DD>.md` (themes: MCP ecosystem, agents, autoresearch, new models, tooling —
an indirect view of what's trending on X, sourced from HN / GitHub / HF / Reddit free APIs).

```bash
# Trends report only (fast prompt iteration, ~$0.15/run)
ANTHROPIC_API_KEY=... uv run python -m ai_research_agent.main --trends-only

# Paper pipeline only
ANTHROPIC_API_KEY=... OPENAI_API_KEY=... uv run python -m ai_research_agent.main --skip-trends
```

Trend sources are configured in the `trends:` block of `interests.yaml` (queries, subreddits,
star/point floors). Reddit is fetched via RSS and may be blocked from CI IPs — the report
just runs on the remaining sources and says so in its footer.
```

Also update the cost line in README "Cost controls": expected weekly spend ~$0.70 (papers ~$0.55 + trends ~$0.15), worst case unchanged at cap $3.00.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add src/ai_research_agent/main.py .github/workflows/weekly.yml interests.yaml README.md tests/test_main.py
git commit -m "feat(trends): wire trends stage into weekly run + workflow + docs"
```

---

### Task 9: Live verification

**Files:** none created (verification only; a scratch report may be written to a temp dir)

- [ ] **Step 1: Live smoke test of all four fetchers (keyless, free)**

Run a scratch script with `uv run python` that calls each fetcher with `load_config(yaml.safe_load(open("interests.yaml")))` and prints `len(items)` + 3 sample titles per source.

Expected: HN, GitHub, HF all return > 0 items. Reddit may return 0 from blocked IPs — acceptable only via the fail-soft path (warning logged, no exception).

- [ ] **Step 2: Full test suite + ruff**

Run: `uv run pytest && uv run ruff check src tests`
Expected: all PASS, no lint errors

- [ ] **Step 3: End-to-end `--trends-only` run (needs ANTHROPIC_API_KEY)**

If `ANTHROPIC_API_KEY` is available locally:

Run: `uv run python -m ai_research_agent.main --trends-only --trends-dir /tmp-scratch/trends-test`
Expected: exit 0, a dated `.md` with TL;DR/Themes/Radar sections and a footer with per-source counts. Inspect the report for sanity (real links, no invented items).

If no key is available locally: skip — mocked tests + first CI run cover it; note this in the final report to the user.

- [ ] **Step 4: Verify paper pipeline untouched**

Run: `git diff HEAD~7 --stat -- src/ai_research_agent/arxiv_client.py src/ai_research_agent/prefilter.py src/ai_research_agent/ranker.py src/ai_research_agent/repo_resolver.py src/ai_research_agent/synthesizer.py src/ai_research_agent/pdf_parser.py`
Expected: empty (zero changes to paper-stage modules)

- [ ] **Step 5: Final commit if anything is uncommitted**

```bash
git status --porcelain && git add -A && git commit -m "chore(trends): verification artifacts" || echo "clean"
```
