# AI Research Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a weekly automated GitHub Actions job that fetches AI/LLM papers from arXiv, ranks them for personal relevance and MVP-implementability, and commits LLM-generated implementation-plan markdown files to a private repo.

**Architecture:** Linear pipeline of pure-function stages (`fetch_arxiv → prefilter_embed → rank_llm → resolve_repo → synthesize_md → commit`). Each stage is its own module with one public function and an optional CLI; stages communicate via dataclasses defined in `models.py`. Cost is bounded by an in-process `Budget` object plus account-level vendor billing limits. Failures route to GitHub issues.

**Tech Stack:**
- Python 3.12, `uv` for packaging
- arXiv export API (XML)
- OpenAI `text-embedding-3-small` (semantic pre-filter)
- Anthropic Claude Haiku 4.5 (ranker), Claude Sonnet 4.6 (markdown synthesis)
- `pdfplumber` (PDF parsing), `tiktoken` (token counting), `jinja2` (templates)
- `pytest` + `pytest-httpx` + `respx` + `pyfakefs` (testing)
- GitHub Actions cron + `gh` CLI

**Reference spec:** `docs/superpowers/specs/2026-04-26-ai-research-agent-design.md`

---

## Pre-flight: working directory

All paths in this plan are relative to `/Users/benediktflechsenhar/Desktop/AI_Research_Agent/`. Before starting, restart Claude from that directory (the previous session had a broken CWD reference). Verify with:

```bash
pwd
# /Users/benediktflechsenhar/Desktop/AI_Research_Agent
ls
# (empty or only docs/)
```

---

## Task 1: Project scaffold + git init

**Goal:** Create the package skeleton, declare dependencies, initialize git.

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/ai_research_agent/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/fixtures/.gitkeep`

- [ ] **Step 1: Create the directory layout**

```bash
mkdir -p src/ai_research_agent/templates
mkdir -p tests/fixtures
mkdir -p papers
mkdir -p .github/workflows
touch src/ai_research_agent/__init__.py
touch tests/__init__.py
touch tests/fixtures/.gitkeep
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "ai-research-agent"
version = "0.1.0"
description = "Weekly arXiv digest -> implementation-plan markdown"
requires-python = ">=3.12"
dependencies = [
  "anthropic>=0.40",
  "openai>=1.50",
  "httpx>=0.27",
  "pdfplumber>=0.11",
  "tiktoken>=0.8",
  "jinja2>=3.1",
  "pyyaml>=6.0",
  "tenacity>=9.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "pytest-httpx>=0.30",
  "respx>=0.22",
  "pyfakefs>=5.7",
  "ruff>=0.7",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ai_research_agent"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"

[tool.ruff]
line-length = 100
target-version = "py312"
```

- [ ] **Step 3: Write `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.pytest_cache/
.ruff_cache/

# uv
uv.lock

# Local secrets / scratch
.env
.env.local
/tmp/

# IDE
.vscode/
.idea/
```

- [ ] **Step 4: Install dependencies**

```bash
uv sync --all-extras
```

Expected: creates `.venv/`, installs everything in `pyproject.toml`.

- [ ] **Step 5: Verify the package imports**

```bash
uv run python -c "import ai_research_agent; print('ok')"
```

Expected output: `ok`

- [ ] **Step 6: Run pytest on an empty test suite**

```bash
uv run pytest
```

Expected: `no tests ran in <time>` exit 5 (no tests collected) — that's fine.

- [ ] **Step 7: git init and first commit**

```bash
git init
git add .
git commit -m "chore: scaffold package + dependencies"
```

---

## Task 2: Dataclasses (`models.py`)

**Goal:** Define all stage-boundary types in one place. No logic, just shapes.

**Files:**
- Create: `src/ai_research_agent/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write a failing import test**

`tests/test_models.py`:

```python
from datetime import datetime, timezone
from ai_research_agent.models import Paper, ScoredPaper, RankedCandidate, RepoBundle


def test_paper_is_frozen():
    p = Paper(
        arxiv_id="2404.12345",
        version="v1",
        title="Test",
        authors=["Alice"],
        abstract="abs",
        published=datetime(2026, 4, 20, tzinfo=timezone.utc),
        updated=datetime(2026, 4, 20, tzinfo=timezone.utc),
        categories=["cs.AI"],
        pdf_url="https://arxiv.org/pdf/2404.12345",
        arxiv_url="https://arxiv.org/abs/2404.12345",
    )
    import dataclasses
    assert dataclasses.is_dataclass(p)
    try:
        p.title = "Changed"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("Paper should be frozen")


def test_repo_bundle_kinds():
    rb = RepoBundle(
        repo_url="https://github.com/foo/bar",
        repo_kind="github",
        readme="# foo",
        file_tree=["README.md", "src/main.py"],
        truncated=False,
    )
    assert rb.repo_kind in ("github", "huggingface")


def test_ranked_candidate_score_range():
    # Just test we can construct one with a valid score
    rc = RankedCandidate(
        paper=Paper(
            arxiv_id="2404.12345", version="v1", title="t", authors=[], abstract="",
            published=datetime(2026, 4, 20, tzinfo=timezone.utc),
            updated=datetime(2026, 4, 20, tzinfo=timezone.utc),
            categories=[], pdf_url="", arxiv_url="",
        ),
        embedding_score=0.5,
        llm_score=8,
        llm_reasoning="solid fit",
        has_repo_url_in_abstract=True,
    )
    assert 1 <= rc.llm_score <= 10
```

- [ ] **Step 2: Run tests, expect failure**

```bash
uv run pytest tests/test_models.py
```

Expected: ImportError — `ai_research_agent.models` doesn't exist.

- [ ] **Step 3: Implement `models.py`**

`src/ai_research_agent/models.py`:

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    version: str
    title: str
    authors: list[str]
    abstract: str
    published: datetime
    updated: datetime
    categories: list[str]
    pdf_url: str
    arxiv_url: str


@dataclass(frozen=True)
class ScoredPaper:
    paper: Paper
    embedding_score: float


@dataclass(frozen=True)
class RankedCandidate:
    paper: Paper
    embedding_score: float
    llm_score: int  # 1-10
    llm_reasoning: str
    has_repo_url_in_abstract: bool


@dataclass(frozen=True)
class RepoBundle:
    repo_url: str
    repo_kind: Literal["github", "huggingface"]
    readme: str
    file_tree: list[str]
    truncated: bool
```

- [ ] **Step 4: Run tests, expect pass**

```bash
uv run pytest tests/test_models.py
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ai_research_agent/models.py tests/test_models.py
git commit -m "feat(models): define stage-boundary dataclasses"
```

---

## Task 3: Budget tracking (`budget.py`)

**Goal:** Pure-Python cost ledger that aborts above a USD cap.

**Files:**
- Create: `src/ai_research_agent/budget.py`
- Create: `tests/test_budget.py`

- [ ] **Step 1: Write failing tests**

`tests/test_budget.py`:

```python
import pytest
from ai_research_agent.budget import Budget, BudgetExceeded, PRICING


def test_pricing_table_has_required_models():
    assert "claude-sonnet-4-6" in PRICING
    assert "claude-haiku-4-5" in PRICING
    assert "text-embedding-3-small" in PRICING


def test_charge_accumulates_cost():
    b = Budget(cap_usd=1.0)
    b.charge("rank", "claude-haiku-4-5", in_tok=1000, out_tok=100)
    # Haiku: 1000 * $1e-6 + 100 * $5e-6 = $0.0015
    assert abs(b.spent - 0.0015) < 1e-9
    assert len(b.calls) == 1
    assert b.calls[0][0] == "rank"


def test_charge_records_per_stage():
    b = Budget(cap_usd=10.0)
    b.charge("rank", "claude-haiku-4-5", 1000, 100)
    b.charge("synthesize", "claude-sonnet-4-6", 1000, 100)
    stages = [c[0] for c in b.calls]
    assert stages == ["rank", "synthesize"]


def test_charge_raises_above_cap():
    b = Budget(cap_usd=0.001)
    with pytest.raises(BudgetExceeded) as ei:
        b.charge("rank", "claude-haiku-4-5", in_tok=1000, out_tok=1000)
    assert "$" in str(ei.value)


def test_charge_records_cost_even_when_raising():
    b = Budget(cap_usd=0.001)
    try:
        b.charge("rank", "claude-haiku-4-5", in_tok=1000, out_tok=1000)
    except BudgetExceeded:
        pass
    # Even after exceeding, the failing call should be recorded for visibility
    assert len(b.calls) == 1
    assert b.spent > 0.001


def test_unknown_model_raises_keyerror():
    b = Budget(cap_usd=1.0)
    with pytest.raises(KeyError):
        b.charge("rank", "claude-not-real", 100, 100)
```

- [ ] **Step 2: Run tests, expect ImportError**

```bash
uv run pytest tests/test_budget.py
```

- [ ] **Step 3: Implement `budget.py`**

`src/ai_research_agent/budget.py`:

```python
from dataclasses import dataclass, field

PRICING: dict[str, dict[str, float]] = {
    # USD per token
    "claude-sonnet-4-6":      {"in": 3.00e-6, "out": 15.00e-6},
    "claude-haiku-4-5":       {"in": 1.00e-6, "out":  5.00e-6},
    "text-embedding-3-small": {"in": 0.02e-6, "out":  0.0     },
}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    cap_usd: float
    spent: float = 0.0
    calls: list[tuple[str, str, int, int, float]] = field(default_factory=list)

    def charge(self, stage: str, model: str, in_tok: int, out_tok: int) -> None:
        p = PRICING[model]  # KeyError on unknown model
        cost = in_tok * p["in"] + out_tok * p["out"]
        self.spent += cost
        self.calls.append((stage, model, in_tok, out_tok, cost))
        if self.spent > self.cap_usd:
            raise BudgetExceeded(
                f"Budget exceeded: spent ${self.spent:.4f} > cap ${self.cap_usd:.4f}"
            )

    def report(self) -> str:
        lines = [f"Total: ${self.spent:.4f} ({len(self.calls)} calls)"]
        for stage, model, in_t, out_t, cost in self.calls:
            lines.append(f"  {stage:12s} {model:24s} in={in_t} out={out_t} ${cost:.4f}")
        return "\n".join(lines)
```

- [ ] **Step 4: Run tests, expect pass**

```bash
uv run pytest tests/test_budget.py
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ai_research_agent/budget.py tests/test_budget.py
git commit -m "feat(budget): per-run cost ledger with hard cap"
```

---

## Task 4: arXiv client (`arxiv_client.py`)

**Goal:** Query arXiv export API, parse Atom XML, dedup against existing papers/.

**Files:**
- Create: `src/ai_research_agent/arxiv_client.py`
- Create: `tests/fixtures/arxiv_response.xml`
- Create: `tests/test_arxiv_client.py`

- [ ] **Step 1: Add a fixture XML response**

`tests/fixtures/arxiv_response.xml` — a minimal but realistic Atom feed with 3 entries (use the structure from `https://export.arxiv.org/api/query?search_query=cat:cs.AI&max_results=3`):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <opensearch:totalResults xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">3</opensearch:totalResults>
  <opensearch:startIndex xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">0</opensearch:startIndex>
  <entry>
    <id>http://arxiv.org/abs/2404.12345v2</id>
    <updated>2026-04-22T10:00:00Z</updated>
    <published>2026-04-20T08:00:00Z</published>
    <title>ReAct Tool Use for LLM Agents</title>
    <summary>We propose a tool-using agent. Code at github.com/foo/bar.</summary>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <link href="http://arxiv.org/abs/2404.12345v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2404.12345v2" rel="related" type="application/pdf"/>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2404.12399v1</id>
    <updated>2026-04-21T10:00:00Z</updated>
    <published>2026-04-21T10:00:00Z</published>
    <title>Speculative Decoding Survey</title>
    <summary>Survey of speculative decoding methods.</summary>
    <author><name>Carol Davis</name></author>
    <link href="http://arxiv.org/abs/2404.12399v1" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2404.12399v1" rel="related" type="application/pdf"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2404.12500v1</id>
    <updated>2026-04-22T10:00:00Z</updated>
    <published>2026-04-22T10:00:00Z</published>
    <title>Hybrid RAG with Rerank</title>
    <summary>RAG plus reranker. Repo: huggingface.co/foo/bar.</summary>
    <author><name>Dan Eve</name></author>
    <link href="http://arxiv.org/abs/2404.12500v1" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2404.12500v1" rel="related" type="application/pdf"/>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>
```

- [ ] **Step 2: Write failing tests**

`tests/test_arxiv_client.py`:

```python
from datetime import timezone
from pathlib import Path
import respx
import httpx
from ai_research_agent.arxiv_client import fetch_recent, _parse_atom


FIXTURE = Path(__file__).parent / "fixtures" / "arxiv_response.xml"


def test_parse_atom_extracts_three_entries():
    papers = _parse_atom(FIXTURE.read_text())
    assert len(papers) == 3
    p = papers[0]
    assert p.arxiv_id == "2404.12345"
    assert p.version == "v2"
    assert p.title == "ReAct Tool Use for LLM Agents"
    assert p.authors == ["Alice Smith", "Bob Jones"]
    assert "tool-using agent" in p.abstract
    assert p.categories == ["cs.AI", "cs.CL"]
    assert p.pdf_url == "http://arxiv.org/pdf/2404.12345v2"
    assert p.published.tzinfo == timezone.utc


def test_parse_atom_strips_arxiv_version_from_id():
    papers = _parse_atom(FIXTURE.read_text())
    # Entry 1 was v2 in the XML id; parsed id should be base
    assert papers[0].arxiv_id == "2404.12345"
    assert papers[0].version == "v2"


@respx.mock
def test_fetch_recent_calls_arxiv_and_dedups(tmp_path):
    respx.get("https://export.arxiv.org/api/query").mock(
        return_value=httpx.Response(200, text=FIXTURE.read_text())
    )
    # Pretend 2404.12399 has already been processed
    existing_ids = {"2404.12399"}
    papers = fetch_recent(
        categories=["cs.AI", "cs.CL", "cs.LG"],
        days=7,
        existing_ids=existing_ids,
    )
    ids = [p.arxiv_id for p in papers]
    assert "2404.12345" in ids
    assert "2404.12500" in ids
    assert "2404.12399" not in ids  # deduped


@respx.mock
def test_fetch_recent_retries_on_5xx(tmp_path):
    route = respx.get("https://export.arxiv.org/api/query").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(503),
            httpx.Response(200, text=FIXTURE.read_text()),
        ]
    )
    papers = fetch_recent(categories=["cs.AI"], days=7, existing_ids=set())
    assert len(papers) == 3
    assert route.call_count == 3
```

- [ ] **Step 3: Run tests, expect ImportError**

```bash
uv run pytest tests/test_arxiv_client.py
```

- [ ] **Step 4: Implement `arxiv_client.py`**

`src/ai_research_agent/arxiv_client.py`:

```python
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
        # Find PDF URL among <link> elements
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
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
def _http_get(url: str, params: dict) -> str:
    resp = httpx.get(url, params=params, timeout=30.0)
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
    cat_clause = "+OR+".join(f"cat:{c}" for c in categories)
    date_clause = (
        f"lastUpdatedDate:[{start.strftime('%Y%m%d%H%M')}+TO+"
        f"{end.strftime('%Y%m%d%H%M')}]"
    )
    params = {
        "search_query": f"({cat_clause})+AND+{date_clause}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    xml_text = _http_get(ARXIV_API_URL, params)
    papers = _parse_atom(xml_text)
    return [p for p in papers if p.arxiv_id not in existing_ids]
```

- [ ] **Step 5: Run tests, expect pass**

```bash
uv run pytest tests/test_arxiv_client.py
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ai_research_agent/arxiv_client.py tests/test_arxiv_client.py tests/fixtures/arxiv_response.xml
git commit -m "feat(arxiv): query API, parse Atom XML, dedup against existing"
```

---

## Task 5: Embedding pre-filter (`prefilter.py`)

**Goal:** Use OpenAI embeddings to rank candidates by cosine similarity to interests text, return top-N.

**Files:**
- Create: `src/ai_research_agent/prefilter.py`
- Create: `tests/test_prefilter.py`

- [ ] **Step 1: Write failing tests**

`tests/test_prefilter.py`:

```python
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from ai_research_agent.models import Paper
from ai_research_agent.prefilter import score_by_embedding, _cosine, _interests_to_embedding_text


def make_paper(arxiv_id: str, abstract: str) -> Paper:
    return Paper(
        arxiv_id=arxiv_id, version="v1", title="t", authors=[], abstract=abstract,
        published=datetime(2026, 4, 20, tzinfo=timezone.utc),
        updated=datetime(2026, 4, 20, tzinfo=timezone.utc),
        categories=[], pdf_url="", arxiv_url="",
    )


def test_cosine_identical_vectors_is_one():
    assert abs(_cosine([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) - 1.0) < 1e-9


def test_cosine_orthogonal_is_zero():
    assert abs(_cosine([1.0, 0.0], [0.0, 1.0])) < 1e-9


def test_interests_to_embedding_text_includes_topics_and_examples():
    interests = {
        "interests": [
            {"topic": "agents", "description": "tool use", "examples": ["ReAct"], "anti_examples": []},
        ],
    }
    text = _interests_to_embedding_text(interests)
    assert "agents" in text
    assert "tool use" in text
    assert "ReAct" in text


def test_score_by_embedding_returns_top_n_sorted():
    interests = {"interests": [{"topic": "agents", "description": "tool use", "examples": [], "anti_examples": []}]}
    papers = [
        make_paper("1", "tool-using agent"),
        make_paper("2", "unrelated topic about gardens"),
        make_paper("3", "another agent paper"),
    ]
    fake_resp = MagicMock()
    # interests vector first, then 3 abstract vectors
    fake_resp.data = [
        MagicMock(embedding=[1.0, 0.0]),  # interests
        MagicMock(embedding=[1.0, 0.0]),  # paper 1, perfect match
        MagicMock(embedding=[0.0, 1.0]),  # paper 2, orthogonal
        MagicMock(embedding=[0.9, 0.1]),  # paper 3, close
    ]
    fake_resp.usage = MagicMock(prompt_tokens=100, total_tokens=100)

    fake_client = MagicMock()
    fake_client.embeddings.create.return_value = fake_resp

    with patch("ai_research_agent.prefilter._client", return_value=fake_client):
        result = score_by_embedding(papers, interests, top_n=2, budget=None)

    assert len(result) == 2
    assert result[0].paper.arxiv_id == "1"  # best match
    assert result[1].paper.arxiv_id == "3"  # second
```

- [ ] **Step 2: Run tests, expect ImportError**

```bash
uv run pytest tests/test_prefilter.py
```

- [ ] **Step 3: Implement `prefilter.py`**

`src/ai_research_agent/prefilter.py`:

```python
import math
from functools import lru_cache
from typing import Any

from openai import OpenAI

from ai_research_agent.budget import Budget
from ai_research_agent.models import Paper, ScoredPaper

EMBED_MODEL = "text-embedding-3-small"


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _interests_to_embedding_text(interests: dict[str, Any]) -> str:
    chunks = []
    for entry in interests.get("interests", []):
        chunks.append(entry.get("topic", ""))
        chunks.append(entry.get("description", ""))
        chunks.extend(entry.get("examples", []) or [])
    return "\n".join(c for c in chunks if c)


def score_by_embedding(
    papers: list[Paper],
    interests: dict[str, Any],
    top_n: int,
    budget: Budget | None,
) -> list[ScoredPaper]:
    """Embed interests + abstracts, return top-N papers by cosine similarity."""
    if not papers:
        return []

    interest_text = _interests_to_embedding_text(interests)
    inputs = [interest_text] + [p.abstract for p in papers]

    resp = _client().embeddings.create(model=EMBED_MODEL, input=inputs)
    if budget is not None:
        budget.charge(
            "prefilter",
            EMBED_MODEL,
            in_tok=resp.usage.prompt_tokens,
            out_tok=0,
        )

    interest_vec = resp.data[0].embedding
    scored: list[ScoredPaper] = []
    for paper, item in zip(papers, resp.data[1:]):
        s = _cosine(interest_vec, item.embedding)
        scored.append(ScoredPaper(paper=paper, embedding_score=s))

    scored.sort(key=lambda sp: sp.embedding_score, reverse=True)
    return scored[:top_n]
```

- [ ] **Step 4: Run tests, expect pass**

```bash
uv run pytest tests/test_prefilter.py
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ai_research_agent/prefilter.py tests/test_prefilter.py
git commit -m "feat(prefilter): cosine-similarity top-N filter via embeddings"
```

---

## Task 6: LLM ranker (`ranker.py`)

**Goal:** Score 30 candidates with Claude Haiku 4.5, return top-N as `RankedCandidate`s.

**Files:**
- Create: `src/ai_research_agent/ranker.py`
- Create: `tests/fixtures/haiku_rank_response.json`
- Create: `tests/test_ranker.py`

- [ ] **Step 1: Add a fixture LLM response**

`tests/fixtures/haiku_rank_response.json`:

```json
[
  {"arxiv_id": "2404.12345", "score": 9, "reasoning": "Strong agents paper, repo released."},
  {"arxiv_id": "2404.12500", "score": 7, "reasoning": "RAG fit; HF model card present."},
  {"arxiv_id": "2404.12399", "score": 3, "reasoning": "Survey, no implementation."}
]
```

- [ ] **Step 2: Write failing tests**

`tests/test_ranker.py`:

```python
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from ai_research_agent.budget import Budget
from ai_research_agent.models import Paper, ScoredPaper
from ai_research_agent.ranker import rank_candidates, _parse_ranking_json


FIXTURE = Path(__file__).parent / "fixtures" / "haiku_rank_response.json"


def make_scored(arxiv_id: str, abstract: str = "abs", score: float = 0.5) -> ScoredPaper:
    return ScoredPaper(
        paper=Paper(
            arxiv_id=arxiv_id, version="v1", title=arxiv_id, authors=[], abstract=abstract,
            published=datetime(2026, 4, 20, tzinfo=timezone.utc),
            updated=datetime(2026, 4, 20, tzinfo=timezone.utc),
            categories=[], pdf_url="", arxiv_url="",
        ),
        embedding_score=score,
    )


def test_parse_ranking_json_strict():
    raw = FIXTURE.read_text()
    parsed = _parse_ranking_json(raw)
    assert len(parsed) == 3
    assert parsed[0]["arxiv_id"] == "2404.12345"
    assert parsed[0]["score"] == 9


def test_parse_ranking_json_handles_markdown_wrapper():
    raw = "Here you go:\n```json\n" + FIXTURE.read_text() + "\n```\nThanks!"
    parsed = _parse_ranking_json(raw)
    assert len(parsed) == 3


def test_parse_ranking_json_drops_malformed_entries():
    raw = '[{"arxiv_id": "x", "score": 8, "reasoning": "ok"}, {"foo": "bar"}]'
    parsed = _parse_ranking_json(raw)
    assert len(parsed) == 1
    assert parsed[0]["arxiv_id"] == "x"


def test_rank_candidates_returns_top_n():
    scored = [
        make_scored("2404.12345", abstract="agent code at github.com/a/b"),
        make_scored("2404.12399", abstract="theory paper"),
        make_scored("2404.12500", abstract="RAG hf.co/foo/bar"),
    ]
    interests = {"interests": [], "mvp_constraints": {"hard_drops": [], "preferred_signals": []}}

    fake_msg = MagicMock()
    fake_msg.content = [MagicMock(text=FIXTURE.read_text())]
    fake_msg.usage = MagicMock(input_tokens=500, output_tokens=200)

    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_msg

    budget = Budget(cap_usd=10.0)
    with patch("ai_research_agent.ranker._client", return_value=fake_client):
        result = rank_candidates(scored, interests, top_n=2, budget=budget)

    assert len(result) == 2
    # Should come back sorted by llm_score desc
    assert result[0].llm_score == 9
    assert result[1].llm_score == 7
    assert result[0].has_repo_url_in_abstract  # github.com in abstract
    assert budget.spent > 0


def test_rank_candidates_empty_input_returns_empty():
    interests = {"interests": [], "mvp_constraints": {"hard_drops": [], "preferred_signals": []}}
    out = rank_candidates([], interests, top_n=5, budget=Budget(cap_usd=1.0))
    assert out == []
```

- [ ] **Step 3: Run tests, expect ImportError**

```bash
uv run pytest tests/test_ranker.py
```

- [ ] **Step 4: Implement `ranker.py`**

`src/ai_research_agent/ranker.py`:

```python
import json
import logging
import re
from functools import lru_cache
from typing import Any

from anthropic import Anthropic

from ai_research_agent.budget import Budget
from ai_research_agent.models import RankedCandidate, ScoredPaper

logger = logging.getLogger(__name__)

RANKER_MODEL = "claude-haiku-4-5"
REPO_URL_RE = re.compile(r"(github\.com/[\w\-]+/[\w\-\.]+|huggingface\.co/[\w\-]+/[\w\-\.]+)", re.I)
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.S)


@lru_cache(maxsize=1)
def _client() -> Anthropic:
    return Anthropic()


def _parse_ranking_json(text: str) -> list[dict[str, Any]]:
    """Parse the ranker response. Tolerates markdown code-block wrappers."""
    m = JSON_BLOCK_RE.search(text)
    payload = m.group(1) if m else text.strip()
    # Strip leading prose if any
    start = payload.find("[")
    end = payload.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        items = json.loads(payload[start:end + 1])
    except json.JSONDecodeError:
        logger.warning("Ranker returned malformed JSON: %s", text[:200])
        return []
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if "arxiv_id" not in item or "score" not in item:
            continue
        out.append(item)
    return out


def _build_prompt(scored: list[ScoredPaper], interests: dict[str, Any]) -> str:
    interest_yaml = json.dumps(interests, indent=2)
    candidates = []
    for sp in scored:
        candidates.append(
            f"---\narxiv_id: {sp.paper.arxiv_id}\n"
            f"title: {sp.paper.title}\n"
            f"abstract: {sp.paper.abstract}\n"
        )
    return (
        "You are ranking arXiv papers for a single developer who wants to BUILD things from "
        "papers — not survey the field. They will read the top 3 you select.\n\n"
        "## Their interests and constraints\n"
        f"{interest_yaml}\n\n"
        "## Candidate papers\n"
        f"{chr(10).join(candidates)}\n\n"
        "## Task\n"
        "Score each candidate 1-10 on (a) fit with the user's stated interests, "
        "(b) MVP-implementability for one developer, (c) whether the abstract suggests "
        "open-source code/repos exist. A paper that hits a hard_drop is automatically <=3.\n\n"
        "Reply with ONLY a JSON array, no commentary. Each item must have keys: "
        "`arxiv_id` (string), `score` (int 1-10), `reasoning` (one short sentence)."
    )


def rank_candidates(
    scored: list[ScoredPaper],
    interests: dict[str, Any],
    top_n: int,
    budget: Budget,
) -> list[RankedCandidate]:
    if not scored:
        return []

    prompt = _build_prompt(scored, interests)
    resp = _client().messages.create(
        model=RANKER_MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text
    budget.charge(
        "rank",
        RANKER_MODEL,
        in_tok=resp.usage.input_tokens,
        out_tok=resp.usage.output_tokens,
    )

    parsed = _parse_ranking_json(text)
    by_id = {item["arxiv_id"]: item for item in parsed}

    ranked: list[RankedCandidate] = []
    for sp in scored:
        item = by_id.get(sp.paper.arxiv_id)
        if item is None:
            continue
        ranked.append(RankedCandidate(
            paper=sp.paper,
            embedding_score=sp.embedding_score,
            llm_score=int(item["score"]),
            llm_reasoning=str(item.get("reasoning", "")),
            has_repo_url_in_abstract=bool(REPO_URL_RE.search(sp.paper.abstract)),
        ))

    ranked.sort(key=lambda r: r.llm_score, reverse=True)
    return ranked[:top_n]
```

- [ ] **Step 5: Run tests, expect pass**

```bash
uv run pytest tests/test_ranker.py
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ai_research_agent/ranker.py tests/test_ranker.py tests/fixtures/haiku_rank_response.json
git commit -m "feat(ranker): Haiku-based MVP-ability scoring of candidates"
```

---

## Task 7: PDF parser (`pdf_parser.py`)

**Goal:** Extract text from a PDF byte stream, capped at N tokens (using `tiktoken` for counting).

**Files:**
- Create: `src/ai_research_agent/pdf_parser.py`
- Create: `tests/fixtures/sample_paper.pdf` (any small real arXiv PDF — see Step 1)
- Create: `tests/test_pdf_parser.py`

- [ ] **Step 1: Drop in a real fixture PDF**

Download a small real arXiv PDF (≤5 pages) for the fixture:

```bash
curl -L -o tests/fixtures/sample_paper.pdf https://arxiv.org/pdf/2310.06825v1
# (Mistral 7B paper — small, well-formed)
```

If that URL changes, any small (<10 pages) arXiv PDF works.

- [ ] **Step 2: Write failing tests**

`tests/test_pdf_parser.py`:

```python
from pathlib import Path

from ai_research_agent.pdf_parser import parse, _count_tokens, _truncate_to_tokens


FIXTURE = Path(__file__).parent / "fixtures" / "sample_paper.pdf"


def test_count_tokens_basic():
    assert _count_tokens("hello world") > 0
    assert _count_tokens("") == 0


def test_truncate_to_tokens_returns_within_budget():
    text = "word " * 1000
    out = _truncate_to_tokens(text, max_tokens=50)
    assert _count_tokens(out) <= 50


def test_parse_returns_text_under_cap():
    pdf_bytes = FIXTURE.read_bytes()
    text = parse(pdf_bytes, max_tokens=2000)
    assert len(text) > 100
    assert _count_tokens(text) <= 2000


def test_parse_high_cap_returns_more_text():
    pdf_bytes = FIXTURE.read_bytes()
    short = parse(pdf_bytes, max_tokens=200)
    long = parse(pdf_bytes, max_tokens=10000)
    assert len(long) > len(short)


def test_parse_handles_empty_bytes():
    text = parse(b"", max_tokens=100)
    assert text == ""
```

- [ ] **Step 3: Run tests, expect ImportError**

```bash
uv run pytest tests/test_pdf_parser.py
```

- [ ] **Step 4: Implement `pdf_parser.py`**

`src/ai_research_agent/pdf_parser.py`:

```python
import io
import logging

import pdfplumber
import tiktoken

logger = logging.getLogger(__name__)

# cl100k_base is the OpenAI tokenizer; close enough to Anthropic's for budgeting purposes
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_ENCODING.encode(text))


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if not text:
        return ""
    tokens = _ENCODING.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _ENCODING.decode(tokens[:max_tokens])


def parse(pdf_bytes: bytes, max_tokens: int) -> str:
    """Extract text from a PDF byte stream, truncated to max_tokens."""
    if not pdf_bytes:
        return ""
    pages: list[str] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
    except Exception as e:
        logger.warning("PDF parse failed: %s", e)
        return ""
    full = "\n\n".join(pages)
    return _truncate_to_tokens(full, max_tokens)
```

- [ ] **Step 5: Run tests, expect pass**

```bash
uv run pytest tests/test_pdf_parser.py
```

Expected: 5 passed. (If the sample PDF is unusual, may need to adjust the `> 100` length floor.)

- [ ] **Step 6: Commit**

```bash
git add src/ai_research_agent/pdf_parser.py tests/test_pdf_parser.py tests/fixtures/sample_paper.pdf
git commit -m "feat(pdf): pdfplumber wrapper with token-budget truncation"
```

---

## Task 8: Repo resolver (`repo_resolver.py`)

**Goal:** Find a GitHub or HuggingFace URL for a paper (abstract → PDF page 1), fetch README + file tree.

**Files:**
- Create: `src/ai_research_agent/repo_resolver.py`
- Create: `tests/fixtures/sample_readme.md`
- Create: `tests/test_repo_resolver.py`

- [ ] **Step 1: Add a fixture README**

`tests/fixtures/sample_readme.md`:

```markdown
# Sample Project

A reference implementation of the method from the paper.

## Setup
```bash
pip install -r requirements.txt
```

## Training
Run `python train.py --config configs/base.yaml`.

## Evaluation
Run `python eval.py --checkpoint ckpt.pt`.
```

- [ ] **Step 2: Write failing tests**

`tests/test_repo_resolver.py`:

```python
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import respx

from ai_research_agent.budget import Budget
from ai_research_agent.models import Paper, RankedCandidate
from ai_research_agent.repo_resolver import (
    resolve, _extract_repo_url, _fetch_github_repo, _fetch_hf_repo,
)


README = (Path(__file__).parent / "fixtures" / "sample_readme.md").read_text()


def make_candidate(abstract: str = "", pdf_url: str = "") -> RankedCandidate:
    return RankedCandidate(
        paper=Paper(
            arxiv_id="2404.12345", version="v1", title="t", authors=[], abstract=abstract,
            published=datetime(2026, 4, 20, tzinfo=timezone.utc),
            updated=datetime(2026, 4, 20, tzinfo=timezone.utc),
            categories=[], pdf_url=pdf_url, arxiv_url="",
        ),
        embedding_score=0.9, llm_score=8, llm_reasoning="r",
        has_repo_url_in_abstract=bool(abstract),
    )


def test_extract_repo_url_from_github_link():
    out = _extract_repo_url("Code at https://github.com/foo/bar")
    assert out == ("https://github.com/foo/bar", "github")


def test_extract_repo_url_from_hf_link():
    out = _extract_repo_url("model on huggingface.co/foo/bar")
    assert out == ("https://huggingface.co/foo/bar", "huggingface")


def test_extract_repo_url_returns_none_when_absent():
    assert _extract_repo_url("no urls here") is None


@respx.mock
def test_fetch_github_repo_returns_bundle():
    respx.get("https://api.github.com/repos/foo/bar/readme").mock(
        return_value=httpx.Response(200, json={
            "content": __import__("base64").b64encode(README.encode()).decode(),
            "encoding": "base64",
        })
    )
    respx.get("https://api.github.com/repos/foo/bar/git/trees/HEAD").mock(
        return_value=httpx.Response(200, json={
            "tree": [{"path": "README.md", "type": "blob"},
                     {"path": "src/train.py", "type": "blob"}],
        })
    )
    bundle = _fetch_github_repo("https://github.com/foo/bar", readme_max_tokens=1000, tree_max_tokens=200)
    assert "Sample Project" in bundle.readme
    assert bundle.repo_kind == "github"
    assert "src/train.py" in bundle.file_tree


@respx.mock
def test_fetch_github_repo_handles_404():
    respx.get("https://api.github.com/repos/foo/missing/readme").mock(
        return_value=httpx.Response(404)
    )
    bundle = _fetch_github_repo("https://github.com/foo/missing", readme_max_tokens=1000, tree_max_tokens=200)
    assert bundle is None


@respx.mock
def test_fetch_hf_repo_returns_bundle():
    respx.get("https://huggingface.co/foo/bar/raw/main/README.md").mock(
        return_value=httpx.Response(200, text=README)
    )
    respx.get("https://huggingface.co/api/models/foo/bar/tree/main").mock(
        return_value=httpx.Response(200, json=[
            {"type": "file", "path": "README.md"},
            {"type": "file", "path": "config.json"},
        ])
    )
    bundle = _fetch_hf_repo("https://huggingface.co/foo/bar", readme_max_tokens=1000, tree_max_tokens=200)
    assert "Sample Project" in bundle.readme
    assert bundle.repo_kind == "huggingface"
    assert "config.json" in bundle.file_tree


@respx.mock
def test_resolve_uses_abstract_first():
    respx.get("https://api.github.com/repos/foo/bar/readme").mock(
        return_value=httpx.Response(200, json={
            "content": __import__("base64").b64encode(README.encode()).decode(),
            "encoding": "base64",
        })
    )
    respx.get("https://api.github.com/repos/foo/bar/git/trees/HEAD").mock(
        return_value=httpx.Response(200, json={"tree": [{"path": "README.md", "type": "blob"}]})
    )
    cand = make_candidate(abstract="Code at https://github.com/foo/bar")
    bundle = resolve(cand, budget=Budget(cap_usd=1.0))
    assert bundle is not None
    assert bundle.repo_url == "https://github.com/foo/bar"


def test_resolve_returns_none_when_no_url_anywhere():
    cand = make_candidate(abstract="no urls here", pdf_url="")
    # No PDF fetch will happen because pdf_url is empty
    bundle = resolve(cand, budget=Budget(cap_usd=1.0))
    assert bundle is None
```

- [ ] **Step 3: Run tests, expect ImportError**

```bash
uv run pytest tests/test_repo_resolver.py
```

- [ ] **Step 4: Implement `repo_resolver.py`**

`src/ai_research_agent/repo_resolver.py`:

```python
import base64
import logging
import re
from typing import Literal

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ai_research_agent.budget import Budget
from ai_research_agent.models import RankedCandidate, RepoBundle
from ai_research_agent.pdf_parser import _truncate_to_tokens, parse as parse_pdf

logger = logging.getLogger(__name__)

GITHUB_RE = re.compile(r"https?://(?:www\.)?github\.com/([\w\-]+)/([\w\-\.]+)", re.I)
HF_RE = re.compile(r"https?://(?:www\.)?huggingface\.co/([\w\-]+)/([\w\-\.]+)", re.I)
# For URLs without scheme (e.g., "github.com/foo/bar")
GITHUB_BARE_RE = re.compile(r"(?<!\.)github\.com/([\w\-]+)/([\w\-\.]+)", re.I)
HF_BARE_RE = re.compile(r"(?<!\.)huggingface\.co/([\w\-]+)/([\w\-\.]+)", re.I)


def _extract_repo_url(text: str) -> tuple[str, Literal["github", "huggingface"]] | None:
    if m := GITHUB_RE.search(text):
        return f"https://github.com/{m.group(1)}/{m.group(2).rstrip('.')}", "github"
    if m := HF_RE.search(text):
        return f"https://huggingface.co/{m.group(1)}/{m.group(2).rstrip('.')}", "huggingface"
    if m := GITHUB_BARE_RE.search(text):
        return f"https://github.com/{m.group(1)}/{m.group(2).rstrip('.')}", "github"
    if m := HF_BARE_RE.search(text):
        return f"https://huggingface.co/{m.group(1)}/{m.group(2).rstrip('.')}", "huggingface"
    return None


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    reraise=True,
)
def _http_get(url: str, headers: dict | None = None) -> httpx.Response:
    return httpx.get(url, headers=headers or {}, timeout=30.0, follow_redirects=True)


def _format_tree(paths: list[str], max_tokens: int) -> list[str]:
    """Truncate the path list to fit max_tokens."""
    out: list[str] = []
    joined_len = 0
    for p in paths:
        # rough: 1 token ≈ 4 chars
        joined_len += len(p) // 4 + 1
        if joined_len > max_tokens:
            break
        out.append(p)
    return out


def _fetch_github_repo(
    repo_url: str, readme_max_tokens: int, tree_max_tokens: int
) -> RepoBundle | None:
    m = re.match(r"https://github\.com/([\w\-]+)/([\w\-\.]+)", repo_url)
    if not m:
        return None
    owner, repo = m.group(1), m.group(2).rstrip(".")
    api_base = f"https://api.github.com/repos/{owner}/{repo}"

    readme_resp = _http_get(f"{api_base}/readme", headers={"Accept": "application/vnd.github+json"})
    if readme_resp.status_code == 404:
        return None
    readme_resp.raise_for_status()
    payload = readme_resp.json()
    readme = base64.b64decode(payload["content"]).decode("utf-8", errors="replace")
    readme_truncated = _truncate_to_tokens(readme, readme_max_tokens)

    tree_resp = _http_get(f"{api_base}/git/trees/HEAD", headers={"Accept": "application/vnd.github+json"})
    paths: list[str] = []
    if tree_resp.status_code == 200:
        for entry in tree_resp.json().get("tree", []):
            if entry.get("type") == "blob":
                paths.append(entry["path"])
    truncated = len(readme_truncated) < len(readme) or len(paths) > tree_max_tokens

    return RepoBundle(
        repo_url=repo_url,
        repo_kind="github",
        readme=readme_truncated,
        file_tree=_format_tree(paths, tree_max_tokens),
        truncated=truncated,
    )


def _fetch_hf_repo(
    repo_url: str, readme_max_tokens: int, tree_max_tokens: int
) -> RepoBundle | None:
    m = re.match(r"https://huggingface\.co/([\w\-]+)/([\w\-\.]+)", repo_url)
    if not m:
        return None
    owner, repo = m.group(1), m.group(2).rstrip(".")

    readme_resp = _http_get(f"https://huggingface.co/{owner}/{repo}/raw/main/README.md")
    if readme_resp.status_code == 404:
        return None
    readme_resp.raise_for_status()
    readme = readme_resp.text
    readme_truncated = _truncate_to_tokens(readme, readme_max_tokens)

    tree_resp = _http_get(f"https://huggingface.co/api/models/{owner}/{repo}/tree/main")
    paths: list[str] = []
    if tree_resp.status_code == 200:
        for entry in tree_resp.json():
            if entry.get("type") == "file":
                paths.append(entry["path"])
    truncated = len(readme_truncated) < len(readme) or len(paths) > tree_max_tokens

    return RepoBundle(
        repo_url=repo_url,
        repo_kind="huggingface",
        readme=readme_truncated,
        file_tree=_format_tree(paths, tree_max_tokens),
        truncated=truncated,
    )


def resolve(
    candidate: RankedCandidate,
    budget: Budget,
    readme_max_tokens: int = 8000,
    tree_max_tokens: int = 2000,
) -> RepoBundle | None:
    """Find the candidate's repo: try abstract first, then PDF page 1.
    Returns None if no repo URL can be found or the repo is unreachable."""
    # 1. Try abstract
    found = _extract_repo_url(candidate.paper.abstract)

    # 2. Fall back to PDF page 1
    if found is None and candidate.paper.pdf_url:
        try:
            pdf_resp = _http_get(candidate.paper.pdf_url)
            if pdf_resp.status_code == 200:
                first_page_text = parse_pdf(pdf_resp.content, max_tokens=2000)
                found = _extract_repo_url(first_page_text)
        except httpx.HTTPError as e:
            logger.warning("PDF fetch failed for repo discovery: %s", e)

    if found is None:
        return None

    repo_url, kind = found
    try:
        if kind == "github":
            return _fetch_github_repo(repo_url, readme_max_tokens, tree_max_tokens)
        else:
            return _fetch_hf_repo(repo_url, readme_max_tokens, tree_max_tokens)
    except httpx.HTTPError as e:
        logger.warning("Repo fetch failed for %s: %s", repo_url, e)
        return None
```

- [ ] **Step 5: Run tests, expect pass**

```bash
uv run pytest tests/test_repo_resolver.py
```

Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ai_research_agent/repo_resolver.py tests/test_repo_resolver.py tests/fixtures/sample_readme.md
git commit -m "feat(repo): resolve GitHub/HF URLs from abstract or PDF page 1"
```

---

## Task 9: Markdown template (`paper.md.j2`)

**Goal:** Static Jinja2 template the synthesizer fills in.

**Files:**
- Create: `src/ai_research_agent/templates/paper.md.j2`
- Create: `tests/test_template.py`

- [ ] **Step 1: Write the template**

`src/ai_research_agent/templates/paper.md.j2`:

```jinja
# {{ title }}

## Metadata
- arXiv ID: {{ arxiv_id }}
- Authors: {{ authors | join(", ") }}
- Published: {{ published }}
- arXiv link: {{ arxiv_url }}
- PDF link: {{ pdf_url }}
- Code/repo: {{ repo_url or "(none found)" }}

## Why this matters
{{ why_this_matters }}

## Technical idea
{{ technical_idea }}

## Implementation plan
{{ implementation_plan }}

## Dependencies
{{ dependencies }}

## Limitations / risks
{{ limitations_risks }}

## Next steps
{{ next_steps }}
```

- [ ] **Step 2: Write failing tests**

`tests/test_template.py`:

```python
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

TEMPLATE_DIR = Path(__file__).parents[1] / "src" / "ai_research_agent" / "templates"


def test_template_renders_with_required_fields():
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    tmpl = env.get_template("paper.md.j2")
    out = tmpl.render(
        title="Test Paper",
        arxiv_id="2404.12345",
        authors=["Alice", "Bob"],
        published="2026-04-20",
        arxiv_url="https://arxiv.org/abs/2404.12345",
        pdf_url="https://arxiv.org/pdf/2404.12345",
        repo_url="https://github.com/foo/bar",
        why_this_matters="Because.",
        technical_idea="An idea.",
        implementation_plan="Steps.",
        dependencies="deps",
        limitations_risks="risks",
        next_steps="next",
    )
    assert "# Test Paper" in out
    assert "Alice, Bob" in out
    assert "github.com/foo/bar" in out
    assert "## Implementation plan" in out


def test_template_handles_missing_repo_url():
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    tmpl = env.get_template("paper.md.j2")
    out = tmpl.render(
        title="t", arxiv_id="x", authors=[], published="", arxiv_url="", pdf_url="",
        repo_url=None,
        why_this_matters="", technical_idea="", implementation_plan="",
        dependencies="", limitations_risks="", next_steps="",
    )
    assert "(none found)" in out
```

- [ ] **Step 3: Run tests, expect pass** (template should exist)

```bash
uv run pytest tests/test_template.py
```

Expected: 2 passed.

- [ ] **Step 4: Commit**

```bash
git add src/ai_research_agent/templates/paper.md.j2 tests/test_template.py
git commit -m "feat(template): paper markdown Jinja2 template"
```

---

## Task 10: Synthesizer (`synthesizer.py`)

**Goal:** Given a paper + repo bundle + parsed PDF, ask Sonnet 4.6 to fill the template, return the full markdown.

**Files:**
- Create: `src/ai_research_agent/synthesizer.py`
- Create: `tests/fixtures/sonnet_synthesis_response.json`
- Create: `tests/test_synthesizer.py`

- [ ] **Step 1: Add a fixture LLM response**

`tests/fixtures/sonnet_synthesis_response.json`:

```json
{
  "why_this_matters": "Because LLMs need tools.",
  "technical_idea": "ReAct-style tool use with a planner.",
  "implementation_plan": "1. Pick a base model. 2. Wire tool dispatcher. 3. Eval on benchmark.",
  "dependencies": "Python, transformers, a small LLM.",
  "limitations_risks": "Tools may misfire on edge cases.",
  "next_steps": "Add multi-tool composition."
}
```

- [ ] **Step 2: Write failing tests**

`tests/test_synthesizer.py`:

```python
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from ai_research_agent.budget import Budget
from ai_research_agent.models import Paper, RepoBundle
from ai_research_agent.synthesizer import synthesize, _parse_synthesis_json


FIXTURE = Path(__file__).parent / "fixtures" / "sonnet_synthesis_response.json"


def make_paper() -> Paper:
    return Paper(
        arxiv_id="2404.12345", version="v1", title="Test Paper",
        authors=["Alice"], abstract="abs",
        published=datetime(2026, 4, 20, tzinfo=timezone.utc),
        updated=datetime(2026, 4, 20, tzinfo=timezone.utc),
        categories=["cs.AI"], pdf_url="http://arxiv.org/pdf/x",
        arxiv_url="http://arxiv.org/abs/x",
    )


def make_repo() -> RepoBundle:
    return RepoBundle(
        repo_url="https://github.com/foo/bar",
        repo_kind="github",
        readme="# foo\nrun python train.py",
        file_tree=["README.md", "train.py"],
        truncated=False,
    )


def test_parse_synthesis_json_strict():
    parsed = _parse_synthesis_json(FIXTURE.read_text())
    assert "why_this_matters" in parsed
    assert "implementation_plan" in parsed


def test_parse_synthesis_json_handles_markdown_wrapper():
    raw = "Here:\n```json\n" + FIXTURE.read_text() + "\n```"
    parsed = _parse_synthesis_json(raw)
    assert parsed["technical_idea"]


def test_synthesize_renders_markdown_with_filled_template():
    fake_msg = MagicMock()
    fake_msg.content = [MagicMock(text=FIXTURE.read_text())]
    fake_msg.usage = MagicMock(input_tokens=2000, output_tokens=500)

    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_msg

    budget = Budget(cap_usd=10.0)
    with patch("ai_research_agent.synthesizer._client", return_value=fake_client):
        md = synthesize(
            paper=make_paper(),
            repo=make_repo(),
            full_pdf_text="full paper body",
            budget=budget,
        )

    assert "# Test Paper" in md
    assert "## Implementation plan" in md
    assert "ReAct-style tool use" in md
    assert "github.com/foo/bar" in md
    assert budget.spent > 0


def test_synthesize_handles_missing_repo():
    fake_msg = MagicMock()
    fake_msg.content = [MagicMock(text=FIXTURE.read_text())]
    fake_msg.usage = MagicMock(input_tokens=1000, output_tokens=500)

    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_msg

    with patch("ai_research_agent.synthesizer._client", return_value=fake_client):
        md = synthesize(
            paper=make_paper(),
            repo=None,
            full_pdf_text="full body",
            budget=Budget(cap_usd=10.0),
        )
    assert "(none found)" in md
```

- [ ] **Step 3: Run tests, expect ImportError**

```bash
uv run pytest tests/test_synthesizer.py
```

- [ ] **Step 4: Implement `synthesizer.py`**

`src/ai_research_agent/synthesizer.py`:

```python
import json
import logging
import re
from functools import lru_cache
from pathlib import Path

from anthropic import Anthropic
from jinja2 import Environment, FileSystemLoader

from ai_research_agent.budget import Budget
from ai_research_agent.models import Paper, RepoBundle

logger = logging.getLogger(__name__)

SYNTH_MODEL = "claude-sonnet-4-6"
TEMPLATE_DIR = Path(__file__).parent / "templates"

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)

REQUIRED_FIELDS = [
    "why_this_matters", "technical_idea", "implementation_plan",
    "dependencies", "limitations_risks", "next_steps",
]


@lru_cache(maxsize=1)
def _client() -> Anthropic:
    return Anthropic()


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(loader=FileSystemLoader(TEMPLATE_DIR))


def _parse_synthesis_json(text: str) -> dict[str, str]:
    m = JSON_BLOCK_RE.search(text)
    payload = m.group(1) if m else text.strip()
    start = payload.find("{")
    end = payload.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in synthesis response: {text[:200]}")
    obj = json.loads(payload[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError(f"Synthesis response was not a JSON object: {obj!r}")
    for f in REQUIRED_FIELDS:
        obj.setdefault(f, "")
    return obj


def _build_prompt(paper: Paper, repo: RepoBundle | None, full_pdf_text: str) -> str:
    repo_section = "(no repo found)"
    if repo is not None:
        repo_section = (
            f"Repo URL: {repo.repo_url} ({repo.repo_kind})\n\n"
            f"## README\n{repo.readme}\n\n"
            f"## File tree\n" + "\n".join(repo.file_tree)
        )
    return (
        "You are turning an arXiv paper into a practical implementation-plan markdown for a "
        "single developer who wants to BUILD this. Output JSON only — no commentary, no "
        "code fences. The user will fill the JSON into a Jinja2 template.\n\n"
        f"## Paper metadata\n"
        f"Title: {paper.title}\n"
        f"Authors: {', '.join(paper.authors)}\n"
        f"arXiv ID: {paper.arxiv_id}\n\n"
        f"## Paper body\n{full_pdf_text}\n\n"
        f"## Repo context\n{repo_section}\n\n"
        "## Required JSON keys (all strings, multi-line allowed):\n"
        "- why_this_matters: motivation + concrete use cases\n"
        "- technical_idea: what the paper does, in a paragraph\n"
        "- implementation_plan: numbered step-by-step recipe to reproduce the core "
        "result, MVP-scoped to one developer\n"
        "- dependencies: stack, model sizes, compute, data\n"
        "- limitations_risks: what could break or won't work\n"
        "- next_steps: natural extensions after MVP\n\n"
        "Reply with ONLY the JSON object."
    )


def synthesize(
    paper: Paper,
    repo: RepoBundle | None,
    full_pdf_text: str,
    budget: Budget,
) -> str:
    """Generate the full markdown for a paper. Charges `budget`."""
    prompt = _build_prompt(paper, repo, full_pdf_text)
    resp = _client().messages.create(
        model=SYNTH_MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text
    budget.charge(
        "synthesize",
        SYNTH_MODEL,
        in_tok=resp.usage.input_tokens,
        out_tok=resp.usage.output_tokens,
    )

    fields = _parse_synthesis_json(text)
    tmpl = _env().get_template("paper.md.j2")
    return tmpl.render(
        title=paper.title,
        arxiv_id=paper.arxiv_id,
        authors=paper.authors,
        published=paper.published.date().isoformat(),
        arxiv_url=paper.arxiv_url,
        pdf_url=paper.pdf_url,
        repo_url=repo.repo_url if repo else None,
        **fields,
    )
```

- [ ] **Step 5: Run tests, expect pass**

```bash
uv run pytest tests/test_synthesizer.py
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ai_research_agent/synthesizer.py tests/test_synthesizer.py tests/fixtures/sonnet_synthesis_response.json
git commit -m "feat(synthesize): Sonnet-driven markdown generation from template"
```

---

## Task 11: Notifier (`notifier.py`)

**Goal:** Open a GitHub issue with stage + traceback when something fails.

**Files:**
- Create: `src/ai_research_agent/notifier.py`
- Create: `tests/test_notifier.py`

- [ ] **Step 1: Write failing tests**

`tests/test_notifier.py`:

```python
import subprocess
from unittest.mock import MagicMock, patch

from ai_research_agent.notifier import open_failure_issue


def test_open_failure_issue_calls_gh_cli():
    completed = MagicMock(returncode=0, stdout="https://github.com/foo/bar/issues/42\n")
    with patch("subprocess.run", return_value=completed) as run:
        url = open_failure_issue(
            stage="rank",
            exc=ValueError("boom"),
            run_url="https://example.com/run/1",
            repo="foo/bar",
        )
    assert url == "https://github.com/foo/bar/issues/42"
    args = run.call_args
    cmd = args.kwargs.get("args") or args.args[0]
    assert cmd[0] == "gh"
    assert "rank" in " ".join(cmd)


def test_open_failure_issue_swallows_subprocess_failure(caplog):
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "gh")):
        url = open_failure_issue(
            stage="rank", exc=ValueError("boom"),
            run_url="https://example.com/run/1", repo="foo/bar",
        )
    assert url is None
    assert any("notifier" in rec.name or "failed" in rec.message.lower() for rec in caplog.records)


def test_open_failure_issue_truncates_long_traceback():
    completed = MagicMock(returncode=0, stdout="https://example.com/issues/1\n")
    big_exc = RuntimeError("x" * 100_000)
    with patch("subprocess.run", return_value=completed) as run:
        open_failure_issue(stage="synth", exc=big_exc, run_url="u", repo="r/r")
    cmd = run.call_args.kwargs.get("args") or run.call_args.args[0]
    body_idx = cmd.index("--body")
    assert len(cmd[body_idx + 1]) < 70_000  # bounded
```

- [ ] **Step 2: Run tests, expect ImportError**

```bash
uv run pytest tests/test_notifier.py
```

- [ ] **Step 3: Implement `notifier.py`**

`src/ai_research_agent/notifier.py`:

```python
import logging
import subprocess
import traceback

logger = logging.getLogger(__name__)

MAX_BODY_CHARS = 60_000


def open_failure_issue(
    stage: str,
    exc: BaseException,
    run_url: str,
    repo: str,
) -> str | None:
    """Open a GitHub issue describing a stage failure.

    Returns the issue URL (stdout from `gh`) on success, None on failure.
    Never raises — callers in `main.py` are already in a failure path."""
    title = f"Weekly digest failure in stage `{stage}`"
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    body = (
        f"**Stage:** `{stage}`\n"
        f"**Run:** {run_url}\n"
        f"**Exception:** `{type(exc).__name__}: {exc}`\n\n"
        "## Traceback\n```\n"
        f"{tb}"
        "```\n"
    )
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + "\n\n_(truncated)_"

    cmd = ["gh", "issue", "create",
           "--repo", repo,
           "--title", title,
           "--body", body,
           "--label", "weekly-digest-failure"]
    try:
        result = subprocess.run(args=cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("notifier failed to create issue: %s", e)
        return None
```

- [ ] **Step 4: Run tests, expect pass**

```bash
uv run pytest tests/test_notifier.py
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ai_research_agent/notifier.py tests/test_notifier.py
git commit -m "feat(notifier): open GitHub issue on stage failure via gh CLI"
```

---

## Task 12: Main wiring + INDEX.md generator + CLI flags

**Goal:** Wire all stages, handle the failure-mode matrix, support `--dry-run`, `--validate-config`, regenerate `INDEX.md`.

**Files:**
- Create: `src/ai_research_agent/main.py`
- Create: `tests/test_main.py`

- [ ] **Step 1: Write failing tests**

`tests/test_main.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock, patch

from ai_research_agent.main import (
    _slugify, _existing_arxiv_ids, _write_paper_file, _regenerate_index, run,
)


def test_slugify_basic():
    assert _slugify("ReAct: Tool Use") == "react-tool-use"


def test_slugify_caps_at_60_chars():
    long = "x" * 200
    assert len(_slugify(long)) <= 60


def test_slugify_strips_punctuation():
    assert _slugify("Hello, World! v2") == "hello-world-v2"


def test_existing_arxiv_ids_scans_papers_dir(tmp_path):
    (tmp_path / "2026").mkdir()
    (tmp_path / "2026" / "2404.12345-foo.md").write_text("x")
    (tmp_path / "2026" / "2403.99999-bar.md").write_text("y")
    (tmp_path / "INDEX.md").write_text("z")  # should be ignored
    ids = _existing_arxiv_ids(tmp_path)
    assert ids == {"2404.12345", "2403.99999"}


def test_write_paper_file_creates_year_subdir(tmp_path):
    path = _write_paper_file(
        papers_dir=tmp_path,
        arxiv_id="2404.12345",
        title="ReAct: Tool Use",
        published_year=2026,
        markdown="# foo",
    )
    assert path.exists()
    assert path.parent.name == "2026"
    assert "2404.12345" in path.name
    assert path.read_text() == "# foo"


def test_regenerate_index_lists_papers_in_reverse_chrono(tmp_path):
    (tmp_path / "2026").mkdir()
    (tmp_path / "2025").mkdir()
    (tmp_path / "2026" / "2404.12345-foo.md").write_text(
        "# Foo Paper\n\n## Metadata\n- arXiv ID: 2404.12345\n"
    )
    (tmp_path / "2025" / "2312.99999-bar.md").write_text(
        "# Bar Paper\n\n## Metadata\n- arXiv ID: 2312.99999\n"
    )
    _regenerate_index(tmp_path)
    idx = (tmp_path / "INDEX.md").read_text()
    # 2026 (newer) appears before 2025 (older)
    assert idx.find("Foo Paper") < idx.find("Bar Paper")


def test_run_validate_config_returns_zero_on_clean_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GH_TOKEN", "token")
    monkeypatch.setenv("GH_REPO", "foo/bar")
    monkeypatch.setenv("BUDGET_USD_CAP", "3.00")
    interests = tmp_path / "interests.yaml"
    interests.write_text("interests: []\nmvp_constraints:\n  hard_drops: []\n  preferred_signals: []\n")
    papers = tmp_path / "papers"
    papers.mkdir()
    rc = run(["--validate-config", "--interests", str(interests), "--papers-dir", str(papers)])
    assert rc == 0
```

- [ ] **Step 2: Run tests, expect ImportError**

```bash
uv run pytest tests/test_main.py
```

- [ ] **Step 3: Implement `main.py`**

`src/ai_research_agent/main.py`:

```python
import argparse
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ai_research_agent.arxiv_client import fetch_recent
from ai_research_agent.budget import Budget, BudgetExceeded
from ai_research_agent.notifier import open_failure_issue
from ai_research_agent.pdf_parser import parse as parse_pdf
from ai_research_agent.prefilter import score_by_embedding
from ai_research_agent.ranker import rank_candidates
from ai_research_agent.repo_resolver import resolve as resolve_repo
from ai_research_agent.synthesizer import synthesize

logger = logging.getLogger(__name__)

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})")
SLUG_RE = re.compile(r"[^a-z0-9]+")
DEFAULT_CATEGORIES = ["cs.AI", "cs.CL", "cs.LG"]


def _slugify(s: str, max_len: int = 60) -> str:
    s = s.lower()
    s = SLUG_RE.sub("-", s).strip("-")
    return s[:max_len].rstrip("-")


def _existing_arxiv_ids(papers_dir: Path) -> set[str]:
    ids: set[str] = set()
    for md in papers_dir.glob("**/*.md"):
        if md.name == "INDEX.md":
            continue
        m = ARXIV_ID_RE.search(md.name)
        if m:
            ids.add(m.group(1))
    return ids


def _write_paper_file(
    papers_dir: Path,
    arxiv_id: str,
    title: str,
    published_year: int,
    markdown: str,
) -> Path:
    year_dir = papers_dir / str(published_year)
    year_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(title)
    path = year_dir / f"{arxiv_id}-{slug}.md"
    path.write_text(markdown)
    return path


def _regenerate_index(papers_dir: Path) -> None:
    rows = []
    for md in sorted(papers_dir.glob("**/*.md"), reverse=True):
        if md.name == "INDEX.md":
            continue
        title = md.stem
        first_line = md.read_text().splitlines()[0] if md.exists() else ""
        if first_line.startswith("# "):
            title = first_line[2:].strip()
        rel = md.relative_to(papers_dir).as_posix()
        rows.append(f"- [{title}]({rel})")
    body = "# Index\n\n" + "\n".join(rows) + "\n" if rows else "# Index\n\n_(empty)_\n"
    (papers_dir / "INDEX.md").write_text(body)


def _step_summary(line: str) -> None:
    """Append a line to GITHUB_STEP_SUMMARY if running in CI."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as f:
            f.write(line + "\n")


def _validate_config(interests_path: Path, papers_dir: Path) -> int:
    import importlib
    problems = []
    for var in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]:
        if not os.environ.get(var):
            problems.append(f"missing env var: {var}")
    if not interests_path.exists():
        problems.append(f"interests.yaml not found at {interests_path}")
    else:
        try:
            yaml.safe_load(interests_path.read_text())
        except yaml.YAMLError as e:
            problems.append(f"interests.yaml malformed: {e}")
    if not papers_dir.exists():
        problems.append(f"papers dir does not exist: {papers_dir}")
    elif not os.access(papers_dir, os.W_OK):
        problems.append(f"papers dir not writable: {papers_dir}")
    for mod in ["pdfplumber", "tiktoken", "jinja2", "yaml", "anthropic", "openai", "httpx"]:
        try:
            importlib.import_module(mod)
        except ImportError as e:
            problems.append(f"can't import {mod}: {e}")

    if problems:
        for p in problems:
            print(f"FAIL: {p}", file=sys.stderr)
        return 1
    print("validate-config: OK")
    return 0


def run(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--interests", default="interests.yaml")
    p.add_argument("--papers-dir", default="papers")
    p.add_argument("--dry-run", action="store_true",
                   help="Generate .md files but do not git-commit (no-op for git here; the workflow handles commit).")
    p.add_argument("--validate-config", action="store_true")
    args = p.parse_args(argv)

    interests_path = Path(args.interests)
    papers_dir = Path(args.papers_dir)

    if args.validate_config:
        return _validate_config(interests_path, papers_dir)

    interests = yaml.safe_load(interests_path.read_text())
    cap_usd = float(os.environ.get("BUDGET_USD_CAP", "3.00"))
    budget = Budget(cap_usd=cap_usd)
    repo = os.environ.get("GH_REPO", "")
    run_url = os.environ.get("GH_RUN_URL", "")

    counters = {"fetched": 0, "after_embed": 0, "after_rank": 0, "synthesized": 0}

    try:
        # Stage 1
        existing = _existing_arxiv_ids(papers_dir)
        papers = fetch_recent(DEFAULT_CATEGORIES, days=7, existing_ids=existing)
        counters["fetched"] = len(papers)
        logger.info("[stage 1/6] fetched %d papers (after dedup against %d existing)",
                    len(papers), len(existing))

        # Stage 2
        scored = score_by_embedding(papers, interests, top_n=30, budget=budget)
        counters["after_embed"] = len(scored)
        logger.info("[stage 2/6] embedding pre-filter -> top %d", len(scored))

        # Stage 3
        ranked = rank_candidates(scored, interests, top_n=5, budget=budget)
        counters["after_rank"] = len(ranked)
        logger.info("[stage 3/6] LLM ranker -> top %d", len(ranked))

        # Stages 4 + 5: resolve repos, synthesize markdown
        success_count = 0
        skipped: list[tuple[str, str]] = []  # (arxiv_id, reason)
        for cand in ranked:
            if success_count >= 3:
                break
            try:
                bundle = resolve_repo(cand, budget=budget)
                if bundle is None:
                    skipped.append((cand.paper.arxiv_id, "no repo found"))
                    continue
                # Fetch full PDF for synthesis
                import httpx
                pdf_resp = httpx.get(cand.paper.pdf_url, timeout=60.0, follow_redirects=True)
                pdf_resp.raise_for_status()
                full_text = parse_pdf(pdf_resp.content, max_tokens=30_000)

                md = synthesize(cand.paper, bundle, full_text, budget=budget)
                _write_paper_file(
                    papers_dir=papers_dir,
                    arxiv_id=cand.paper.arxiv_id,
                    title=cand.paper.title,
                    published_year=cand.paper.published.year,
                    markdown=md,
                )
                success_count += 1
                logger.info("[stage 4-5/6] wrote %s", cand.paper.arxiv_id)
            except Exception as e:
                logger.warning("paper %s failed: %s", cand.paper.arxiv_id, e)
                skipped.append((cand.paper.arxiv_id, str(e)))
                if repo and run_url:
                    open_failure_issue(stage=f"paper:{cand.paper.arxiv_id}",
                                       exc=e, run_url=run_url, repo=repo)

        counters["synthesized"] = success_count

        _regenerate_index(papers_dir)
        logger.info("[stage 6/6] regenerated INDEX.md")

        # Job summary
        _step_summary("## Weekly Digest — " + datetime.now(timezone.utc).date().isoformat())
        _step_summary("| Stage | Result |")
        _step_summary("|---|---|")
        _step_summary(f"| Fetched | {counters['fetched']} |")
        _step_summary(f"| After embeddings | {counters['after_embed']} |")
        _step_summary(f"| After LLM rank | {counters['after_rank']} |")
        _step_summary(f"| Synthesized & committed | {counters['synthesized']} |")
        _step_summary(f"| Total spend | ${budget.spent:.2f} |")
        if skipped:
            _step_summary("\n### Skipped\n")
            for aid, reason in skipped:
                _step_summary(f"- `{aid}`: {reason}")

        return 0 if success_count > 0 else 1

    except BudgetExceeded as e:
        logger.error("budget exceeded mid-run: %s", e)
        try:
            _regenerate_index(papers_dir)
        except Exception:
            pass
        if repo and run_url:
            open_failure_issue(stage="budget", exc=e, run_url=run_url, repo=repo)
        return 2
    except Exception as e:
        logger.exception("unexpected failure")
        if repo and run_url:
            open_failure_issue(stage="main", exc=e, run_url=run_url, repo=repo)
        return 3


if __name__ == "__main__":
    sys.exit(run())
```

- [ ] **Step 4: Run tests, expect pass**

```bash
uv run pytest tests/test_main.py
```

Expected: 6 passed.

- [ ] **Step 5: Run the full test suite**

```bash
uv run pytest
```

Expected: all tests across all modules pass.

- [ ] **Step 6: Commit**

```bash
git add src/ai_research_agent/main.py tests/test_main.py
git commit -m "feat(main): wire pipeline + CLI flags + INDEX.md regeneration"
```

---

## Task 13: Author the real `interests.yaml`

**Goal:** Drop in the actual user interests config from the spec.

**Files:**
- Create: `interests.yaml`

- [ ] **Step 1: Write `interests.yaml`** (full content from spec section "interests.yaml schema"; no test needed — `--validate-config` covers parse-ability)

```yaml
interests:
  - topic: LLM agents
    description: |
      Autonomous LLM-driven agents that plan, use tools, and execute
      multi-step tasks. ReAct-style reasoning, tool-calling, agentic workflows.
    examples:
      - tool-using agents
      - planning loops
      - multi-agent collaboration
    anti_examples:
      - reinforcement-learning agents in games
      - robotics control

  - topic: Model Context Protocol (MCP)
    description: |
      The Model Context Protocol and surrounding ecosystem — server
      implementations, client integrations, tool/resource exposure to LLMs,
      and patterns for connecting models to external systems. Includes
      papers on standardized agent-tool interfaces, structured tool calling,
      and protocol design for LLM integration.
    examples:
      - MCP server design
      - tool exposure patterns
      - structured tool-calling protocols
      - LLM integration standards
    anti_examples:
      - generic API design unrelated to LLMs
      - ad-hoc plugin systems without protocol contribution

  - topic: RAG and retrieval
    description: |
      Retrieval-augmented generation, hybrid search, query rewriting,
      and grounding LLM outputs in external knowledge.
    examples: [chunking strategies, reranking, hybrid search]
    anti_examples: [generic IR theory without LLM context]

  - topic: inference efficiency
    description: |
      Practical techniques to make LLM inference faster, cheaper,
      or more memory-efficient at deployment time.
    examples: [speculative decoding, KV cache, quantization for inference]
    anti_examples: [training-time-only optimizations]

  - topic: fine-tuning
    description: |
      Practical methods to adapt pretrained LLMs to specific tasks, domains,
      or behaviors. Parameter-efficient fine-tuning (LoRA/QLoRA/adapters),
      instruction tuning, preference optimization (DPO/ORPO/KTO), and
      domain adaptation. Emphasis on recipes a single developer can run.
    examples:
      - LoRA / QLoRA recipes
      - DPO and preference optimization
      - instruction tuning on small models
      - domain-specific adapters
    anti_examples:
      - trillion-parameter pretraining from scratch
      - papers requiring 100+ GPUs to reproduce
      - pretraining data curation only

mvp_constraints:
  hard_drops:
    - dataset_paper_only
    - pure_theory
    - frontier_model_only
  preferred_signals:
    - small_compute_footprint
    - small_model_size
    - reproducible_pipeline
```

- [ ] **Step 2: Run validate-config locally**

```bash
ANTHROPIC_API_KEY=sk-fake OPENAI_API_KEY=sk-fake uv run python -m ai_research_agent.main --validate-config
```

Expected: `validate-config: OK` and exit 0. (Real keys not yet needed; envar presence is enough at this stage.)

- [ ] **Step 3: Commit**

```bash
git add interests.yaml
git commit -m "feat(config): add interests.yaml with topics + MVP constraints"
```

---

## Task 14: GitHub Actions workflow

**Goal:** Drop in the CI workflow exactly as specified.

**Files:**
- Create: `.github/workflows/weekly.yml`

- [ ] **Step 1: Write the workflow file**

`.github/workflows/weekly.yml` (verbatim from spec section "GitHub Actions Workflow", but with package name `ai_research_agent`):

```yaml
name: Weekly arXiv Digest

on:
  schedule:
    - cron: "0 8 * * 0"        # Sunday 08:00 UTC (10:00 CEST / 09:00 CET)
  workflow_dispatch:

permissions:
  contents: write
  issues: write

concurrency:
  group: weekly-digest
  cancel-in-progress: false

jobs:
  digest:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4

      - name: Setup uv
        uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true

      - name: Install dependencies
        run: uv sync --frozen

      - name: Run weekly digest
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          OPENAI_API_KEY:    ${{ secrets.OPENAI_API_KEY }}
          GH_TOKEN:          ${{ secrets.GITHUB_TOKEN }}
          GH_REPO:           ${{ github.repository }}
          GH_RUN_URL:        ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}
          BUDGET_USD_CAP:    "3.00"
        run: uv run python -m ai_research_agent.main

      - name: Commit and push new papers
        if: always()
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          if [[ -n "$(git status --porcelain papers/)" ]]; then
            git add papers/
            git commit -m "weekly digest: $(date -u +%Y-%m-%d)"
            git push
          else
            echo "No new papers this week."
          fi

      - name: Last-resort failure issue
        if: failure()
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          gh issue create \
            --title "Weekly digest hard-crashed: $(date -u +%Y-%m-%d)" \
            --body  "Run did not complete cleanly. See $GH_RUN_URL"
```

- [ ] **Step 2: Lint the YAML**

```bash
uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/weekly.yml').read()); print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/weekly.yml
git commit -m "ci: weekly cron workflow for digest"
```

---

## Task 15: README + setup instructions

**Goal:** Brief operator README so future-you remembers how to set this up.

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# AI Research Agent

A weekly automated job that fetches AI/LLM papers from arXiv, ranks them for personal relevance and MVP-implementability, and commits LLM-generated implementation-plan markdown files to this repository.

See `docs/superpowers/specs/2026-04-26-ai-research-agent-design.md` for the full design.

## Setup

1. Push this repo to GitHub (private).
2. In **Settings → Secrets and variables → Actions**, add:
   - `ANTHROPIC_API_KEY`
   - `OPENAI_API_KEY`
3. Set monthly billing limits in vendor consoles:
   - Anthropic: $20/month
   - OpenAI: $5/month
4. Edit `interests.yaml` to taste.
5. Trigger a first manual run from the **Actions** tab → **Weekly arXiv Digest** → **Run workflow**.

## Local development

```bash
uv sync --all-extras

# Pre-flight: validate environment + config
ANTHROPIC_API_KEY=... OPENAI_API_KEY=... uv run python -m ai_research_agent.main --validate-config

# Full run, write papers/ but skip git operations (the workflow does git)
ANTHROPIC_API_KEY=... OPENAI_API_KEY=... uv run python -m ai_research_agent.main --dry-run

# Run unit tests
uv run pytest
```

## Pipeline

```
arXiv ── prefilter (embeddings) ── rank (Haiku) ── resolve repo ── synthesize (Sonnet) ── commit
```

Output lands in `papers/<year>/<arxiv-id>-<slug>.md` and `papers/INDEX.md`.

## Cost controls

- **Per-run cap:** $3.00 (env: `BUDGET_USD_CAP`) — enforced in `budget.py`
- **Account caps:** $20/month Anthropic, $5/month OpenAI — set in vendor consoles
- **Expected weekly spend:** ~$0.55 (worst case ~$1.50)
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with setup + local-dev instructions"
```

---

## Task 16: End-to-end smoke verification

**Goal:** Run the full test suite green, then do a manual `--validate-config` and a controlled real run.

- [ ] **Step 1: Full test suite**

```bash
uv run pytest -v
```

Expected: all tests pass (count should be ~36+ across all modules).

- [ ] **Step 2: Validate-config with real keys**

```bash
export ANTHROPIC_API_KEY=...   # real
export OPENAI_API_KEY=...      # real
uv run python -m ai_research_agent.main --validate-config
```

Expected: `validate-config: OK`.

- [ ] **Step 3: Run the cost-cap failure mode deliberately**

```bash
BUDGET_USD_CAP=0.001 uv run python -m ai_research_agent.main --dry-run || echo "exited non-zero as expected"
```

Expected:
- Process exits non-zero.
- An issue is *not* opened (no `GH_REPO`/`GH_RUN_URL` set in this local dev test).
- Logs say "budget exceeded mid-run".

- [ ] **Step 4: Push to GitHub and trigger a manual run**

```bash
# Assuming you've created the private repo on GitHub:
git remote add origin git@github.com:<you>/AI_Research_Agent.git
git push -u origin main
# Then: GitHub → Actions → Weekly arXiv Digest → "Run workflow"
```

Expected after the run completes:
- 3 new `.md` files in `papers/<year>/`.
- `papers/INDEX.md` updated.
- A commit by `github-actions[bot]`.
- Job summary on the run page shows the stage breakdown + total spend < $3.

- [ ] **Step 5: Verify acceptance criteria from spec**

From spec § Acceptance Criteria:
1. ✅ All unit tests pass — Step 1.
2. ✅ `--validate-config` clean against real secrets — Step 2.
3. ✅ Manual `workflow_dispatch` produces 3 sensible `.md` files committed — Step 4.
4. ✅ Budget-exceeded path verified — Step 3.

(Partial-synthesis-failure path will be exercised naturally on a future Sunday when something flakes; not worth a deliberate setup here.)

- [ ] **Step 6: Final commit (if you've made any tweaks during smoke testing)**

```bash
git add -A
git diff --cached --stat
git commit -m "fix: smoke-test adjustments"   # only if there are changes
```

---

## Self-review notes

The plan covers each spec section:

- **Spec § Pipeline architecture** → Tasks 4–10, 12 (one task per stage module)
- **Spec § Module layout** → Task 1 (scaffold) + Tasks 2–11
- **Spec § Data contracts → interests.yaml** → Tasks 5, 6 (consumes), 13 (authors the real file)
- **Spec § Data contracts → dataclasses** → Task 2
- **Spec § Data contracts → directory layout / naming / dedup** → Task 12 (`_slugify`, `_existing_arxiv_ids`, `_write_paper_file`)
- **Spec § Markdown template** → Task 9
- **Spec § GitHub Actions workflow** → Task 14
- **Spec § Cost control** → Task 3 (`Budget`) + Task 12 (`BudgetExceeded` handling) + Task 14 (env wiring)
- **Spec § Observability** → Task 12 (`_step_summary`) + Task 11 (`notifier`)
- **Spec § Failure-mode matrix** → Task 12 (per-paper try/except, BudgetExceeded handling)
- **Spec § Testing strategy** → Tests in every task (layer 1) + Task 12 CLI flags (layer 2) + Task 16 (layer 3 manual)
- **Spec § Acceptance criteria** → Task 16

No placeholders, no contradictions, no undefined references. Build-order dependencies are clean: each module imports only what was implemented in earlier tasks (`models` → `budget` → stages → `main`).
