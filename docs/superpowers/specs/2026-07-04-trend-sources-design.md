# Weekly ML Trends Report — Design Spec

**Date:** 2026-07-04
**Status:** Approved (user pre-approved direction via Q&A before going AFK; alternatives recorded below)
**Author:** Benedikt Flechsenhar (with Claude)

## Overview

An add-on component to the existing weekly digest pipeline. Alongside the arXiv
paper pipeline (which stays **unchanged** — volumes, budget, and behavior as-is),
the weekly run now also fetches trending AI/ML signals from four free sources
(Hacker News, GitHub, Hugging Face, Reddit), synthesizes them into a single
themed markdown report, and commits it to `trends/YYYY-MM-DD.md`.

This captures the discourse the user cares about on X.com (new ML trends, MCP
servers, agent skills, autoresearch) **indirectly**: X's ML conversation mirrors
into HN, r/LocalLLaMA, GitHub stars, and HF trending within hours, and those
have free, stable, ToS-clean APIs.

## Goals

- One `trends/YYYY-MM-DD.md` per weekly run: what's trending across the ML
  ecosystem this week, grouped by theme (MCP ecosystem, agents, autoresearch,
  new models, tooling, ...), every claim linked to its source item.
- More than double the raw data volume the weekly run processes (hundreds of
  new trend items/week on top of ~1000 arXiv abstracts).
- Zero new paid dependencies; zero new required secrets.
- Fail-soft everywhere: a dead source (or all four) must never break the paper
  digest; a trends failure opens a GitHub issue and the run carries on.

## Non-Goals

- Paid X/Twitter API access ($200/month) — rejected by user.
- Scraping X via unofficial routes (Nitter, syndication) — brittle, ToS-hostile.
- Increasing arXiv paper-pipeline volume — explicitly retracted by user.
- Cross-week trend deltas ("rising vs. last week") — YAGNI for v1; weekly
  snapshots are standalone.

## Alternatives considered

1. **Official X API** — rejected: $200/month basic tier for search.
2. **Feed trend items into the existing rank→synthesize paper pipeline** —
   rejected: the implementation-plan template fits papers, not tweets/repos/
   threads; would distort both flows.
3. **Chosen: parallel trends component** — independent fetchers → one cheap
   synthesis call → separate output directory. Clean isolation from the paper
   pipeline, shared `budget.py`/`models.py` only (matches existing architecture
   rule: stage modules never import each other).

## Sources (all verified live on 2026-07-04)

| Source | Endpoint | Auth | Verified behavior |
|---|---|---|---|
| Hacker News | `https://hn.algolia.com/api/v1/search?query=<q>&tags=story&numericFilters=created_at_i><ts>` | none | Works. `points` is NOT filterable server-side (400) → filter client-side. Dedup across queries by `objectID`. |
| GitHub | `https://api.github.com/search/repositories?q=<topic/keyword>+created:><date>+stars:><n>&sort=stars` | optional `GH_TOKEN` (already in CI) | Works keyless (10 req/min; 30 with token). Raw "new+popular" is unfiltered noise → per-topic/keyword queries required. |
| Hugging Face papers | `https://huggingface.co/api/daily_papers?limit=50` | none | Works. Community-upvoted papers with `publishedAt`; filter to last 7 days client-side. |
| Hugging Face models | `https://huggingface.co/api/models?sort=trendingScore&limit=30` | none | Works. `trendingScore` confirmed a valid sort. |
| Reddit | `https://www.reddit.com/r/<sub>/top.rss?t=week&limit=30` | none | JSON endpoints 403 (blocked); **RSS returns 200** at low request rates but 429s on rapid retries and may 403 from cloud IPs → strictly fail-soft, no retry storm (1 retry with generous backoff). RSS lacks scores → `score=None`. |

## Architecture

```
main.py (after paper stages, own try/except — fail-soft)
    │
    ▼
trends/report.py :: generate(interests, budget, out_dir)
    │
    ├── trends/hackernews.py       fetch(cfg) → list[TrendItem]   ┐
    ├── trends/github_trending.py  fetch(cfg) → list[TrendItem]   │ each fail-soft:
    ├── trends/huggingface.py      fetch(cfg) → list[TrendItem]   │ exception → log
    └── trends/reddit.py           fetch(cfg) → list[TrendItem]   ┘ warning + []
    │
    ▼
one Claude Sonnet call (stage="trends", charged via Budget)
    items (capped ~30/source, compact JSON) + interests summary
    → themed markdown body
    │
    ▼
render templates/trends.md.j2 (deterministic shell: title, date,
    LLM body, per-source fetch counts) → trends/YYYY-MM-DD.md
    → regenerate trends/INDEX.md
```

### Module layout

```
src/ai_research_agent/
├── models.py                    # + TrendItem dataclass (shared, like the rest)
├── trends/
│   ├── __init__.py
│   ├── hackernews.py            # fetch(cfg) -> list[TrendItem]
│   ├── github_trending.py       # fetch(cfg) -> list[TrendItem]
│   ├── huggingface.py           # fetch(cfg) -> list[TrendItem]  (papers + models)
│   ├── reddit.py                # fetch(cfg) -> list[TrendItem]  (RSS/Atom)
│   └── report.py                # orchestrator + synthesis + file writing
└── templates/
    └── trends.md.j2
```

Rule preserved from the original spec: fetcher modules never import each other;
they share only `models.py`. `report.py` is the only consumer of the fetchers.

### `TrendItem` (added to `models.py`)

```python
@dataclass(frozen=True)
class TrendItem:
    source: Literal["hackernews", "github", "hf_papers", "hf_models", "reddit"]
    title: str
    url: str                      # canonical item URL (repo page, HN item, ...)
    score: int | None             # points / stars / upvotes / trendingScore; None if unavailable (Reddit RSS)
    detail: str                   # short context: repo description, comment count, pipeline tag, subreddit...
    created_at: datetime | None
```

### Configuration (`interests.yaml`, new `trends:` block)

```yaml
trends:
  hn_queries: ["LLM", "MCP", "AI agent", "Claude", "GPT", "RAG",
               "fine-tuning", "open model", "autoresearch", "agent skills"]
  min_hn_points: 40
  github_topics: [mcp, llm, ai-agents, rag, llm-inference, fine-tuning]
  github_keywords: ["mcp server", "llm agent", "agent skills"]
  min_github_stars: 100
  github_days: 14                # repos *created* in this window (else old giants dominate)
  subreddits: [LocalLLaMA, MachineLearning]
  max_items_per_source: 30
```

All keys optional — code carries these exact values as defaults, so a missing
`trends:` block still works. Fetchers receive the merged config dict (`cfg`).

## Synthesis

One Sonnet call (same model id the synthesizer already uses, priced in
`budget.py`). Prompt contains:

- the user's interest topics (names + one-line descriptions from `interests.yaml`)
- all fetched items as a compact list per source: `title — score — detail — url`
  (descriptions truncated ~200 chars; ≤ `max_items_per_source` each)
- instructions to produce the report body (see structure below), grouping items
  into cross-source themes, citing every claim as a markdown link, calling out
  MCP/agents/autoresearch explicitly when present, and flagging weak signals.

Report structure (LLM produces body; Jinja shell adds header/footer):

```markdown
# ML Trends — Week of <YYYY-MM-DD>

## TL;DR                      (≤5 bullets)
## Themes                     (### per theme, cross-source, linked)
## Notable new tools & repos
## Notable models & papers
## Radar                      (weak signals worth watching)

---
_Sources this week: HN <n> stories · GitHub <n> repos · HF <n> papers,
<n> models · Reddit <n> posts. Failed sources listed if any._
```

Estimated cost: ~10–20K input tokens + ~2–3K output ≈ **$0.10–0.15/run** on
top of the existing ~$0.55. `BUDGET_USD_CAP` stays at **3.00** — no change.

## Wiring (`main.py`)

- New stage after INDEX regeneration, wrapped in its own `try/except`:
  trends failure → `logger.warning` + `open_failure_issue(stage="trends", ...)`,
  paper digest outcome unaffected. `BudgetExceeded` inside trends is caught the
  same way (papers are already written by then).
- Per-source failures are handled *inside* `report.py` (warning + empty list +
  named in the report footer); only zero-items-total or synthesis failure
  aborts the trends stage.
- CLI: `--skip-trends` (papers only) and `--trends-only` (skip stages 1–6; for
  local iteration on the trends prompt, ~$0.15/iteration).
- Exit code semantics unchanged for papers. `--trends-only` returns 0 iff the
  report was written.
- Step summary gains a `Trends` row (items fetched per source, report path or
  failure note).

## Workflow (`weekly.yml`)

- Workflow name → "Weekly AI Digest".
- Commit step: `git add papers/ trends/` and porcelain-check both paths.
- No new secrets. `GH_TOKEN` (already present) is passed through and used by
  the GitHub fetcher when set.

## Failure-Mode Matrix (additions)

| What fails | Behavior |
|---|---|
| One source API down / 403 / 429 | Warning log; source contributes 0 items; named under "failed sources" in report footer. 1 retry with backoff (Reddit: single generous backoff, no retry storm). |
| All sources return 0 items | Skip synthesis, no report file, open issue (stage="trends"). Papers unaffected. |
| Trends synthesis LLM call fails | Open issue (stage="trends"); papers unaffected. |
| Budget exhausted before trends stage | `BudgetExceeded` caught by trends wrapper; issue opened; papers (already written) commit normally. |
| Reddit blocked from CI IPs permanently | Report simply runs on 3 sources; footer says so. Optional future upgrade: Reddit OAuth app creds as secrets (documented in README, not built). |

## Testing

Match existing conventions (`pytest`, mocked HTTP, fixtures dir):

| Test file | Covers |
|---|---|
| `test_trend_sources.py` | Each fetcher: parses canned JSON/Atom fixtures; client-side filters (points/stars/date); dedup (HN objectID, GH full_name); HTTP error → `[]` not raise; respects `max_items_per_source`. |
| `test_trend_report.py` | Orchestrator: partial source failure still synthesizes; all-fail → no file + raises; synthesis mocked (anthropic); budget charged; file written to `trends/<date>.md`; INDEX regenerated; footer counts correct. |
| `test_main.py` (extended) | Trends failure doesn't affect paper exit code; `--skip-trends` / `--trends-only` flags. |

Live smoke check (manual, keyless): each fetcher has
`python -m ai_research_agent.trends.<module>` CLI printing fetched items.

## Acceptance Criteria

1. Full test suite passes.
2. Live smoke run of all four fetchers returns items (Reddit may be 0 from
   blocked IPs — acceptable if the code path is the fail-soft one).
3. `--trends-only` dry run produces a sensible `trends/YYYY-MM-DD.md` +
   `trends/INDEX.md` (requires ANTHROPIC_API_KEY; if unavailable locally,
   mocked-synthesis test + first CI run cover it).
4. Paper pipeline behavior byte-for-byte unchanged when `--skip-trends`.
