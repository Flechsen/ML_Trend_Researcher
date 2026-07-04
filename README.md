# AI Research Agent

A weekly automated job that (1) fetches AI/LLM papers from arXiv, ranks them for personal relevance and MVP-implementability, and commits LLM-generated implementation-plan markdown files to this repository, and (2) generates a weekly ML trends report from Hacker News, GitHub, Hugging Face, and Reddit.

See `docs/superpowers/specs/2026-04-26-ai-research-agent-design.md` (papers) and `docs/superpowers/specs/2026-07-04-trend-sources-design.md` (trends) for the full designs.

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

HN + GitHub + Hugging Face + Reddit ── synthesize trends (Sonnet) ── trends/<date>.md
(each trend source fail-soft; the report survives partial outages)
```

Papers land in `papers/<year>/<arxiv-id>-<slug>.md` and `papers/INDEX.md`. The weekly trends
report lands in `trends/<YYYY-MM-DD>.md` (themes: MCP ecosystem, agents, autoresearch, new
models, tooling — an indirect view of what's trending on X, sourced from free APIs).

```bash
# Trends report only (fast prompt iteration, ~$0.15/run)
ANTHROPIC_API_KEY=... uv run python -m ai_research_agent.main --trends-only

# Paper pipeline only
ANTHROPIC_API_KEY=... OPENAI_API_KEY=... uv run python -m ai_research_agent.main --skip-trends
```

Trend sources are configured in the `trends:` block of `interests.yaml` (queries, subreddits,
star/point floors). Reddit is fetched via RSS and may be blocked from CI IPs — the report
then runs on the remaining sources and says so in its footer.

## Cost controls

- **Per-run cap:** $3.00 (env: `BUDGET_USD_CAP`) — enforced in `budget.py`
- **Account caps:** $20/month Anthropic, $5/month OpenAI — set in vendor consoles
- **Expected weekly spend:** ~$0.70 (papers ~$0.55 + trends ~$0.15; worst case ~$1.70)
