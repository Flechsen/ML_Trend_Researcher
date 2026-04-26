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
