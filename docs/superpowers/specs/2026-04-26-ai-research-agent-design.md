# AI Research Agent — Design Spec

**Date:** 2026-04-26
**Status:** Approved (brainstorming complete, awaiting user spec review)
**Author:** Benedikt Flechsenhar (with Claude)

## Overview

A weekly automated job that fetches recent AI/LLM papers from arXiv, filters them for personal relevance and MVP-implementability, generates implementation-plan markdown files using an LLM, and commits them to a private GitHub repository.

## Goals

- Surface 3 high-signal AI/LLM papers per week, ranked for *practical implementability*, not just novelty
- Generate implementation-plan markdown files rich enough to start prototyping from
- Run unattended on a weekly cron with bounded cost
- Make failures visible (not silent) and easy to debug

## Non-Goals

- Comprehensive coverage of the AI literature
- Theoretical / scaling-law / pretraining papers
- Multi-user functionality
- A web UI or dashboard

## Stack

- **Language:** Python 3.12
- **Package manager:** `uv`
- **arXiv API:** for paper retrieval
- **OpenAI `text-embedding-3-small`:** semantic pre-filter
- **Anthropic Claude Haiku 4.5:** LLM ranker
- **Anthropic Claude Sonnet 4.6:** markdown synthesis
- **`pdfplumber`:** PDF parsing
- **GitHub Actions:** weekly cron + commit/push
- **Single private GitHub repo:** code + markdown output, one place

## Pipeline Architecture

```
GitHub Actions (cron: 0 8 * * 0, Sunday 08:00 UTC)
        │
        ▼
1. fetch_arxiv         → query cs.AI/CL/LG, last 7 days, dedup against existing papers/   → list[Paper] (~1000)
        │
        ▼
2. prefilter_embed     → cosine(embed(abstract), embed(interests.yaml))                   → top 30
        │
        ▼
3. rank_llm (Haiku)    → score on {interests fit, MVPability, code-availability hint}     → top 5
        │
        ▼
4. resolve_repo        → find GitHub/HF URL: abstract → PDF page 1. Drop if none found.   → top 3 survivors
        │
        ▼
5. synthesize_md       → full PDF (≤30K tok) + README + tree (≤10K tok) + metadata
                         → Sonnet 4.6 → fill template                                      → 3 .md files
        │
        ▼
6. commit_and_push     → write papers/<year>/<id>-<slug>.md, regenerate INDEX.md, push
        │
        ▼
7. on any failure      → notifier opens GitHub issue with stage + traceback
                         (best-effort: continue with next candidate where possible)
```

### Why this shape

- **Cheap stages first.** Embedding 1500 abstracts costs ~$0.01; ranking 30 candidates with Haiku costs ~$0.03; only the final 3 papers go through the expensive Sonnet synthesis (~$0.50). Inverting this order would cost 10–100× more.
- **Repo-availability gate (stage 4) sits *between* ranking and synthesis.** This means we never spend Sonnet tokens on a paper that has no usable code, but we don't have to fetch every candidate's repo upfront.
- **Top-5 buffer before resolve_repo.** Some top-3 candidates will lose their repo-resolution check (no GitHub link found in abstract or PDF page 1). The extra 2 candidates give the pipeline room to recover without a re-rank.

## Module Layout

```
ai_research_agent/
├── pyproject.toml
├── interests.yaml                   # the user's stable interest spec
├── papers/                          # output: committed .md files (the deliverable)
│   ├── INDEX.md
│   └── 2026/
│       └── ...
├── src/ai_research_agent/
│   ├── __init__.py
│   ├── main.py                      # entry point, wires stages, handles top-level errors
│   ├── models.py                    # Paper, ScoredPaper, RankedCandidate, RepoBundle dataclasses
│   ├── arxiv_client.py              # stage 1: arXiv API + dedup vs existing papers/
│   ├── prefilter.py                 # stage 2: OpenAI embeddings + cosine similarity
│   ├── ranker.py                    # stage 3: Haiku 4.5 batch scoring
│   ├── repo_resolver.py             # stage 4: find repo URL, fetch README + tree
│   ├── pdf_parser.py                # pdfplumber wrapper with token-cap
│   ├── synthesizer.py               # stage 5: Sonnet 4.6 markdown generation
│   ├── budget.py                    # token + cost tracking, hard cap, abort
│   ├── notifier.py                  # GitHub issue creation on failure
│   └── templates/
│       └── paper.md.j2              # markdown template (Jinja2)
├── tests/
│   ├── conftest.py                  # factories: make_paper(), make_candidate(), ...
│   ├── fixtures/                    # canned arXiv XML, sample PDFs, READMEs, LLM responses
│   └── test_<each_module>.py        # unit tests, mock external APIs
└── .github/
    └── workflows/
        └── weekly.yml               # the cron trigger
```

### Module responsibilities

Each stage module exposes a single public function plus a CLI entry point. Boundaries are dataclasses defined in `models.py`. No module imports another stage module — they only share `models.py`, `budget.py`, and external libs.

| Module | Public API | External deps |
|---|---|---|
| `arxiv_client.py` | `fetch_recent(categories, days, existing_ids) -> list[Paper]` | `httpx`, arXiv XML |
| `prefilter.py` | `score_by_embedding(papers, interests_yaml, top_n) -> list[ScoredPaper]` | `openai` |
| `ranker.py` | `rank_candidates(candidates, interests_yaml, top_n, budget) -> list[RankedCandidate]` | `anthropic` |
| `repo_resolver.py` | `resolve(candidate, budget) -> RepoBundle \| None` | `httpx`, `pdfplumber` |
| `pdf_parser.py` | `parse(pdf_bytes, max_tokens) -> str` | `pdfplumber`, `tiktoken` |
| `synthesizer.py` | `synthesize(paper, repo, full_pdf_text, template, budget) -> str` | `anthropic`, `jinja2` |
| `budget.py` | `Budget(cap_usd)` with `.charge(stage, model, in_tok, out_tok)` | none (pure) |
| `notifier.py` | `open_failure_issue(stage, exc, run_url)` | `subprocess`/`gh` CLI or `httpx` |

`main.py` also owns: regenerating `papers/INDEX.md` at the end of each run (after synthesis, before commit) by scanning `papers/**/*.md` and rebuilding the reverse-chronological table.

## Data Contracts

### `interests.yaml` schema

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
    - dataset_paper_only          # benchmark/dataset release without a method
    - pure_theory                 # convergence proofs, no algorithmic contribution
    - frontier_model_only         # "we trained a 70B model" with no reproducible recipe
  preferred_signals:
    - small_compute_footprint
    - small_model_size
    - reproducible_pipeline
```

The `description + examples` text is what gets concatenated and embedded for the cosine pre-filter. `anti_examples` give the LLM ranker explicit disambiguators. `mvp_constraints` are explicit so the ranker prompt can cite them directly.

### Dataclasses (`src/ai_research_agent/models.py`)

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

@dataclass(frozen=True)
class Paper:                       # raw arXiv metadata
    arxiv_id: str                  # "2404.12345" (base, no version)
    version: str                   # "v1", "v2", ...
    title: str
    authors: list[str]
    abstract: str
    published: datetime
    updated: datetime
    categories: list[str]
    pdf_url: str
    arxiv_url: str

@dataclass(frozen=True)
class ScoredPaper:                 # output of stage 2 (embeddings)
    paper: Paper
    embedding_score: float

@dataclass(frozen=True)
class RankedCandidate:             # output of stage 3 (Haiku ranker)
    paper: Paper
    embedding_score: float
    llm_score: int                 # 1-10
    llm_reasoning: str             # one-sentence justification
    has_repo_url_in_abstract: bool

@dataclass(frozen=True)
class RepoBundle:                  # output of stage 4 (resolver)
    repo_url: str
    repo_kind: Literal["github", "huggingface"]
    readme: str                    # truncated to budget
    file_tree: list[str]           # paths only, no contents
    truncated: bool                # was readme/tree clipped?
```

### Directory & file layout

```
papers/
├── INDEX.md                                      # auto-regenerated each run
├── 2026/
│   ├── 2404.12345-react-tool-use-for-agents.md
│   ├── 2404.12399-speculative-decoding-survey.md
│   └── 2404.12500-hybrid-rag-with-rerank.md
└── 2025/
    └── ...
```

**Naming rule:** `<arxiv-base-id>-<slug>.md` where slug = lowercased title, alphanumeric + `-` only, capped at 60 chars. No date prefix — the year directory gives ordering, and `git log <file>` provides exact dates.

**Dedup mechanism:** at start of each run, `glob('papers/**/*.md')` → extract arXiv IDs from filenames via regex → drop any candidate whose base ID is already in the set. The output directory *is* the state. No sidecar JSON, no SQLite.

**`INDEX.md`:** regenerated every run by scanning `papers/`. Reverse-chrono table with: date added (from `git log`), title, topics, repo link.

## Markdown Template

```markdown
# <Paper Title>

## Metadata
- arXiv ID:
- Authors:
- Published:
- arXiv link:
- PDF link:
- Code/repo:

## Why this matters
(motivation + concrete use cases)

## Technical idea
(what the paper does, in a paragraph)

## Implementation plan
(step-by-step recipe to reproduce the core result, MVP-scoped)

## Dependencies
(stack, model sizes, compute, data)

## Limitations / risks
(what could break or won't work)

## Next steps
(natural extensions, what to try after MVP)
```

## GitHub Actions Workflow

`.github/workflows/weekly.yml`:

```yaml
name: Weekly arXiv Digest

on:
  schedule:
    - cron: "0 8 * * 0"        # Sunday 08:00 UTC (10:00 CEST / 09:00 CET)
  workflow_dispatch:            # manual trigger for testing/dev

permissions:
  contents: write               # commit papers/ and INDEX.md
  issues: write                 # open failure-summary issues

concurrency:
  group: weekly-digest
  cancel-in-progress: false     # let in-flight runs finish

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

### Secrets

| Name | Used by | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | Haiku ranker + Sonnet synthesis | Set $20/month limit in Anthropic console |
| `OPENAI_API_KEY` | Embeddings only | Set $5/month limit in OpenAI console |
| `GITHUB_TOKEN` | issue creation, commit/push | Auto-provided by Actions; declared via `permissions:` block |

## Cost Control

Two layers:

1. **In-code per-run cap:** `BUDGET_USD_CAP=3.00` — enforced by `budget.py`. Every LLM/embedding call charges through `Budget.charge()`; exceeding the cap raises `BudgetExceeded`, which propagates to `main.py`, which commits whatever was completed and opens a failure issue.
2. **Account-level monthly caps:** $20/month Anthropic, $5/month OpenAI — set in vendor consoles. Defends against everything (compromised keys, runaway external triggers, etc.) that the in-code cap can't see.

### Expected cost

| Stage | Model | Per run |
|---|---|---|
| Embeddings | `text-embedding-3-small` | ~$0.01 |
| Ranking | Claude Haiku 4.5 | ~$0.03 |
| Synthesis | Claude Sonnet 4.6 | ~$0.50 |
| **Total** | | **~$0.55** (worst case ~$1.50) |

Per month (4–5 runs): expected **~$2.50**, worst case **~$7.50**. Well under the $20/month limit.

`budget.py` core:

```python
PRICING = {  # USD per token
    "claude-sonnet-4-6":      {"in": 3.00e-6,  "out": 15.00e-6},
    "claude-haiku-4-5":       {"in": 1.00e-6,  "out":  5.00e-6},
    "text-embedding-3-small": {"in": 0.02e-6,  "out":  0.0     },
}

class BudgetExceeded(RuntimeError): ...

class Budget:
    def __init__(self, cap_usd: float):
        self.cap_usd = cap_usd
        self.spent   = 0.0
        self.calls   = []                 # (stage, model, in_tok, out_tok, cost)

    def charge(self, stage: str, model: str, in_tok: int, out_tok: int):
        p = PRICING[model]
        cost = in_tok * p["in"] + out_tok * p["out"]
        self.spent += cost
        self.calls.append((stage, model, in_tok, out_tok, cost))
        if self.spent > self.cap_usd:
            raise BudgetExceeded(f"${self.spent:.2f} > ${self.cap_usd:.2f}")
```

## Observability

Three channels:

1. **GitHub Actions run logs** — every stage emits structured progress (`[stage 2/6] embedding 1247 abstracts → top 30`).
2. **Job summary** — written to `$GITHUB_STEP_SUMMARY` at end of `main.py`, e.g.:

   ```markdown
   ## Weekly Digest — 2026-04-26
   | Stage              | Result        |
   |--------------------|---------------|
   | Fetched            | 1247 papers   |
   | After embeddings   | 30            |
   | After LLM rank     | 5             |
   | After repo resolve | 3             |
   | Committed          | papers/2026/… |
   | Total spend        | $1.42         |
   ```
3. **GitHub issues** — `notifier.py` opens an issue on graceful failures (with stage + traceback + run URL); the workflow's `if: failure()` step opens a last-resort issue on hard crashes.

## Failure-Mode Matrix

| What fails | Behavior |
|---|---|
| arXiv API down | Retry 3× w/ backoff. If still down → exit, no commit, open issue. |
| Embedding API rate-limited | Retry 3× w/ backoff. If still failing → exit, no commit, open issue. (No keyword fallback — embedding API is reliable enough that adding a parallel code path costs more than it saves.) |
| LLM rank fails on 1 candidate | Skip that candidate, continue ranking the rest. |
| Repo resolution fails for a top-3 paper | Drop it, promote next-best from the top-5 candidate set. |
| PDF parse fails | Skip that paper, promote next-best. |
| Synthesis fails (LLM error) for 1 paper | Skip; commit the 2 that worked; open issue listing the skipped one. |
| Budget exceeded mid-run | Stop, commit what was written, open issue with cost breakdown. |
| Hard crash (import error, OOM, etc.) | Workflow's `if: failure()` step opens last-resort issue. No commit. |

## Testing Strategy

### Layer 1: Unit tests (every push, mocked, ~5s)

| Module | Mocked at | Key tests |
|---|---|---|
| `arxiv_client.py` | `httpx` response | parses XML; pagination; dedup against `papers/`; resilient to malformed entries |
| `prefilter.py` | `openai.embeddings.create` | cosine ranks correctly; batches inputs > 2048; returns top-N |
| `ranker.py` | `anthropic.messages.create` | builds prompt from `interests.yaml`; parses JSON; bad response → skip, don't crash |
| `repo_resolver.py` | `httpx` (GitHub + HF) | regex extracts repo URL; 404 → None; truncates README |
| `pdf_parser.py` | real fixture PDFs | extracts text from two-column layout; enforces token cap |
| `synthesizer.py` | `anthropic.messages.create` | renders Jinja2 template; charges budget |
| `budget.py` | nothing (pure) | cost math; `BudgetExceeded` fires above cap |
| `notifier.py` | `subprocess` / `httpx` | issue body format; network fail logged but not propagated |
| `main.py` | all of the above via fixtures | happy path; partial failure (paper 2 fails) → 2 files + 1 issue; budget exceeded → commits, opens issue, exit non-zero |

Stack: `pytest`, `pytest-httpx`, `pyfakefs` for the `papers/` directory in main-test. No `vcrpy`.

### Layer 2: Stage CLIs for local iteration

```bash
# Run the whole thing locally; write .md but don't commit
$ uv run python -m ai_research_agent.main --dry-run

# Pin to one specific paper, skip all earlier stages
$ uv run python -m ai_research_agent.synthesizer \
      --arxiv-id 2404.12345 \
      --output /tmp/test.md

# Test the ranker against a fixture set of candidates
$ uv run python -m ai_research_agent.ranker --from-fixtures

# Validate environment before the first real run
$ uv run python -m ai_research_agent.main --validate-config
# → checks: API keys valid, interests.yaml parses, pdfplumber imports,
#   GH_TOKEN can create issues, papers/ writable
```

`synthesizer --arxiv-id` is the workhorse for prompt iteration: pin one paper, skip embeddings + ranking, ~$0.17 per iteration.

### Layer 3: Production cron is the integration test

Sunday cron failures route to a GitHub issue. Iterate locally with stage CLIs.

## Acceptance Criteria

1. All unit tests pass.
2. `--validate-config` clean against real secrets.
3. One successful manual `workflow_dispatch` run produces 3 sensible `.md` files committed to `papers/`.
4. Failure-mode matrix verified: at least *budget exceeded* and *partial synthesis failure* paths confirmed by deliberate triggers (e.g., set `BUDGET_USD_CAP=0.01` for one run).

## Open Questions for Implementation

None at design time. All decisions are pinned in this spec.
