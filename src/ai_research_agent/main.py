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
from ai_research_agent.trends.report import generate as generate_trends_report

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


def run(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--interests", default="interests.yaml")
    p.add_argument("--papers-dir", default="papers")
    p.add_argument("--dry-run", action="store_true",
                   help="Generate .md files but do not git-commit (no-op for git here; the workflow handles commit).")
    p.add_argument("--validate-config", action="store_true")
    p.add_argument("--trends-dir", default="trends")
    p.add_argument("--skip-trends", action="store_true",
                   help="Run the paper pipeline only.")
    p.add_argument("--trends-only", action="store_true",
                   help="Skip the paper pipeline; only generate the trends report.")
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

    if args.trends_only:
        note = _run_trends(interests, budget, Path(args.trends_dir), repo, run_url)
        _step_summary("## Weekly Trends — " + datetime.now(timezone.utc).date().isoformat())
        _step_summary(f"Trends: {note} — spend ${budget.spent:.2f}")
        return 0 if not note.startswith("FAILED") else 1

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

        trends_note = "skipped (--skip-trends)"
        if not args.skip_trends:
            trends_note = _run_trends(interests, budget, Path(args.trends_dir), repo, run_url)

        # Job summary
        _step_summary("## Weekly Digest — " + datetime.now(timezone.utc).date().isoformat())
        _step_summary("| Stage | Result |")
        _step_summary("|---|---|")
        _step_summary(f"| Fetched | {counters['fetched']} |")
        _step_summary(f"| After embeddings | {counters['after_embed']} |")
        _step_summary(f"| After LLM rank | {counters['after_rank']} |")
        _step_summary(f"| Synthesized & committed | {counters['synthesized']} |")
        _step_summary(f"| Total spend | ${budget.spent:.2f} |")
        _step_summary(f"| Trends report | {trends_note} |")
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
