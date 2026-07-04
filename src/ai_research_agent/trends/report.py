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
