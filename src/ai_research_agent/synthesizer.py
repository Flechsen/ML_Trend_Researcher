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
