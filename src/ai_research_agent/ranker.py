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
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text
    budget.charge(
        "rank",
        RANKER_MODEL,
        in_tok=resp.usage.input_tokens,
        out_tok=resp.usage.output_tokens,
    )
    if getattr(resp, "stop_reason", None) == "max_tokens":
        logger.warning("ranker hit max_tokens; JSON likely truncated, parsed=%d", 0)

    parsed = _parse_ranking_json(text)
    if not parsed:
        logger.warning("ranker returned 0 parseable items; raw response head: %s",
                       text[:500].replace("\n", " "))
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
