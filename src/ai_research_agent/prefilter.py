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
