import math
from functools import lru_cache
from typing import Any

from openai import OpenAI

from ai_research_agent.budget import Budget
from ai_research_agent.models import Paper, ScoredPaper

EMBED_MODEL = "text-embedding-3-small"
# OpenAI caps each embedding request at 300k tokens. Abstracts run ~250 tokens
# on average, so 500 per batch leaves comfortable headroom under the cap.
EMBED_BATCH_SIZE = 500


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


def _embed_batch(inputs: list[str], budget: Budget | None) -> list[list[float]]:
    resp = _client().embeddings.create(model=EMBED_MODEL, input=inputs)
    if budget is not None:
        budget.charge("prefilter", EMBED_MODEL, in_tok=resp.usage.prompt_tokens, out_tok=0)
    return [item.embedding for item in resp.data]


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
    interest_vec = _embed_batch([interest_text], budget)[0]

    paper_vecs: list[list[float]] = []
    for i in range(0, len(papers), EMBED_BATCH_SIZE):
        chunk = [p.abstract for p in papers[i:i + EMBED_BATCH_SIZE]]
        paper_vecs.extend(_embed_batch(chunk, budget))

    scored = [
        ScoredPaper(paper=paper, embedding_score=_cosine(interest_vec, vec))
        for paper, vec in zip(papers, paper_vecs)
    ]
    scored.sort(key=lambda sp: sp.embedding_score, reverse=True)
    return scored[:top_n]
