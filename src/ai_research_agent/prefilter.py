import math
from functools import lru_cache
from typing import Any

import tiktoken
from openai import OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ai_research_agent.budget import Budget
from ai_research_agent.models import Paper, ScoredPaper

EMBED_MODEL = "text-embedding-3-small"
# OpenAI free tier caps embeddings at 40k tokens-per-minute. Stay under that
# per request (so a single batch never trips the limit) and let tenacity
# absorb the cooldown between batches if we still hit the rolling window.
EMBED_MAX_TOKENS_PER_INPUT = 400
EMBED_MAX_TOKENS_PER_BATCH = 30_000

_ENCODING = tiktoken.get_encoding("cl100k_base")


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


def _truncate(text: str, max_tokens: int) -> tuple[str, int]:
    tokens = _ENCODING.encode(text)
    if len(tokens) <= max_tokens:
        return text, len(tokens)
    return _ENCODING.decode(tokens[:max_tokens]), max_tokens


def _build_token_batches(texts: list[str]) -> list[list[str]]:
    """Split inputs into batches that stay under EMBED_MAX_TOKENS_PER_BATCH."""
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for text in texts:
        truncated, n_tok = _truncate(text, EMBED_MAX_TOKENS_PER_INPUT)
        if current and current_tokens + n_tok > EMBED_MAX_TOKENS_PER_BATCH:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(truncated)
        current_tokens += n_tok
    if current:
        batches.append(current)
    return batches


@retry(
    retry=retry_if_exception_type(RateLimitError),
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=20, max=90),
    reraise=True,
)
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

    abstracts = [p.abstract for p in papers]
    paper_vecs: list[list[float]] = []
    for batch in _build_token_batches(abstracts):
        paper_vecs.extend(_embed_batch(batch, budget))

    scored = [
        ScoredPaper(paper=paper, embedding_score=_cosine(interest_vec, vec))
        for paper, vec in zip(papers, paper_vecs)
    ]
    scored.sort(key=lambda sp: sp.embedding_score, reverse=True)
    return scored[:top_n]
