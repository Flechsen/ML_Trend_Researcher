from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    version: str
    title: str
    authors: list[str]
    abstract: str
    published: datetime
    updated: datetime
    categories: list[str]
    pdf_url: str
    arxiv_url: str


@dataclass(frozen=True)
class ScoredPaper:
    paper: Paper
    embedding_score: float


@dataclass(frozen=True)
class RankedCandidate:
    paper: Paper
    embedding_score: float
    llm_score: int  # 1-10
    llm_reasoning: str
    has_repo_url_in_abstract: bool


@dataclass(frozen=True)
class RepoBundle:
    repo_url: str
    repo_kind: Literal["github", "huggingface"]
    readme: str
    file_tree: list[str]
    truncated: bool
