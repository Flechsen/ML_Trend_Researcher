from datetime import datetime, timezone
from pathlib import Path

import httpx
import respx

from ai_research_agent.budget import Budget
from ai_research_agent.models import Paper, RankedCandidate
from ai_research_agent.repo_resolver import (
    resolve, _extract_repo_url, _fetch_github_repo, _fetch_hf_repo,
)


README = (Path(__file__).parent / "fixtures" / "sample_readme.md").read_text()


def make_candidate(abstract: str = "", pdf_url: str = "") -> RankedCandidate:
    return RankedCandidate(
        paper=Paper(
            arxiv_id="2404.12345", version="v1", title="t", authors=[], abstract=abstract,
            published=datetime(2026, 4, 20, tzinfo=timezone.utc),
            updated=datetime(2026, 4, 20, tzinfo=timezone.utc),
            categories=[], pdf_url=pdf_url, arxiv_url="",
        ),
        embedding_score=0.9, llm_score=8, llm_reasoning="r",
        has_repo_url_in_abstract=bool(abstract),
    )


def test_extract_repo_url_from_github_link():
    out = _extract_repo_url("Code at https://github.com/foo/bar")
    assert out == ("https://github.com/foo/bar", "github")


def test_extract_repo_url_from_hf_link():
    out = _extract_repo_url("model on huggingface.co/foo/bar")
    assert out == ("https://huggingface.co/foo/bar", "huggingface")


def test_extract_repo_url_returns_none_when_absent():
    assert _extract_repo_url("no urls here") is None


@respx.mock
def test_fetch_github_repo_returns_bundle():
    respx.get("https://api.github.com/repos/foo/bar/readme").mock(
        return_value=httpx.Response(200, json={
            "content": __import__("base64").b64encode(README.encode()).decode(),
            "encoding": "base64",
        })
    )
    respx.get("https://api.github.com/repos/foo/bar/git/trees/HEAD").mock(
        return_value=httpx.Response(200, json={
            "tree": [{"path": "README.md", "type": "blob"},
                     {"path": "src/train.py", "type": "blob"}],
        })
    )
    bundle = _fetch_github_repo("https://github.com/foo/bar", readme_max_tokens=1000, tree_max_tokens=200)
    assert "Sample Project" in bundle.readme
    assert bundle.repo_kind == "github"
    assert "src/train.py" in bundle.file_tree


@respx.mock
def test_fetch_github_repo_handles_404():
    respx.get("https://api.github.com/repos/foo/missing/readme").mock(
        return_value=httpx.Response(404)
    )
    bundle = _fetch_github_repo("https://github.com/foo/missing", readme_max_tokens=1000, tree_max_tokens=200)
    assert bundle is None


@respx.mock
def test_fetch_hf_repo_returns_bundle():
    respx.get("https://huggingface.co/foo/bar/raw/main/README.md").mock(
        return_value=httpx.Response(200, text=README)
    )
    respx.get("https://huggingface.co/api/models/foo/bar/tree/main").mock(
        return_value=httpx.Response(200, json=[
            {"type": "file", "path": "README.md"},
            {"type": "file", "path": "config.json"},
        ])
    )
    bundle = _fetch_hf_repo("https://huggingface.co/foo/bar", readme_max_tokens=1000, tree_max_tokens=200)
    assert "Sample Project" in bundle.readme
    assert bundle.repo_kind == "huggingface"
    assert "config.json" in bundle.file_tree


@respx.mock
def test_resolve_uses_abstract_first():
    respx.get("https://api.github.com/repos/foo/bar/readme").mock(
        return_value=httpx.Response(200, json={
            "content": __import__("base64").b64encode(README.encode()).decode(),
            "encoding": "base64",
        })
    )
    respx.get("https://api.github.com/repos/foo/bar/git/trees/HEAD").mock(
        return_value=httpx.Response(200, json={"tree": [{"path": "README.md", "type": "blob"}]})
    )
    cand = make_candidate(abstract="Code at https://github.com/foo/bar")
    bundle = resolve(cand, budget=Budget(cap_usd=1.0))
    assert bundle is not None
    assert bundle.repo_url == "https://github.com/foo/bar"


def test_resolve_returns_none_when_no_url_anywhere():
    cand = make_candidate(abstract="no urls here", pdf_url="")
    # No PDF fetch will happen because pdf_url is empty
    bundle = resolve(cand, budget=Budget(cap_usd=1.0))
    assert bundle is None
