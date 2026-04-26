import base64
import logging
import re
from typing import Literal

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ai_research_agent.budget import Budget
from ai_research_agent.models import RankedCandidate, RepoBundle
from ai_research_agent.pdf_parser import _truncate_to_tokens, parse as parse_pdf

logger = logging.getLogger(__name__)

GITHUB_RE = re.compile(r"https?://(?:www\.)?github\.com/([\w\-]+)/([\w\-\.]+)", re.I)
HF_RE = re.compile(r"https?://(?:www\.)?huggingface\.co/([\w\-]+)/([\w\-\.]+)", re.I)
# For URLs without scheme (e.g., "github.com/foo/bar")
GITHUB_BARE_RE = re.compile(r"(?<!\.)github\.com/([\w\-]+)/([\w\-\.]+)", re.I)
HF_BARE_RE = re.compile(r"(?<!\.)huggingface\.co/([\w\-]+)/([\w\-\.]+)", re.I)


def _extract_repo_url(text: str) -> tuple[str, Literal["github", "huggingface"]] | None:
    if m := GITHUB_RE.search(text):
        return f"https://github.com/{m.group(1)}/{m.group(2).rstrip('.')}", "github"
    if m := HF_RE.search(text):
        return f"https://huggingface.co/{m.group(1)}/{m.group(2).rstrip('.')}", "huggingface"
    if m := GITHUB_BARE_RE.search(text):
        return f"https://github.com/{m.group(1)}/{m.group(2).rstrip('.')}", "github"
    if m := HF_BARE_RE.search(text):
        return f"https://huggingface.co/{m.group(1)}/{m.group(2).rstrip('.')}", "huggingface"
    return None


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    reraise=True,
)
def _http_get(url: str, headers: dict | None = None) -> httpx.Response:
    return httpx.get(url, headers=headers or {}, timeout=30.0, follow_redirects=True)


def _format_tree(paths: list[str], max_tokens: int) -> list[str]:
    """Truncate the path list to fit max_tokens."""
    out: list[str] = []
    joined_len = 0
    for p in paths:
        # rough: 1 token ≈ 4 chars
        joined_len += len(p) // 4 + 1
        if joined_len > max_tokens:
            break
        out.append(p)
    return out


def _fetch_github_repo(
    repo_url: str, readme_max_tokens: int, tree_max_tokens: int
) -> RepoBundle | None:
    m = re.match(r"https://github\.com/([\w\-]+)/([\w\-\.]+)", repo_url)
    if not m:
        return None
    owner, repo = m.group(1), m.group(2).rstrip(".")
    api_base = f"https://api.github.com/repos/{owner}/{repo}"

    readme_resp = _http_get(f"{api_base}/readme", headers={"Accept": "application/vnd.github+json"})
    if readme_resp.status_code == 404:
        return None
    readme_resp.raise_for_status()
    payload = readme_resp.json()
    readme = base64.b64decode(payload["content"]).decode("utf-8", errors="replace")
    readme_truncated = _truncate_to_tokens(readme, readme_max_tokens)

    tree_resp = _http_get(f"{api_base}/git/trees/HEAD", headers={"Accept": "application/vnd.github+json"})
    paths: list[str] = []
    if tree_resp.status_code == 200:
        for entry in tree_resp.json().get("tree", []):
            if entry.get("type") == "blob":
                paths.append(entry["path"])
    truncated = len(readme_truncated) < len(readme) or len(paths) > tree_max_tokens

    return RepoBundle(
        repo_url=repo_url,
        repo_kind="github",
        readme=readme_truncated,
        file_tree=_format_tree(paths, tree_max_tokens),
        truncated=truncated,
    )


def _fetch_hf_repo(
    repo_url: str, readme_max_tokens: int, tree_max_tokens: int
) -> RepoBundle | None:
    m = re.match(r"https://huggingface\.co/([\w\-]+)/([\w\-\.]+)", repo_url)
    if not m:
        return None
    owner, repo = m.group(1), m.group(2).rstrip(".")

    readme_resp = _http_get(f"https://huggingface.co/{owner}/{repo}/raw/main/README.md")
    if readme_resp.status_code == 404:
        return None
    readme_resp.raise_for_status()
    readme = readme_resp.text
    readme_truncated = _truncate_to_tokens(readme, readme_max_tokens)

    tree_resp = _http_get(f"https://huggingface.co/api/models/{owner}/{repo}/tree/main")
    paths: list[str] = []
    if tree_resp.status_code == 200:
        for entry in tree_resp.json():
            if entry.get("type") == "file":
                paths.append(entry["path"])
    truncated = len(readme_truncated) < len(readme) or len(paths) > tree_max_tokens

    return RepoBundle(
        repo_url=repo_url,
        repo_kind="huggingface",
        readme=readme_truncated,
        file_tree=_format_tree(paths, tree_max_tokens),
        truncated=truncated,
    )


def resolve(
    candidate: RankedCandidate,
    budget: Budget,
    readme_max_tokens: int = 8000,
    tree_max_tokens: int = 2000,
) -> RepoBundle | None:
    """Find the candidate's repo: try abstract first, then PDF page 1.
    Returns None if no repo URL can be found or the repo is unreachable."""
    # 1. Try abstract
    found = _extract_repo_url(candidate.paper.abstract)

    # 2. Fall back to PDF page 1
    if found is None and candidate.paper.pdf_url:
        try:
            pdf_resp = _http_get(candidate.paper.pdf_url)
            if pdf_resp.status_code == 200:
                first_page_text = parse_pdf(pdf_resp.content, max_tokens=2000)
                found = _extract_repo_url(first_page_text)
        except httpx.HTTPError as e:
            logger.warning("PDF fetch failed for repo discovery: %s", e)

    if found is None:
        return None

    repo_url, kind = found
    try:
        if kind == "github":
            return _fetch_github_repo(repo_url, readme_max_tokens, tree_max_tokens)
        else:
            return _fetch_hf_repo(repo_url, readme_max_tokens, tree_max_tokens)
    except httpx.HTTPError as e:
        logger.warning("Repo fetch failed for %s: %s", repo_url, e)
        return None
