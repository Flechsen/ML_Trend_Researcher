from datetime import timezone
from pathlib import Path
import respx
import httpx
from ai_research_agent.arxiv_client import fetch_recent, _parse_atom


FIXTURE = Path(__file__).parent / "fixtures" / "arxiv_response.xml"


def test_parse_atom_extracts_three_entries():
    papers = _parse_atom(FIXTURE.read_text())
    assert len(papers) == 3
    p = papers[0]
    assert p.arxiv_id == "2404.12345"
    assert p.version == "v2"
    assert p.title == "ReAct Tool Use for LLM Agents"
    assert p.authors == ["Alice Smith", "Bob Jones"]
    assert "tool-using agent" in p.abstract
    assert p.categories == ["cs.AI", "cs.CL"]
    assert p.pdf_url == "http://arxiv.org/pdf/2404.12345v2"
    assert p.published.tzinfo == timezone.utc


def test_parse_atom_strips_arxiv_version_from_id():
    papers = _parse_atom(FIXTURE.read_text())
    assert papers[0].arxiv_id == "2404.12345"
    assert papers[0].version == "v2"


@respx.mock
def test_fetch_recent_calls_arxiv_and_dedups(tmp_path):
    respx.get("https://export.arxiv.org/api/query").mock(
        return_value=httpx.Response(200, text=FIXTURE.read_text())
    )
    existing_ids = {"2404.12399"}
    papers = fetch_recent(
        categories=["cs.AI", "cs.CL", "cs.LG"],
        days=7,
        existing_ids=existing_ids,
    )
    ids = [p.arxiv_id for p in papers]
    assert "2404.12345" in ids
    assert "2404.12500" in ids
    assert "2404.12399" not in ids


@respx.mock
def test_fetch_recent_retries_on_5xx(tmp_path):
    route = respx.get("https://export.arxiv.org/api/query").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(503),
            httpx.Response(200, text=FIXTURE.read_text()),
        ]
    )
    papers = fetch_recent(categories=["cs.AI"], days=7, existing_ids=set())
    assert len(papers) == 3
    assert route.call_count == 3
