from pathlib import Path

from ai_research_agent.pdf_parser import parse, _count_tokens, _truncate_to_tokens


FIXTURE = Path(__file__).parent / "fixtures" / "sample_paper.pdf"


def test_count_tokens_basic():
    assert _count_tokens("hello world") > 0
    assert _count_tokens("") == 0


def test_truncate_to_tokens_returns_within_budget():
    text = "word " * 1000
    out = _truncate_to_tokens(text, max_tokens=50)
    assert _count_tokens(out) <= 50


def test_parse_returns_text_under_cap():
    pdf_bytes = FIXTURE.read_bytes()
    text = parse(pdf_bytes, max_tokens=2000)
    assert len(text) > 100
    assert _count_tokens(text) <= 2000


def test_parse_high_cap_returns_more_text():
    pdf_bytes = FIXTURE.read_bytes()
    short = parse(pdf_bytes, max_tokens=200)
    long = parse(pdf_bytes, max_tokens=10000)
    assert len(long) > len(short)


def test_parse_handles_empty_bytes():
    text = parse(b"", max_tokens=100)
    assert text == ""
