from pathlib import Path
from jinja2 import Environment, FileSystemLoader

TEMPLATE_DIR = Path(__file__).parents[1] / "src" / "ai_research_agent" / "templates"


def test_template_renders_with_required_fields():
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    tmpl = env.get_template("paper.md.j2")
    out = tmpl.render(
        title="Test Paper",
        arxiv_id="2404.12345",
        authors=["Alice", "Bob"],
        published="2026-04-20",
        arxiv_url="https://arxiv.org/abs/2404.12345",
        pdf_url="https://arxiv.org/pdf/2404.12345",
        repo_url="https://github.com/foo/bar",
        why_this_matters="Because.",
        technical_idea="An idea.",
        implementation_plan="Steps.",
        dependencies="deps",
        limitations_risks="risks",
        next_steps="next",
    )
    assert "# Test Paper" in out
    assert "Alice, Bob" in out
    assert "github.com/foo/bar" in out
    assert "## Implementation plan" in out


def test_template_handles_missing_repo_url():
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    tmpl = env.get_template("paper.md.j2")
    out = tmpl.render(
        title="t", arxiv_id="x", authors=[], published="", arxiv_url="", pdf_url="",
        repo_url=None,
        why_this_matters="", technical_idea="", implementation_plan="",
        dependencies="", limitations_risks="", next_steps="",
    )
    assert "(none found)" in out
