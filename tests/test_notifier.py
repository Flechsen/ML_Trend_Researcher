import subprocess
from unittest.mock import MagicMock, patch

from ai_research_agent.notifier import open_failure_issue


def test_open_failure_issue_calls_gh_cli():
    completed = MagicMock(returncode=0, stdout="https://github.com/foo/bar/issues/42\n")
    with patch("subprocess.run", return_value=completed) as run:
        url = open_failure_issue(
            stage="rank",
            exc=ValueError("boom"),
            run_url="https://example.com/run/1",
            repo="foo/bar",
        )
    assert url == "https://github.com/foo/bar/issues/42"
    args = run.call_args
    cmd = args.kwargs.get("args") or args.args[0]
    assert cmd[0] == "gh"
    assert "rank" in " ".join(cmd)


def test_open_failure_issue_swallows_subprocess_failure(caplog):
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "gh")):
        url = open_failure_issue(
            stage="rank", exc=ValueError("boom"),
            run_url="https://example.com/run/1", repo="foo/bar",
        )
    assert url is None
    assert any("notifier" in rec.name or "failed" in rec.message.lower() for rec in caplog.records)


def test_open_failure_issue_truncates_long_traceback():
    completed = MagicMock(returncode=0, stdout="https://example.com/issues/1\n")
    big_exc = RuntimeError("x" * 100_000)
    with patch("subprocess.run", return_value=completed) as run:
        open_failure_issue(stage="synth", exc=big_exc, run_url="u", repo="r/r")
    cmd = run.call_args.kwargs.get("args") or run.call_args.args[0]
    body_idx = cmd.index("--body")
    assert len(cmd[body_idx + 1]) < 70_000  # bounded
