import logging
import subprocess
import traceback

logger = logging.getLogger(__name__)

MAX_BODY_CHARS = 60_000


def open_failure_issue(
    stage: str,
    exc: BaseException,
    run_url: str,
    repo: str,
) -> str | None:
    """Open a GitHub issue describing a stage failure.

    Returns the issue URL (stdout from `gh`) on success, None on failure.
    Never raises — callers in `main.py` are already in a failure path."""
    title = f"Weekly digest failure in stage `{stage}`"
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    body = (
        f"**Stage:** `{stage}`\n"
        f"**Run:** {run_url}\n"
        f"**Exception:** `{type(exc).__name__}: {exc}`\n\n"
        "## Traceback\n```\n"
        f"{tb}"
        "```\n"
    )
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + "\n\n_(truncated)_"

    cmd = ["gh", "issue", "create",
           "--repo", repo,
           "--title", title,
           "--body", body,
           "--label", "weekly-digest-failure"]
    try:
        result = subprocess.run(args=cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("notifier failed to create issue: %s", e)
        return None
