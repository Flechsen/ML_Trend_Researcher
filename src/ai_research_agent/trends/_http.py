import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

DEFAULT_TIMEOUT = 30.0


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        # 403 included: GitHub signals secondary rate limits with 403.
        return code in (403, 429) or code >= 500
    return False


_RETRY = dict(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    reraise=True,
)


@retry(**_RETRY)
def get_json(url: str, params: dict, headers: dict | None = None):
    resp = httpx.get(url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


@retry(**_RETRY)
def get_text(url: str, params: dict, headers: dict | None = None) -> str:
    resp = httpx.get(url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.text
