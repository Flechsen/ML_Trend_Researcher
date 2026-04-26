import io
import logging

import pdfplumber
import tiktoken

logger = logging.getLogger(__name__)

# cl100k_base is the OpenAI tokenizer; close enough to Anthropic's for budgeting purposes
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_ENCODING.encode(text))


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if not text:
        return ""
    tokens = _ENCODING.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _ENCODING.decode(tokens[:max_tokens])


def parse(pdf_bytes: bytes, max_tokens: int) -> str:
    """Extract text from a PDF byte stream, truncated to max_tokens."""
    if not pdf_bytes:
        return ""
    pages: list[str] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
    except Exception as e:
        logger.warning("PDF parse failed: %s", e)
        return ""
    full = "\n\n".join(pages)
    return _truncate_to_tokens(full, max_tokens)
