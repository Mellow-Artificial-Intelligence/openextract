"""Shared constants, environment knobs, and option validation."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence

_DEFAULT_URL_FETCH_TIMEOUT = 30.0
_DEFAULT_MAX_REDIRECTS = 10
_DEFAULT_RETRY_MAX_BACKOFF = 60.0
_DEFAULT_MAX_INPUT_BYTES = 50 * 1024 * 1024
_MAX_SWARM_SIZE = 16
_URL_TIMEOUT_ENV = "OPENEXTRACT_URL_TIMEOUT"
_MAX_REDIRECTS_ENV = "OPENEXTRACT_MAX_REDIRECTS"
_ALLOW_PRIVATE_URLS_ENV = "OPENEXTRACT_ALLOW_PRIVATE_URLS"
_MAX_INPUT_BYTES_ENV = "OPENEXTRACT_MAX_INPUT_BYTES"


def _env_positive[N: (int, float)](name: str, default: N, parse: Callable[[str], N]) -> N:
    """Parse a positive number from ``name``; return ``default`` when unset or invalid."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = parse(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _allow_private_urls() -> bool:
    """Return True when SSRF host validation is disabled via env var."""
    return os.environ.get(_ALLOW_PRIVATE_URLS_ENV, "").lower() in ("1", "true", "yes")


def _url_fetch_timeout() -> float:
    """HTTP timeout in seconds for URL fetches (``OPENEXTRACT_URL_TIMEOUT``)."""
    return _env_positive(_URL_TIMEOUT_ENV, _DEFAULT_URL_FETCH_TIMEOUT, float)


def _max_redirects() -> int:
    """Maximum redirect hops when fetching URLs (``OPENEXTRACT_MAX_REDIRECTS``)."""
    return _env_positive(_MAX_REDIRECTS_ENV, _DEFAULT_MAX_REDIRECTS, int)


def _positive_int(value: object) -> int | None:
    """Return ``value`` when it is a real ``int`` (not ``bool``) of at least 1."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _finite_number(value: object) -> float | None:
    """Return ``value`` as a float when it is a real finite number (not ``bool``)."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def _resolve_max_input_bytes(max_input_bytes: object) -> int:
    """Resolve and validate the per-input byte limit.

    An explicit value wins over ``OPENEXTRACT_MAX_INPUT_BYTES``. Invalid
    configured values fail closed instead of silently disabling the limit.
    """
    value = max_input_bytes
    from_environment = False
    if value is None:
        raw = os.environ.get(_MAX_INPUT_BYTES_ENV, "").strip()
        if not raw:
            return _DEFAULT_MAX_INPUT_BYTES
        from_environment = True
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{_MAX_INPUT_BYTES_ENV} must be a positive integer.") from exc
    limit = _positive_int(value)
    if limit is None:
        if from_environment:
            raise ValueError(f"{_MAX_INPUT_BYTES_ENV} must be a positive integer.")
        raise ValueError("max_input_bytes must be a positive integer.")
    return limit


def _validate_non_negative_seconds(value: object, *, name: str) -> None:
    """Reject anything that is not a finite number of seconds at or above zero."""
    seconds = _finite_number(value)
    if seconds is None or seconds < 0:
        raise ValueError(f"{name} must be a finite non-negative number of seconds.")


def _validate_retry_options(
    max_retries: object,
    retry_backoff: object,
    retry_max_backoff: object,
) -> None:
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer.")
    _validate_non_negative_seconds(retry_backoff, name="retry_backoff")
    _validate_non_negative_seconds(retry_max_backoff, name="retry_max_backoff")


def _validate_max_concurrency(max_concurrency: object) -> None:
    if _positive_int(max_concurrency) is None:
        raise ValueError("max_concurrency must be a positive integer.")


def _validate_swarm_size(size: object) -> int:
    """Validate a swarm agent count and return it.

    The upper bound keeps a typo (``size=1000``) from fanning out into a
    provider-rate-limit incident before any model call is made.
    """
    if isinstance(size, bool) or not isinstance(size, int) or size < 1 or size > _MAX_SWARM_SIZE:
        raise ValueError(f"size must be an integer from 1 to {_MAX_SWARM_SIZE}.")
    return size


def _validate_timeout(value: object, *, name: str) -> float:
    """Return ``value`` as a float, rejecting anything but a finite positive number."""
    seconds = _finite_number(value)
    if seconds is None or seconds <= 0:
        raise ValueError(f"{name} must be a finite positive number of seconds.")
    return seconds


def _resolve_url_timeout(url_timeout: object) -> float:
    """Resolve a URL-fetch timeout; ``None`` uses env/default, else validate."""
    return (
        _url_fetch_timeout()
        if url_timeout is None
        else _validate_timeout(url_timeout, name="url_timeout")
    )


def _validate_cite_min_confidence(value: object) -> float | None:
    """Return ``value`` when it is a finite number in ``[0, 1]``, or ``None``."""
    if value is None:
        return None
    number = _finite_number(value)
    if number is None or number < 0 or number > 1:
        raise ValueError("cite_min_confidence must be a finite number in [0, 1].")
    return number


def _validate_pages(value: object) -> tuple[int, ...] | None:
    """Return unique 1-based page numbers, or ``None`` for all pages.

    Out-of-range numbers are left in the tuple; the parser ignores them after
    the document's page count is known. An empty sequence raises ``ValueError``.
    """
    if value is None:
        return None
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError("pages must be a sequence of 1-based page numbers.")
    pages: list[int] = []
    seen: set[int] = set()
    for item in value:
        page = _positive_int(item)
        if page is None:
            raise ValueError("pages must be a sequence of 1-based page numbers.")
        if page not in seen:
            seen.add(page)
            pages.append(page)
    if not pages:
        raise ValueError("pages must include at least one 1-based page number.")
    return tuple(pages)


def parse_page_range(spec: str) -> tuple[int, ...]:
    """Parse compact 1-based page ranges such as ``1-3,5,8``.

    Tokens are comma-separated singles or ``start-end`` spans (inclusive).
    Whitespace around tokens is ignored. Duplicates keep first-seen order.
    Empty specs, empty tokens, non-digits, and inverted ranges raise
    ``ValueError``.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("pages must be a non-empty range such as '1-3,5,8'.")
    collected: list[int] = []
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            raise ValueError("pages contains an empty token.")
        parts = token.split("-")
        if len(parts) == 1:
            collected.append(_parse_page_number(parts[0].strip(), token))
            continue
        if len(parts) != 2:
            raise ValueError(f"invalid page range {token!r}.")
        start_text, end_text = parts[0].strip(), parts[1].strip()
        if not start_text or not end_text:
            raise ValueError(f"invalid page range {token!r}.")
        start = _parse_page_number(start_text, token)
        end = _parse_page_number(end_text, token)
        if end < start:
            raise ValueError(f"invalid page range {token!r}.")
        collected.extend(range(start, end + 1))
    pages = _validate_pages(collected)
    assert pages is not None
    return pages


def _parse_page_number(text: str, token: str) -> int:
    """Parse one 1-based page number; ``token`` is the original range fragment."""
    if not text.isdigit():
        label = "page range" if "-" in token else "page number"
        raise ValueError(f"invalid {label} {token!r}.")
    page = int(text)
    if page < 1:
        raise ValueError(f"invalid page number {token!r}.")
    return page
