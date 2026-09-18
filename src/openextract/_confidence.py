"""Heuristic citation match strength. Not a model-provided probability."""

from __future__ import annotations

from typing import Final, Literal, assert_never, cast

# How the local parser (or quote/value check) supported this citation.
CitationMatch = Literal["exact", "numeric", "fuzzy", "value", "page", "quote"]

# Verified scores: a local parse located the quote or extracted value.
MATCH_SCORES: Final[dict[CitationMatch, float]] = {
    "exact": 0.95,
    "numeric": 0.85,
    "value": 0.80,
    "fuzzy": 0.65,
    "page": 0.40,
    "quote": 0.50,
}

# No local parse: the quote/page is model-claimed, not span-matched.
_UNVERIFIED_PAGE = 0.30
_UNVERIFIED_QUOTE = 0.45
_QUOTE_AGREES = 0.55
_QUOTE_MISMATCH = 0.25


def score_citation_match(
    match: str | None,
    *,
    parsed: bool,
    agrees: bool | None = None,
) -> float | None:
    """Return a ``[0, 1]`` heuristic, or ``None`` when ``match`` is unknown."""
    if match not in MATCH_SCORES:
        return None
    kind = cast(CitationMatch, match)
    if parsed:
        return MATCH_SCORES[kind]
    match kind:
        case "quote":
            if agrees is True:
                return _QUOTE_AGREES
            if agrees is False:
                return _QUOTE_MISMATCH
            return _UNVERIFIED_QUOTE
        case "page":
            return _UNVERIFIED_PAGE
        case "exact" | "numeric" | "fuzzy" | "value":
            return MATCH_SCORES[kind]
    assert_never(kind)  # pragma: no cover - exhaustive CitationMatch
