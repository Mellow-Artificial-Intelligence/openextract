"""Heuristic citation match strength. Not a model-provided probability."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Final, Literal, assert_never, cast

if TYPE_CHECKING:
    from ._types import Citation

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


def field_confidence(citations: Iterable[Citation]) -> dict[str, float]:
    """Minimum heuristic confidence per dotted field path.

    Among citations that share a :attr:`Citation.field`, the **minimum**
    non-``None`` :attr:`Citation.confidence` is kept so review workflows can
    gate on the weakest supporting span. Fields whose citations are all
    ``confidence is None`` are omitted. Empty input returns ``{}``.
    """
    scores: dict[str, float] = {}
    for citation in citations:
        confidence = citation.confidence
        if confidence is None:
            continue
        field = citation.field
        current = scores.get(field)
        if current is None or confidence < current:
            scores[field] = confidence
    return scores


def citations_by_field(citations: Iterable[Citation]) -> dict[str, tuple[Citation, ...]]:
    """Group citations by dotted field path, preserving order within each field.

    Field keys follow first-seen order. Empty input returns ``{}``.
    """
    grouped: dict[str, list[Citation]] = {}
    for citation in citations:
        grouped.setdefault(citation.field, []).append(citation)
    return {field: tuple(items) for field, items in grouped.items()}


def filter_citations(
    citations: Iterable[Citation],
    *,
    min_confidence: float | None = None,
    fields: Iterable[str] | None = None,
) -> tuple[Citation, ...]:
    """Keep citations that pass optional confidence and field filters.

    ``min_confidence`` compares :attr:`Citation.confidence`. Citations with
    ``confidence is None`` fail a min-confidence gate. ``fields``, when set,
    keeps only those dotted paths. Both ``None`` leaves the list unchanged
    (still returned as a tuple). Empty input returns ``()``.
    """
    allowed = None if fields is None else frozenset(fields)
    if min_confidence is None and allowed is None:
        return tuple(citations)
    return tuple(
        citation
        for citation in citations
        if (allowed is None or citation.field in allowed)
        and (
            min_confidence is None
            or (citation.confidence is not None and citation.confidence >= min_confidence)
        )
    )
