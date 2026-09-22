"""Stable, secret-free diagnostic strings for :class:`ExtractionResult.warnings`."""

from __future__ import annotations

from collections.abc import Sequence

from ._parse import ParsedDocument


def pages_requested_warning(dropped: int, requested: int) -> str:
    """One requested-page drop, with counts only."""
    return f"pages filter dropped {dropped} of {requested} requested pages"


def pages_available_warning(dropped: int, available: int) -> str:
    """Pages allowlist excluded available document pages."""
    return f"pages filter dropped {dropped} of {available} available pages"


def max_pages_warning(max_pages: int, dropped: int, available: int) -> str:
    """``max_pages`` excluded available document pages."""
    return f"max_pages={max_pages} dropped {dropped} of {available} available pages"


def cite_min_confidence_warning(threshold: float, dropped: int) -> str:
    """``cite_min_confidence`` dropped grounded citations."""
    noun = "citation" if dropped == 1 else "citations"
    return f"cite_min_confidence={threshold:g} dropped {dropped} {noun}"


def page_filter_warnings(
    *,
    available: int | None,
    kept: int,
    pages: Sequence[int] | None,
    max_pages: int | None,
) -> tuple[str, ...]:
    """Warn when ``pages`` / ``max_pages`` drop requested or available pages.

    ``available`` is the source page count before filters. Missing it (non-PDF
    or unparsed) yields no page warnings. Intentional no-ops — every requested
    page exists and every available page is kept — stay silent.
    """
    if available is None or (pages is None and max_pages is None):
        return ()
    warnings: list[str] = []
    if pages is not None:
        requested = len(pages)
        missed = sum(1 for page in pages if page < 1 or page > available)
        if missed:
            warnings.append(pages_requested_warning(missed, requested))
    if kept < available:
        dropped = available - kept
        if max_pages is not None and pages is None:
            warnings.append(max_pages_warning(max_pages, dropped, available))
        else:
            warnings.append(pages_available_warning(dropped, available))
    return tuple(warnings)


def extraction_warnings(
    parsed: ParsedDocument | None,
    pages: Sequence[int] | None,
    max_pages: int | None,
    extra: Sequence[str] = (),
) -> tuple[str, ...]:
    """Page-filter warnings plus any citation (or other) extras, de-duplicated."""
    page = ()
    if parsed is not None:
        page = page_filter_warnings(
            available=parsed.available_pages,
            kept=len(parsed.pages),
            pages=pages,
            max_pages=max_pages,
        )
    return merge_warnings(page, extra)


def merge_warnings(*groups: Sequence[str]) -> tuple[str, ...]:
    """Flatten warning groups, dropping duplicates while keeping first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for group in groups:
        for item in group:
            if item not in seen:
                seen.add(item)
                out.append(item)
    return tuple(out)
