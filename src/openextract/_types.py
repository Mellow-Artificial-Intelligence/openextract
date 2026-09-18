"""Public input, result, usage, and retry contracts."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import BinaryIO, TypeVar

from pydantic import BaseModel

from ._config import _DEFAULT_RETRY_MAX_BACKOFF, _validate_retry_options

T = TypeVar("T", bound=BaseModel)

# A raw media source accepted directly by the public APIs: a local path or
# http(s) URL string, an ``os.PathLike`` (e.g. ``pathlib.Path``), raw bytes, or
# a binary file-like object with a ``.read()`` method.
MediaSource = str | os.PathLike[str] | bytes | BinaryIO

# A media source after ``os.fspath`` normalization: no ``os.PathLike`` remains.
ResolvedSource = str | bytes | BinaryIO


@dataclass(frozen=True)
class ExtractionInput:
    """A single input for extraction with optional per-item media metadata.

    Wraps a raw :data:`MediaSource` so heterogeneous batch inputs can specify
    their own ``media_type`` (and an optional safe ``name`` for diagnostics)
    without falling back to a single batch-wide media type.

    Attributes:
        source: The media source — a local path, ``http(s)://`` URL,
            ``os.PathLike``, raw ``bytes``, or a binary file-like object.
        media_type: Optional MIME type for this item. Required when ``source``
            is ``bytes`` or a file-like object and no batch-wide ``media_type``
            override is supplied. Overrides inference for path/URL sources.
        name: Optional safe source name recorded on :class:`ExtractionResult`
            diagnostics. Never populated with raw content or credentials.
    """

    source: MediaSource
    media_type: str | None = None
    name: str | None = None


# Anything accepted as a single input or batch item: a raw :data:`MediaSource`
# or a structured :class:`ExtractionInput`.
ExtractionInputLike = MediaSource | ExtractionInput


@dataclass(frozen=True)
class Usage:
    """Token usage information for a single extraction call."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class ExtractProgress:
    """One parse-window tick during a single extraction.

    Emitted immediately before that window is sent to the model, including the
    one-window fast path. ``current`` is 1-indexed. Unparsed inputs (no local
    PDF parse) emit a single ``current=1, total=1`` event with empty pages.

    Attributes:
        current: 1-indexed window about to run.
        total: Number of windows for this input.
        page: First 1-indexed page in this window, or ``None`` when unparsed.
        pages: All 1-indexed pages in this window (empty when unparsed).
    """

    current: int
    total: int
    page: int | None = None
    pages: tuple[int, ...] = ()


OnProgress = Callable[[ExtractProgress], None]


@dataclass(frozen=True)
class Citation:
    """A sanitized source span supporting one extracted field.

    Produced when ``cite=True`` is passed to an extract API. Shaped to map
    onto ExtractBench ``FieldCitation`` (``field_path``, ``page``, ``bbox``,
    ``reference_text``). Never holds raw media, credentials, query strings,
    fragments, or provider internals. Boxes are attached only when a local
    parser matched the quoted span; they are never invented or taken from
    the model.

    ``confidence`` is a local heuristic in ``[0, 1]``, not a model-reported
    probability. ``match`` says how it was derived: ``exact`` (quote in the
    parse), ``numeric`` (numeric/punctuation variant), ``fuzzy`` (approximate
    quote), ``value`` (extracted field value located after the quote missed),
    ``page`` (page only, no span), or ``quote`` (quote present but no parse
    to verify). Grounding stamps both; they stay ``None`` on manually built
    citations until then.

    Attributes:
        field: Dotted schema path (for example ``vendor`` or ``lines[0].qty``).
        quote: Verbatim text span from the source, when present.
        page: 1-indexed page number when the source is paginated.
        bbox: Normalized COCO ``(x, y, width, height)`` in ``[0, 1]`` when
            the parser located the span. ``None`` when no span matches
            (page-level grounding can still score).
        confidence: Heuristic match strength in ``[0, 1]``, or ``None``.
        match: How ``confidence`` was derived, or ``None``.
    """

    field: str
    quote: str | None = None
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    confidence: float | None = None
    match: str | None = None

    def as_dict(self) -> dict[str, object]:
        """JSON-stable citation including additive ``confidence`` / ``match``.

        Keys are ``field``, ``quote``, ``page``, ``bbox``, ``confidence``,
        ``match``. ``bbox`` is a list of four floats or ``None`` (never
        invented). ``confidence`` is a heuristic ``[0, 1]`` float or
        ``None``; ``match`` is the derivation label or ``None``. Quote-only
        citations are included; :meth:`as_field_citation` still requires a
        page for ExtractBench scoring and omits ``confidence`` / ``match``.
        """
        return {
            "field": self.field,
            "quote": self.quote,
            "page": self.page,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "confidence": self.confidence,
            "match": self.match,
        }

    def as_field_citation(self) -> dict[str, object] | None:
        """ExtractBench ``FieldCitation`` payload, or ``None`` without a page.

        ExtractBench requires ``page >= 1``. A quote-only citation is kept on
        :class:`ExtractionResult` but cannot be scored there. ``bbox`` is
        omitted unless a local parser supplied a normalized box.
        """
        if self.page is None:
            return None
        dumped = self.as_dict()
        return {
            "field_path": dumped["field"],
            "page": dumped["page"],
            "bbox": dumped["bbox"],
            "reference_text": dumped["quote"],
        }


@dataclass(frozen=True)
class ExtractionResult[T]:
    """Diagnostics-rich result of one extraction.

    ``extract_many_with_results`` returns these so callers can account for
    token usage, observe retries and timing, and record safe provenance without
    retaining raw media, credentials, query strings, or provider internals.

    Attributes:
        output: The validated schema instance.
        usage: Token usage from the successful model call.
        attempts: Number of model-call attempts, including the initial call and
            any :class:`ModelError` retries (always ``>= 1`` on success).
        duration: Wall-clock seconds spent on this item, including retries.
        model: The model identifier that produced the output, when known.
        media_type: The media type requested for this item, when provided.
        source: A sanitized source label (``ExtractionInput.name``, or a
            credential/query-stripped path/URL context); ``None`` for unnamed
            bytes/file-like inputs.
        warnings: Extensible, currently empty diagnostics channel.
        citations: Per-field source spans when ``cite=True``; empty otherwise.
    """

    output: T
    usage: Usage
    attempts: int
    duration: float
    model: str | None
    media_type: str | None
    source: str | None
    warnings: tuple[str, ...] = ()
    citations: tuple[Citation, ...] = ()


@dataclass(frozen=True)
class RetryPolicy:
    """Retry configuration shared by every call made through an extractor session."""

    max_retries: int = 0
    backoff: float = 1.0
    max_backoff: float = _DEFAULT_RETRY_MAX_BACKOFF

    def __post_init__(self) -> None:
        _validate_retry_options(self.max_retries, self.backoff, self.max_backoff)


def total_usage(results: Iterable[ExtractionResult[T]]) -> Usage:
    """Sum token usage across batch extraction results.

    ``results`` typically comes from :func:`extract_many_with_results` or
    :func:`extract_many_with_results_async`. Only successful items carry a
    :class:`Usage`, so totals reflect the successful calls in the batch.
    """
    return _sum_usage(result.usage for result in results)


def _extraction_result(
    output: T,
    usage: Usage,
    *,
    attempts: int,
    started: float,
    model: str | None,
    media_type: str | None,
    source: str | None,
    citations: tuple[Citation, ...] = (),
) -> ExtractionResult[T]:
    """Build the diagnostics wrapper shared by batch and swarm successes."""
    return ExtractionResult(
        output=output,
        usage=usage,
        attempts=attempts,
        duration=time.perf_counter() - started,
        model=model,
        media_type=media_type,
        source=source,
        citations=citations,
    )


def _sum_usage(usages: Iterable[Usage]) -> Usage:
    """Sum raw :class:`Usage` values from windowed or multi-call extractions."""
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    for usage in usages:
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens
        total_tokens += usage.total_tokens
    return Usage(input_tokens, output_tokens, total_tokens)


def _resolve_item(
    item: ExtractionInputLike,
    global_media_type: str | None,
) -> tuple[MediaSource, str | None, str | None]:
    """Split a batch item into ``(source, effective media type, safe name)``.

    A per-item :class:`ExtractionInput` media type wins over the batch-wide
    ``media_type`` fallback so heterogeneous inputs can use different types.
    """
    if isinstance(item, ExtractionInput):
        media_type = item.media_type if item.media_type is not None else global_media_type
        return item.source, media_type, item.name
    return item, global_media_type, None
