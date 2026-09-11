"""Opt-in per-field provenance for extraction results."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any, cast

from pydantic import BaseModel, Field, create_model

from ._parse import ParsedDocument, align_citations_to_window, ground_citations
from ._types import Citation, T

_MAX_QUOTE = 2000
_FIELD_PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])*$")
_CREDENTIAL_URL = re.compile(r"https?://[^\s/]*:[^\s/]*@[^\s]+", re.IGNORECASE)
_DATA_URI = re.compile(r"data:[^\s]+", re.IGNORECASE)

CITATION_INSTRUCTIONS = (
    "For every extracted non-null field you MUST return a citation. "
    "Return citations as a list of objects with field, quote, and page. "
    "field is a dotted path matching the schema (for example vendor or lines[0].qty). "
    "quote is the exact text span from the source; never invent text. "
    "A short quote is valid and must not be omitted. "
    "page is the 1-indexed page number from the '--- Page N ---' markers when "
    "present, or the document page when the source is paginated. Always include "
    "page when page markers exist. "
    "Do not include bounding boxes; location is resolved locally. "
    "Do not include raw file bytes, URLs, credentials, or file paths."
)


class CitationDraft(BaseModel):
    """Structured-output shape the model fills when ``cite=True``.

    Boxes are not requested from the model; the local parser attaches them.
    """

    field: str
    quote: str | None = None
    page: int | None = None


_WRAP_CACHE: dict[type[BaseModel], type[BaseModel]] = {}


def cited_output_schema(schema: type[BaseModel]) -> type[BaseModel]:
    """Wrap ``schema`` so the model also returns a ``citations`` list."""
    cached = _WRAP_CACHE.get(schema)
    if cached is not None:
        return cached
    wrapped = create_model(
        f"{schema.__name__}WithCitations",
        __module__="openextract._citations",
        output=(schema, ...),
        citations=(list[CitationDraft], Field(default_factory=list)),
    )
    _WRAP_CACHE[schema] = wrapped
    return wrapped


def json_schema_with_citations(schema: dict[str, Any]) -> dict[str, Any]:
    """Wrap a JSON Schema with the same ``output`` / ``citations`` envelope."""
    return {
        "type": "object",
        "properties": {
            "output": schema,
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string"},
                        "quote": {"type": "string"},
                        "page": {"type": "integer"},
                    },
                    "required": ["field"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["output", "citations"],
        "additionalProperties": False,
    }


def with_citation_instructions(instructions: str | None) -> str:
    """Append citation guidance without dropping caller instructions."""
    if instructions and instructions.strip():
        return f"{instructions.strip()}\n\n{CITATION_INSTRUCTIONS}"
    return CITATION_INSTRUCTIONS


def prepare_cited_run(
    schema: type[T],
    instructions: str | None,
    cite: bool,
) -> tuple[type[BaseModel], str | None]:
    """Return the output type and instructions for one extraction run."""
    if not cite:
        return schema, instructions
    return cited_output_schema(schema), with_citation_instructions(instructions)


def split_cited_output(
    raw: object,
    schema: type[T],
    *,
    cite: bool,
    parsed: ParsedDocument | None = None,
    window: ParsedDocument | None = None,
) -> tuple[T, tuple[Citation, ...]]:
    """Unwrap a cited model payload into ``(schema instance, citations)``.

    When ``cite`` is false the raw output is returned unchanged and citations
    are empty, matching the default extract path. When a local parse is
    provided, pages are stamped from it and boxes come only from parser spans.
    ``window`` remaps page=1 / missing pages onto the single page the model saw.
    """
    if not cite:
        return cast(T, raw), ()
    wrapper_type = cited_output_schema(schema)
    wrapper = cast(Any, raw if isinstance(raw, wrapper_type) else wrapper_type.model_validate(raw))
    output = cast(T, wrapper.output)
    citations = citations_from_payload(wrapper.citations)
    if window is not None:
        citations = align_citations_to_window(citations, window)
    citations = ground_citations(citations, parsed, output)
    return output, citations


def citations_from_payload(payload: object) -> tuple[Citation, ...]:
    """Sanitize a model or JSON citation list into public :class:`Citation` values."""
    if not isinstance(payload, Sequence) or isinstance(payload, str | bytes):
        return ()
    out: list[Citation] = []
    for item in payload:
        citation = sanitize_citation(item)
        if citation is not None:
            out.append(citation)
    return tuple(out)


def sanitize_citation(draft: object) -> Citation | None:
    """Drop unsafe or unusable citation payloads; never raise on a bad item.

    Model-supplied boxes are discarded here. Parser grounding attaches a box
    only when a real span matches. A valid page is enough to keep a citation,
    including when the quote is short.
    """
    if isinstance(draft, Citation | CitationDraft):
        field, quote, page = draft.field, draft.quote, draft.page
    elif isinstance(draft, dict):
        payload = cast(dict[str, Any], draft)
        field = payload.get("field") or payload.get("field_path")
        quote = payload.get("quote", payload.get("reference_text"))
        page = payload.get("page")
    else:
        return None
    if not isinstance(field, str) or not _FIELD_PATH.fullmatch(field):
        return None
    if isinstance(quote, str):
        quote = _sanitize_quote(quote) or None
    elif quote is not None:
        quote = None
    if page is not None and (not isinstance(page, int) or isinstance(page, bool) or page < 1):
        page = None
    if quote is None and page is None:
        return None
    return Citation(field=field, quote=quote, page=page, bbox=None)


def _sanitize_quote(quote: str) -> str:
    """Collapse whitespace and strip credential-bearing URLs and data URIs."""
    text = " ".join(quote.split())
    text = _CREDENTIAL_URL.sub("[redacted]", text)
    text = _DATA_URI.sub("[redacted]", text)
    if len(text) > _MAX_QUOTE:
        text = text[:_MAX_QUOTE]
    return text


def field_citations_for_extractbench(
    citations: Iterable[Citation],
) -> list[dict[str, Any]]:
    """Map library citations to ExtractBench ``FieldCitation`` payloads."""
    return [payload for citation in citations if (payload := citation.as_field_citation())]
