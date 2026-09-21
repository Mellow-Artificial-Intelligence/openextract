"""Representative consumer code type-checked by tests/test_typing.py.

This module is intentionally not run at runtime; ``tests/test_typing.py`` runs
``ty check`` on it to prove the public API infers the documented types. The
explicit annotated assignments below are the assertions: if an overload drifted,
``ty`` would flag a mismatch here.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from openextract import (
    Citation,
    ExtractionInput,
    ExtractionResult,
    ExtractionStyle,
    ExtractProgress,
    SwarmResult,
    Usage,
    extract,
    extract_many,
    extract_many_async,
    extract_many_with_results,
    extract_many_with_results_async,
    extract_swarm_with_results,
    extract_with_result,
    extract_with_result_async,
    extract_with_usage,
    field_confidence,
    schema_from_json,
    total_usage,
)


class Invoice(BaseModel):
    total: float


JsonInvoice = schema_from_json("invoice.json")
extract(JsonInvoice, "openai:gpt-5", Path("/tmp/invoice.pdf"))


def _report(progress: ExtractProgress) -> None:
    _ = progress.current, progress.total, progress.page, progress.pages


# Path / os.PathLike works directly in every public API.
extract(Invoice, "openai:gpt-5", Path("/tmp/invoice.pdf"))
extract_many(Invoice, "openai:gpt-5", [Path("/tmp/a.pdf"), Path("/tmp/b.pdf")])
extract_many(Invoice, "openai:gpt-5", [ExtractionInput(Path("/tmp/a.pdf"))])
extract_many(
    Invoice,
    "openai:gpt-5",
    [
        ExtractionInput(source=b"pdf", media_type="application/pdf"),
        ExtractionInput(source=b"png", media_type="image/png"),
    ],
)

# Heterogeneous bytes/file inputs can carry per-item media types in one batch.
mixed: list[Invoice] = extract_many(
    Invoice,
    "openai:gpt-5",
    [
        ExtractionInput(source=b"pdf", media_type="application/pdf"),
        ExtractionInput(source=b"png", media_type="image/png"),
    ],
)

# return_exceptions infers list[Invoice] vs list[Invoice | Exception].
defaulted: list[Invoice] = extract_many(Invoice, "openai:gpt-5", ["a.pdf"])
with_exceptions: list[Invoice | Exception] = extract_many(
    Invoice, "openai:gpt-5", ["a.pdf"], return_exceptions=True
)

# extract() -> T and the tuple usage helper stay compatible.
single: Invoice = extract(Invoice, "openai:gpt-5", Path("/tmp/x.pdf"))
search: Invoice = extract(
    Invoice, "openai:gpt-5", Path("/tmp/notes.txt"), style=ExtractionStyle.SEARCH
)
table: Invoice = extract(
    Invoice, "openai:gpt-5", Path("/tmp/statement.pdf"), style=ExtractionStyle.TABLE
)
form: Invoice = extract(Invoice, "openai:gpt-5", Path("/tmp/w9.pdf"), style=ExtractionStyle.FORM)
output, usage = extract_with_usage(Invoice, "openai:gpt-5", Path("/tmp/x.pdf"))
_assert_invoice: Invoice = output
_assert_usage: Usage = usage
oneshot: ExtractionResult[Invoice] = extract_with_result(
    Invoice, "openai:gpt-5", Path("/tmp/x.pdf"), cite=True
)
_assert_oneshot: Invoice = oneshot.output
_ = oneshot.citations
cited: Invoice = extract(
    Invoice,
    "openai:gpt-5",
    Path("/tmp/x.pdf"),
    cite=True,
    cite_min_confidence=0.5,
    pages=(1, 2),
    language="es",
    model_settings={"temperature": 0},
    timeout=30,
    on_progress=_report,
)
_cite: Citation = Citation("total", "12.50", 1, (0.1, 0.2, 0.3, 0.05))
_dumped: dict[str, object] = _cite.as_dict()
_field: dict[str, object] | None = _cite.as_field_citation()
_confidence: float | None = _cite.confidence
_match: str | None = _cite.match
_ = cited
_ = _dumped
_ = _field
_ = _confidence
_ = _match
_result_dump: dict[str, object] = oneshot.as_dict()
_per_field: dict[str, float] = field_confidence(oneshot.citations)
_result_field: dict[str, float] = oneshot.field_confidence()
_ = _result_dump
_ = _per_field
_ = _result_field

# extract_swarm_with_results returns SwarmResult with reduced citations.
swarm: SwarmResult[Invoice] = extract_swarm_with_results(
    Invoice, "openai:gpt-5", Path("/tmp/x.pdf"), cite=True
)
_swarm_cite: tuple[Citation, ...] = swarm.citations
_ = _swarm_cite

# extract_many_with_results returns ExtractionResult[Invoice] (or + Exception).
results: list[ExtractionResult[Invoice]] = extract_many_with_results(
    Invoice, "openai:gpt-5", ["a.pdf"]
)
results_with_exceptions: list[ExtractionResult[Invoice] | Exception] = extract_many_with_results(
    Invoice, "openai:gpt-5", ["a.pdf"], return_exceptions=True
)

# Aggregate usage across batch results.
aggregate: Usage = total_usage(results)


async def async_consumer() -> None:
    async_defaulted: list[Invoice] = await extract_many_async(Invoice, "openai:gpt-5", ["a.pdf"])
    async_exceptions: list[Invoice | Exception] = await extract_many_async(
        Invoice, "openai:gpt-5", ["a.pdf"], return_exceptions=True
    )
    async_results: list[ExtractionResult[Invoice]] = await extract_many_with_results_async(
        Invoice, "openai:gpt-5", ["a.pdf"]
    )
    async_oneshot: ExtractionResult[Invoice] = await extract_with_result_async(
        Invoice, "openai:gpt-5", Path("/tmp/x.pdf")
    )
    _ = async_defaulted
    _ = async_exceptions
    _ = async_results
    _ = async_oneshot
