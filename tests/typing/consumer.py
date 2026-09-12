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
    Usage,
    extract,
    extract_many,
    extract_many_async,
    extract_many_with_results,
    extract_many_with_results_async,
    extract_with_usage,
    total_usage,
)


class Invoice(BaseModel):
    total: float


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
output, usage = extract_with_usage(Invoice, "openai:gpt-5", Path("/tmp/x.pdf"))
_assert_invoice: Invoice = output
_assert_usage: Usage = usage
cited: Invoice = extract(Invoice, "openai:gpt-5", Path("/tmp/x.pdf"), cite=True)
_cite: Citation = Citation("total", "12.50", 1, (0.1, 0.2, 0.3, 0.05))
_dumped: dict[str, object] = _cite.as_dict()
_field: dict[str, object] | None = _cite.as_field_citation()
_ = cited
_ = _dumped
_ = _field

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
    _ = async_defaulted
    _ = async_exceptions
    _ = async_results
