"""Cite extracted fields with ``cite=True`` (TestModel, no API key).

``extract()`` / ``extract_with_usage()`` still return the schema instance.
Read citations from ``extract_many_with_results`` → ``ExtractionResult.citations``.

``bbox`` is attached only when a local PDF parser matches the quote
(``openextract[pdf]`` / ``openextract[all]``). Boxes are never invented
or taken from the model.

Live provider: swap ``MODEL`` for a provider id (or ``OPENEXTRACT_MODEL``)
and pass a real PDF.
"""

from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai.models.test import TestModel

from examples._shared import ACME_SNIPPET
from openextract import ExtractionInput, extract_many_with_results, extract_with_usage


class InvoiceSnippet(BaseModel):
    vendor: str
    total: float


MODEL = TestModel(
    custom_output_args={
        "output": {"vendor": "Acme Corp", "total": 42.0},
        "citations": [
            {"field": "vendor", "quote": "Acme Corp", "page": 1},
            {"field": "total", "quote": "42.00", "page": 1},
        ],
    }
)


def main() -> None:
    source = ExtractionInput(ACME_SNIPPET, media_type="application/pdf", name=ACME_SNIPPET.name)

    output, usage = extract_with_usage(
        schema=InvoiceSnippet,
        model=MODEL,
        input_file=source,
        instructions="Extract vendor and total.",
        cite=True,
    )
    print(output.model_dump_json(indent=2))
    print(
        f"\ntokens: {usage.input_tokens} in / {usage.output_tokens} out / "
        f"{usage.total_tokens} total"
    )

    result = extract_many_with_results(
        schema=InvoiceSnippet,
        model=MODEL,
        input_files=[source],
        instructions="Extract vendor and total.",
        cite=True,
    )[0]
    print("\ncitations:")
    for citation in result.citations:
        print(f"  {citation.as_dict()}")
        mapped = citation.as_field_citation()
        if mapped is not None:
            print(f"    ExtractBench: {mapped}")


if __name__ == "__main__":
    main()
