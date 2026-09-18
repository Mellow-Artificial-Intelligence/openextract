"""Extract invoice/statement line items with ``style='table'`` (TestModel, no API key).

``style='table'`` prepends row-oriented guidance and, for PDFs, reuses the
local parse-then-window path so line items across pages merge. Boxes are
still never invented — pass ``cite=True`` for parser-backed citations.

Live provider: swap ``MODEL`` for a provider id (or ``OPENEXTRACT_MODEL``)
and pass a real invoice or statement PDF.
"""

from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai.models.test import TestModel

from examples._shared import ACME_SNIPPET
from openextract import extract


class LineItem(BaseModel):
    description: str
    quantity: float
    amount: float


class Invoice(BaseModel):
    vendor: str
    line_items: list[LineItem]
    total: float


MODEL = TestModel(
    custom_output_args={
        "vendor": "Acme Corp",
        "line_items": [
            {"description": "Widget", "quantity": 2, "amount": 20.0},
            {"description": "Gadget", "quantity": 1, "amount": 22.0},
        ],
        "total": 42.0,
    }
)


def main() -> None:
    invoice = extract(
        schema=Invoice,
        model=MODEL,
        input_file=ACME_SNIPPET,
        style="table",
        instructions="Extract vendor, every line item, and total.",
    )
    print(invoice.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
