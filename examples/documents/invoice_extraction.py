"""Extract structured invoice data from a PDF (or pass --fixture for the sample image)."""

import sys
from datetime import date

from pydantic import BaseModel

from examples._shared import DOCUMENT_PAGE, anthropic_model, require_input
from openextract import extract


class LineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    total: float


class Invoice(BaseModel):
    invoice_number: str | None = None
    issue_date: date | None = None
    seller: str | None = None
    buyer: str | None = None
    line_items: list[LineItem]
    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None


def main() -> None:
    input_file = require_input(
        sys.argv,
        "Usage: uv run python examples/documents/invoice_extraction.py <invoice.pdf>",
    )
    if input_file == "--fixture":
        input_file = str(DOCUMENT_PAGE)

    invoice = extract(
        schema=Invoice,
        model=anthropic_model(),
        input_file=input_file,
        style="table",
        instructions=(
            "Extract invoice metadata, parties, every line item, and totals when present. "
            "Use ISO 8601 for dates. Use null for fields not found."
        ),
    )

    print(invoice.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
