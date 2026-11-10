"""Extract the supplied fictional invoice, or a file path / URL of your own.

From the repository root, after setting OPENAI_API_KEY:
    uv run python examples/basic.py
    uv run python examples/basic.py examples/fixtures/invoice.txt
"""

import sys
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, Field

from openextract import extract


class Invoice(BaseModel):
    vendor: str = Field(description="Seller, not buyer.")
    total: Decimal = Field(description="Final amount due, including tax.")
    currency: str | None = Field(description="Three-letter currency code, or null if unclear.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        argument = sys.argv[1]
        source = argument if argument.startswith(("http://", "https://")) else Path(argument)
    else:
        source = Path(__file__).parent / "fixtures/invoice.txt"
    invoice = extract(Invoice, "openai:gpt-5", source)
    print(invoice.vendor)
    print(invoice.total)
    print(invoice.currency)
