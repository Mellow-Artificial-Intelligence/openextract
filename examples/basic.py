"""Extract an invoice from a file path or URL.

uv run --extra openai python examples/basic.py invoice.pdf
"""

import sys

from pydantic import BaseModel

from openextract import extract


class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str


invoice = extract(Invoice, "openai:gpt-5", sys.argv[1])
print(invoice.model_dump_json(indent=2))
