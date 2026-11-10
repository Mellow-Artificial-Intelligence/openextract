"""Preprocess inline text and reuse extraction with ordinary Python.

After the setup in examples/composition.md:
    uv run --no-sync python examples/composition.py
"""

from decimal import Decimal
from functools import partial

from pydantic import BaseModel, Field
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings

from openextract import extract

INSTRUCTIONS = "Preserve names exactly. Do not guess missing or ambiguous currency."


class Invoice(BaseModel):
    vendor: str = Field(description="Seller, not buyer.")
    total: Decimal = Field(description="Final amount due, including tax.")
    currency: str | None = Field(description="Three-letter currency code, or null if unclear.")


def prepare_text(text: str) -> bytes:
    """Normalize Windows line endings without changing names or amounts."""
    return text.replace("\r\n", "\n").encode("utf-8")


def make_model() -> AnthropicModel:
    """Use the dependency pair verified in configured_models.md."""
    return AnthropicModel("claude-sonnet-4-5", settings=AnthropicModelSettings(max_tokens=1024))


def extract_text(text: str, model: Model) -> Invoice:
    """An application-owned function binds the schema and interpretation policy."""
    extract_invoice = partial(extract, Invoice, model, instructions=INSTRUCTIONS)
    return extract_invoice(source=prepare_text(text), media_type="text/plain")


if __name__ == "__main__":
    text = "Seller: Acme Paper Co.\r\nTotal due: 120.00 USD"
    invoice = extract_text(text, make_model())
    print(invoice.model_dump_json())
