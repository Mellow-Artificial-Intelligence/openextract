"""Frozen media inputs and the invoice schema shared by offline and live tests."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "invoices"


class Item(BaseModel):
    description: str
    quantity: int
    unit_price: float


class Invoice(BaseModel):
    vendor: str
    invoice_number: str
    total: float
    currency: Literal["USD", "EUR"]
    is_paid: bool
    items: list[Item] = Field(
        min_length=1, description="Every line item in the invoice, preserving its description."
    )


@dataclass(frozen=True)
class MediaFixture:
    id: str
    file: str
    expected: str
    media_type: str
    sha256: str

    @property
    def path(self) -> Path:
        return FIXTURE_DIR / self.file

    def expected_invoice(self) -> Invoice:
        return Invoice.model_validate_json((FIXTURE_DIR / self.expected).read_bytes())


MANIFEST = json.loads((FIXTURE_DIR / "manifest.json").read_text())
MEDIA_FIXTURES = tuple(MediaFixture(**entry) for entry in MANIFEST["files"])
