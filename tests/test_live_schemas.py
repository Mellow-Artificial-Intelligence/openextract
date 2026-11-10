"""Opt-in regression checks for provider schema conversion and strict JSON validation."""

from dataclasses import replace
from datetime import date
from decimal import Decimal
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field

from openextract import Model, extract, extract_async

pytestmark = pytest.mark.live


class CardPayment(BaseModel):
    kind: Literal["card"]
    last_four: str


class CashPayment(BaseModel):
    kind: Literal["cash"]
    currency: str


class Invoice(BaseModel):
    model_config = ConfigDict(strict=True)

    issued: date
    total: Decimal = Field(
        description="The total amount as a decimal number, without currency text."
    )
    payment: Annotated[CardPayment | CashPayment, Field(discriminator="kind")]


@pytest.mark.parametrize(
    "model",
    [
        Model("openai:gpt-4.1-mini", settings={"max_output_tokens": 500}),
        Model("openai-chat:gpt-4.1-mini", output_mode="json_schema", settings={"max_tokens": 500}),
        Model("anthropic:claude-haiku-4-5-20251001", settings={"max_tokens": 500}),
    ],
    ids=["openai_responses", "openai_chat_json", "anthropic"],
)
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_live_strict_types_and_discriminated_union(model, asynchronous):
    model = replace(model, max_retries=0, validation_retries=0, timeout=90)
    source = b"Invoice issued 2026-11-10. Total: 12.50 USD. Paid by card ending in 1234."
    result = (
        await extract_async(Invoice, model, source)
        if asynchronous
        else extract(Invoice, model, source)
    )
    assert result == Invoice(
        issued=date(2026, 11, 10),
        total=Decimal("12.50"),
        payment=CardPayment(kind="card", last_four="1234"),
    )
