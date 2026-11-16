"""Opt-in checks against real providers, and the guards that keep ordinary runs offline.

Live cases need ``--live`` and provider credentials. They compare every extracted field
with a frozen invoice, with retries disabled so failures stay visible.
"""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, Field

from openextract import Model, extract, extract_async

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/invoices"
MEDIA = json.loads((FIXTURES / "manifest.json").read_text())["files"]
SIGNATURES = {
    "application/pdf": b"%PDF-",
    "image/png": b"\x89PNG\r\n\x1a\n",
    "image/jpeg": b"\xff\xd8\xff",
}
INSTRUCTIONS = (
    "Read every page of this invoice. Include every line item from every page, in document order. "
    "Preserve the vendor, invoice number, and item descriptions exactly as printed. "
    "Use the final grand total for the whole invoice, not an individual page subtotal. "
    "Read paid status from the final page when present. "
    "Return the currency, quantities, and unit prices."
)
# Provider id: model, the variables that may hold its key, and its token budget.
PROVIDERS = {
    "openai": (
        Model("openai:gpt-4.1-mini", settings={"max_output_tokens": 1500}),
        "OPENAI_API_KEY",
    ),
    "gemini": (
        Model("google:gemini-2.5-flash", settings={"maxOutputTokens": 3000}),
        "GOOGLE_API_KEY or GEMINI_API_KEY",
    ),
    "anthropic": (
        Model("anthropic:claude-haiku-4-5-20251001", settings={"max_tokens": 1500}),
        "ANTHROPIC_API_KEY",
    ),
    "openrouter_gemini": (
        Model(
            "openrouter:google/gemini-2.5-flash",
            output_mode="json_schema",
            settings={"max_tokens": 3000, "provider": {"require_parameters": True}},
        ),
        "OPENROUTER_API_KEY",
    ),
}
KEY_VARIABLES = {name for _, names in PROVIDERS.values() for name in names.split(" or ")}


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


class CardPayment(BaseModel):
    kind: Literal["card"]
    last_four: str


class CashPayment(BaseModel):
    kind: Literal["cash"]
    currency: str


class StrictInvoice(BaseModel):
    model_config = ConfigDict(strict=True)

    issued: date
    total: Decimal = Field(
        description="The total amount as a decimal number, without currency text."
    )
    payment: Annotated[CardPayment | CashPayment, Field(discriminator="kind")]


def expected_invoice(media):
    return Invoice.model_validate_json((FIXTURES / media["expected"]).read_bytes())


def single_attempt(model, **changes):
    return replace(model, max_retries=0, validation_retries=0, timeout=90, **changes)


@pytest.fixture(params=MEDIA, ids=lambda media: media["id"])
def media(request):
    return request.param


@pytest.fixture(params=PROVIDERS)
def live_model(request, media):
    model, key_variables = PROVIDERS[request.param]
    if not any(os.environ.get(name) for name in key_variables.split(" or ")):
        pytest.fail(
            f"Set {key_variables} for the selected live provider, "
            "or load your local .env with uv run --env-file .env.",
            pytrace=False,
        )
    settings = dict(model.settings)
    if request.param == "openrouter_gemini" and media["media_type"] == "application/pdf":
        settings["plugins"] = [{"id": "file-parser", "pdf": {"engine": "native"}}]
    return single_attempt(model, settings=settings)


@pytest.mark.live
def test_live_extract_from_path(live_model, media):
    result = extract(Invoice, live_model, FIXTURES / media["file"], INSTRUCTIONS)
    assert result == expected_invoice(media)


@pytest.mark.live
async def test_live_extract_async_from_bytes(live_model, media):
    data = (FIXTURES / media["file"]).read_bytes()
    assert await extract_async(Invoice, live_model, data, INSTRUCTIONS) == expected_invoice(media)


@pytest.mark.live
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
async def test_live_strict_types_and_tagged_union(model, asynchronous):
    model = single_attempt(model)
    source = "Invoice issued 2026-11-10. Total: 12.50 USD. Paid by card ending in 1234."
    if asynchronous:
        result = await extract_async(StrictInvoice, model, source)
    else:
        result = extract(StrictInvoice, model, source)
    assert result == StrictInvoice(
        issued=date(2026, 11, 10),
        total=Decimal("12.50"),
        payment=CardPayment(kind="card", last_four="1234"),
    )


def run_live_suite(environment, *options):
    """Run this file's live cases in a subprocess, as a contributor would."""
    command = [sys.executable, "-m", "pytest", __file__, "-m", "live", "--no-cov", "-q"]
    environment = {name: value for name, value in environment.items() if name != "PYTEST_ADDOPTS"}
    return subprocess.run(
        [*command, "-p", "no:cacheprovider", *options],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_live_cases_stay_disabled_even_with_credentials():
    result = run_live_suite(os.environ | dict.fromkeys(KEY_VARIABLES, "dummy-not-a-real-key"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "62 skipped" in result.stdout


def test_explicit_live_run_requires_credentials():
    environment = {name: value for name, value in os.environ.items() if name not in KEY_VARIABLES}
    result = run_live_suite(environment, "--live", "-k", "openai and digital_pdf")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Set OPENAI_API_KEY" in result.stdout
    assert "2 errors" in result.stdout


async def test_offline_runs_block_real_http_requests():
    with (
        httpx.Client(trust_env=False) as client,
        pytest.raises(pytest.fail.Exception, match="Real HTTP requests require"),
    ):
        client.get("https://example.invalid")
    async with httpx.AsyncClient(trust_env=False) as client:
        with pytest.raises(pytest.fail.Exception, match="Real HTTP requests require"):
            await client.get("https://example.invalid")


def test_media_fixtures_match_their_manifest(media):
    path = FIXTURES / media["file"]
    assert path.resolve().is_relative_to(FIXTURES.resolve())
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == media["sha256"]
    assert data.startswith(SIGNATURES[media["media_type"]])
    expected = expected_invoice(media)
    line_total = sum(Decimal(str(item.unit_price)) * item.quantity for item in expected.items)
    assert Decimal(str(expected.total)) == line_total


def test_media_fixture_ids_and_files_are_unique():
    assert len({media["id"] for media in MEDIA}) == len(MEDIA)
    assert len({media["file"] for media in MEDIA}) == len(MEDIA)
