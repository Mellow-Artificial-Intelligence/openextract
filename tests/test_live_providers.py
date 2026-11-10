"""Opt-in extraction checks against real providers; never load credentials at import time."""

import os
from dataclasses import dataclass
from typing import Any, Literal

import pytest

from openextract import Model, extract, extract_async
from tests.media_fixtures import MEDIA_FIXTURES, Invoice

pytestmark = pytest.mark.live

INSTRUCTIONS = (
    "Read every page of this invoice. Include every line item from every page, in document order. "
    "Preserve the vendor, invoice number, and item descriptions exactly as printed. "
    "Use the final grand total for the whole invoice, not an individual page subtotal. "
    "Read paid status from the final page when present. "
    "Return the currency, quantities, and unit prices."
)


@dataclass(frozen=True)
class Provider:
    id: str
    model: str
    key_envs: tuple[str, ...]
    settings: dict[str, Any]
    output_mode: Literal["json_schema"] | None = None


PROVIDERS = (
    Provider("openai", "openai:gpt-4.1-mini", ("OPENAI_API_KEY",), {"max_output_tokens": 1500}),
    Provider(
        "gemini",
        "google:gemini-2.5-flash",
        ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        {"maxOutputTokens": 3000},
    ),
    Provider(
        "anthropic",
        "anthropic:claude-haiku-4-5-20251001",
        ("ANTHROPIC_API_KEY",),
        {"max_tokens": 1500},
    ),
    Provider(
        "openrouter_gemini",
        "openrouter:google/gemini-2.5-flash",
        ("OPENROUTER_API_KEY",),
        {"max_tokens": 3000, "provider": {"require_parameters": True}},
        "json_schema",
    ),
)


@pytest.fixture(params=PROVIDERS, ids=lambda provider: provider.id)
def provider(request):
    return request.param


@pytest.fixture(params=MEDIA_FIXTURES, ids=lambda fixture: fixture.id)
def media_fixture(request):
    return request.param


@pytest.fixture
def live_model(provider, media_fixture):
    if not any(os.environ.get(name) for name in provider.key_envs):
        pytest.fail(
            f"Set {' or '.join(provider.key_envs)} for the selected live provider, "
            "or load your local .env with uv run --env-file .env.",
            pytrace=False,
        )
    settings = dict(provider.settings)
    if provider.id == "openrouter_gemini" and media_fixture.media_type == "application/pdf":
        settings["plugins"] = [{"id": "file-parser", "pdf": {"engine": "native"}}]
    return Model(
        provider.model,
        settings=settings,
        output_mode=provider.output_mode,
        max_retries=0,
        validation_retries=0,
        timeout=90,
    )


def test_live_extract_from_path(live_model, media_fixture):
    result = extract(Invoice, live_model, media_fixture.path, INSTRUCTIONS)
    assert result == media_fixture.expected_invoice()


async def test_live_extract_async_from_bytes(live_model, media_fixture):
    result = await extract_async(Invoice, live_model, media_fixture.path.read_bytes(), INSTRUCTIONS)
    assert result == media_fixture.expected_invoice()
