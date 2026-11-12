import json
from datetime import date, datetime, time
from decimal import Decimal
from typing import Annotated

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, Field, RootModel, create_model, field_validator

from openextract import (
    ExtractionResult,
    Model,
    OpenRouterField,
    SchemaValidationError,
    extract,
    extract_async,
)
from tests.helpers import configured, serve

CLAIRIC = "openrouter:cloudflare/clairic"
SCALEDOWN = "openrouter:scaledown/extract"


class Receipt(BaseModel):
    vendor: str


def test_text_returns_validated_data_and_original_metadata(monkeypatch):
    class Invoice(BaseModel):
        model_config = ConfigDict(strict=True)
        vendor: str = Field(alias="seller", description="Seller named on the invoice.")
        total: Annotated[Decimal, OpenRouterField(decimals=2, unit="USD")] = Field(gt=0)
        due_date: date
        paid_at: datetime
        cutoff: time
        purchase_order: Annotated[str, Field(min_length=2)] | None
        quantities: Annotated[list[Annotated[int, Field(gt=0)]], Field(min_length=1)] | None

    source = "📄 Acme invoice"
    options = {
        "reference_date": "2026-11-11",
        "provider": {"options": {"cloudflare": {"return_state_text": True}}},
    }
    answers = {
        "vendor": {"value": "Acme", "span": [2, 6], "confidence": 0.95},
        "total": {"value": 12.5},
        "due_date": {"value": "2026-11-20"},
        "paid_at": {"value": "2026-11-11T12:00:00"},
        "cutoff": {"value": "17:30:00"},
        "purchase_order": {"value": None, "guess": "INV-1", "p_null": 0.9},
        "quantities": {"value": [1, 2], "spans": [None, [7, 8]], "confidences": [0.8, 0.9]},
    }
    metadata = {"state_text": source, "truncated": False, "usage": {"cost": 0.0001}}

    def handler(request):
        assert request.method == "POST"
        assert str(request.url) == "https://proxy.example/api/alpha/extractions"
        assert request.headers["authorization"] == "Bearer env-key"
        body = json.loads(request.content)
        assert body == {
            **options,
            "model": "cloudflare/clairic",
            "state": source,
            "fields": {
                "vendor": {
                    "type": "text",
                    "instructions": "Seller named on the invoice.\nBe exact.",
                },
                "total": {
                    "type": "number",
                    "decimals": 2,
                    "unit": "USD",
                    "instructions": "total\nBe exact.",
                },
                "due_date": {"type": "date", "instructions": "due date\nBe exact."},
                "paid_at": {"type": "datetime", "instructions": "paid at\nBe exact."},
                "cutoff": {"type": "time", "instructions": "cutoff\nBe exact."},
                "purchase_order": {"type": "text", "instructions": "purchase order\nBe exact."},
                "quantities": {
                    "type": "list",
                    "items": "number",
                    "integer": True,
                    "instructions": "quantities\nBe exact.",
                },
            },
        }
        return httpx.Response(200, json={**metadata, "answers": answers})

    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://proxy.example/api/v1/")
    serve(monkeypatch, handler)
    result = extract(
        Invoice,
        Model(CLAIRIC, settings=options),
        source,
        "Be exact.",
        include_metadata=True,
    )
    assert isinstance(result, ExtractionResult)
    assert result.data.vendor == "Acme"
    assert result.data.total == Decimal("12.5")
    assert result.data.due_date == date(2026, 11, 20)
    assert result.data.paid_at == datetime(2026, 11, 11, 12)
    assert result.data.cutoff == time(17, 30)
    assert result.data.purchase_order is None
    assert result.data.quantities == [1, 2]
    assert result.answers == answers
    assert result.metadata == metadata


@pytest.mark.parametrize(
    "model_name,content,kind,file_format,encoded",
    [
        (SCALEDOWN, b"%PDF-1.7", "document", "pdf", "JVBERi0xLjc="),
        (CLAIRIC, b"\x89PNG", "image", "png", "iVBORw=="),
        (CLAIRIC, b"\xff\xd8\xff", "image", "jpeg", "/9j/"),
        (CLAIRIC, b"RIFF1234WEBP", "image", "webp", "UklGRjEyMzRXRUJQ"),
    ],
)
async def test_asset_request_and_values_only_result(
    monkeypatch, model_name, content, kind, file_format, encoded
):
    def handler(request):
        assert request.url.path == "/api/alpha/extractions"
        assert json.loads(request.content) == {
            "model": model_name.split(":", 1)[1],
            "state": "document",
            "assets": {"document": {"type": kind, "format": file_format, "data_base64": encoded}},
            "fields": {"vendor": {"type": "text", "instructions": "vendor"}},
        }
        return httpx.Response(200, json={"answers": {"vendor": {"value": "Acme"}}})

    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await extract_async(Receipt, model_name, content, client=client)
        assert result == Receipt(vendor="Acme")
        assert not client.is_closed


async def test_explicit_api_and_native_field_options(monkeypatch):
    class Terms(BaseModel):
        term_days: Annotated[int, OpenRouterField(type="duration", integer=True, unit="days")]
        days_to_pay: Annotated[float, OpenRouterField(type="estimate", min=0, max=90, step=1)]
        payments: Annotated[
            list[float],
            OpenRouterField(type="forecast", min=0, max=100, horizon=2, interval="month"),
        ]

    def handler(request):
        assert request.url.path == "/api/alpha/extractions"
        assert json.loads(request.content)["fields"] == {
            "term_days": {
                "type": "duration",
                "integer": True,
                "unit": "days",
                "instructions": "term days",
            },
            "days_to_pay": {
                "type": "estimate",
                "min": 0,
                "max": 90,
                "step": 1,
                "instructions": "days to pay",
            },
            "payments": {
                "type": "forecast",
                "min": 0,
                "max": 100,
                "horizon": 2,
                "interval": "month",
                "instructions": "payments",
            },
        }
        return httpx.Response(
            200,
            json={
                "answers": {
                    "term_days": {"value": 30},
                    "days_to_pay": {"value": 35, "p10": 30, "p90": 40},
                    "payments": {"value": [50, 50], "steps": [{"median": 50}, {"median": 50}]},
                }
            },
        )

    serve(monkeypatch, handler)
    model = configured("openrouter:cloudflare/clairic", api="openrouter-extractions")
    result = await extract_async(Terms, model, "Net 30", include_metadata=True)
    assert result.data == Terms(term_days=30, days_to_pay=35, payments=[50, 50])
    assert result.answers["days_to_pay"]["p10"] == 30
    assert result.answers["payments"]["steps"] == [{"median": 50}, {"median": 50}]


def test_validation_retry_preserves_source_and_returns_final_answer(monkeypatch):
    class Invoice(BaseModel):
        total: int

        @field_validator("total")
        @classmethod
        def positive(cls, value: int) -> int:
            if value <= 0:
                raise ValueError("Private validation detail")
            return value

    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        value = -9876 if len(requests) == 1 else 10
        return httpx.Response(200, json={"answers": {"total": {"value": value}}})

    serve(monkeypatch, handler)
    result = extract(Invoice, Model(CLAIRIC, api_key="test-key"), "Due: 10", include_metadata=True)
    assert result.data.total == 10
    assert result.answers == {"total": {"value": 10}}
    assert len(requests) == 2
    assert requests[0]["state"] == requests[1]["state"] == "Due: 10"
    retry = requests[1]["fields"]["total"]["instructions"]
    assert "total: Value did not satisfy a schema validator." in retry
    assert "Private validation detail" not in retry
    assert "-9876" not in retry


@pytest.mark.parametrize(
    "annotation", [Receipt, list[Receipt], list[time], list[()], str | int, bool]
)
def test_unsupported_schema_rejected_before_sending(annotation):
    schema = create_model("Unsupported", value=(annotation, ...))
    with pytest.raises(TypeError, match="cannot infer a field type for 'value'"):
        extract(schema, configured(CLAIRIC), "text")


def test_unsupported_inputs_and_reserved_settings_rejected_before_sending():
    with pytest.raises(TypeError, match="requires a Pydantic model with named fields"):
        extract(RootModel[str], configured(CLAIRIC), "text")
    with pytest.raises(ValueError, match="at least one schema field"):
        extract(BaseModel, configured(CLAIRIC), "text")
    with pytest.raises(ValueError, match="Received image/gif"):
        extract(Receipt, configured(CLAIRIC), b"GIF89a")
    with pytest.raises(ValueError, match="cannot override request fields"):
        extract(Receipt, configured(CLAIRIC, settings={"assets": {}}), "text")
    with pytest.raises(ValueError, match="requires the OpenRouter Extractions API"):
        extract(
            Receipt, configured("openrouter:google/gemini-2.5-flash"), "text", include_metadata=True
        )


@pytest.mark.parametrize(
    "answers",
    [
        [],
        {},
        {"vendor": {"value": "Acme"}, "extra": {"value": "other"}},
        {"vendor": "Acme"},
        {"vendor": {"confidence": 0.8}},
    ],
)
def test_invalid_answer_envelopes_raise_schema_error(monkeypatch, answers):
    serve(monkeypatch, lambda request: httpx.Response(200, json={"answers": answers}))
    with pytest.raises(SchemaValidationError, match="OpenRouter Extractions"):
        extract(Receipt, configured(CLAIRIC), "text")


def test_truncated_input_requires_metadata_opt_in(monkeypatch):
    payload = {"truncated": True, "answers": {"vendor": {"value": "Acme"}}}
    serve(monkeypatch, lambda request: httpx.Response(200, json=payload))
    with pytest.raises(SchemaValidationError, match="read only part of the source"):
        extract(Receipt, configured(CLAIRIC), "text")
    result = extract(Receipt, configured(CLAIRIC), "text", include_metadata=True)
    assert result.data == Receipt(vendor="Acme")
    assert result.metadata["truncated"] is True
