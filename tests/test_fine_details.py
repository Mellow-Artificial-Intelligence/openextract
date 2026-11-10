"""Check published controls offline, without claiming extraction quality."""

import json
import re
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

import openextract._extract as extract_module
from tests.helpers import output, serve

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def guide(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    calls = []
    contents = []
    overrides = []

    def respond(request):
        assert request.url == "https://api.openai.com/v1/responses"
        calls.append(json.loads(request.content))
        # Canned data exercises nullable results; it is not an extraction algorithm.
        currency = None if len(calls) == 2 else "USD"
        return httpx.Response(
            200,
            json=output(
                value={"vendor": "Acme Paper Co.", "total": "120.00", "currency": currency}
            ),
        )

    load = extract_module._load

    def record_load(source, media_type):
        overrides.append(media_type)
        content = load(source, media_type)
        contents.append(content)
        return content

    serve(monkeypatch, respond)
    monkeypatch.setattr(extract_module, "_load", record_load)
    namespace = {}
    blocks = re.findall(
        r"```python\n(.*?)```", (ROOT / "examples/fine_details.md").read_text(), re.S
    )
    assert len(blocks) == 6
    for index, code in enumerate(blocks):
        exec(compile(code, f"fine_details.md:block-{index}", "exec"), namespace)
    return namespace, calls, contents, overrides


def test_published_controls(guide):
    namespace, calls, contents, overrides = guide
    invoice_type = namespace["Invoice"]
    instructions = "Preserve names exactly. Do not guess missing or ambiguous currency."
    assert len(calls) == 3
    assert [call.get("instructions") for call in calls] == [None, instructions, None]
    assert all(call["model"] == "gpt-5" for call in calls)
    assert all(call["text"]["format"]["schema"]["title"] == "Invoice" for call in calls)
    assert overrides == [None, None, "text/plain"]
    assert [content.data for content in contents] == [
        (ROOT / "examples/fixtures/invoice.txt").read_bytes(),
        namespace["ambiguous_source"].encode(),
        namespace["source"].encode(),
    ]
    assert all(content.media_type == "text/plain" for content in contents)
    for name in ("invoice", "customized", "explicit_media"):
        result = namespace[name]
        assert type(result) is invoice_type
        assert result.vendor == "Acme Paper Co."
        assert isinstance(result.total, Decimal)
        assert result.total == Decimal("120.00")
    assert namespace["customized"].currency is None
    assert namespace["invoice"].currency == "USD"
    fields = invoice_type.model_json_schema()
    assert fields["required"] == ["vendor", "total", "currency"]
    assert fields["properties"]["vendor"]["description"] == "Seller, not buyer."
    assert fields["properties"]["total"]["description"] == "Final amount due, including tax."


def test_required_nullable_and_default(guide):
    namespace, *_ = guide
    data = {"vendor": "Acme Paper Co.", "total": "120.001"}
    with pytest.raises(ValidationError) as exc:
        namespace["Invoice"].model_validate(data)
    assert exc.value.errors()[0]["loc"] == ("currency",)
    assert exc.value.errors()[0]["type"] == "missing"
    result = namespace["Invoice"].model_validate({**data, "currency": None})
    assert result.currency is None
    assert result.total == Decimal("120.001")  # Decimal does not round to cents.
    assert namespace["InvoiceWithDefault"].model_validate(data).currency is None


@pytest.mark.parametrize("currency", ["usd", "US", "USDD", "123", "ÜSD", "", " USD "])
def test_currency_validator_rejects_malformed_values(guide, currency):
    namespace, *_ = guide
    with pytest.raises(ValidationError) as exc:
        namespace["InvoiceWithCurrencyCheck"].model_validate(
            {"vendor": "Acme Paper Co.", "total": "120.00", "currency": currency}
        )
    assert exc.value.errors()[0]["loc"] == ("currency",)
    assert exc.value.errors()[0]["type"] == "value_error"


@pytest.mark.parametrize("currency", [None, "USD", "EUR", "ZZZ"])
def test_currency_validator_accepts_shape_not_iso_membership(guide, currency):
    namespace, *_ = guide
    result = namespace["InvoiceWithCurrencyCheck"].model_validate(
        {"vendor": "Acme Paper Co.", "total": "120.00", "currency": currency}
    )
    assert result.currency == currency
