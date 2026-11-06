"""Check published controls offline, without claiming extraction quality."""

import re
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_ai import BinaryContent, models
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

import openextract._extract as extract_module

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def guide(monkeypatch):
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    calls = []
    contents = []
    overrides = []

    def respond(messages, info):
        content = next(
            item for item in messages[0].parts[-1].content if isinstance(item, BinaryContent)
        )
        contents.append(content)
        # Canned data exercises nullable results; it is not an extraction algorithm.
        currency = None if len(contents) == 2 else "USD"
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {"vendor": "Acme Paper Co.", "total": "120.00", "currency": currency},
                )
            ]
        )

    build_agent = extract_module._build_agent
    load = extract_module._load

    def offline_agent(schema, model, instructions):
        calls.append((schema, model, instructions))
        return build_agent(schema, FunctionModel(respond), instructions)

    def record_load(source, media_type):
        overrides.append(media_type)
        return load(source, media_type)

    monkeypatch.setattr(extract_module, "_build_agent", offline_agent)
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
    assert calls == [
        (invoice_type, "openai:gpt-5", None),
        (invoice_type, "openai:gpt-5", instructions),
        (invoice_type, "openai:gpt-5", None),
    ]
    assert overrides == [None, None, "text/plain"]
    assert [content.data for content in contents] == [
        (ROOT / "examples/fixtures/invoice.txt").read_bytes(),
        namespace["ambiguous_source"],
        namespace["source"],
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
