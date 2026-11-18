"""The extraction contract: a typed result, validation with retries, and readable failures."""

import httpx
import pytest
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from openextract import ModelError, SchemaValidationError, extract, extract_async
from tests.helpers import Invoice, configured, output


async def test_extracts_a_validated_instance(run, provider, tmp_path):
    path = tmp_path / "invoice.pdf"
    path.write_bytes(b"%PDF-1.7 fake")
    assert await run(Invoice, configured(), path, "Find the total.") == Invoice(total=42.0)
    request, body = provider.requests[0], provider.bodies[0]
    assert request.url == "https://api.openai.com/v1/responses"
    assert request.headers["authorization"] == "Bearer test-key"
    assert body["instructions"] == "Find the total."
    assert body["input"][0]["content"][-1]["filename"] == "document.pdf"
    assert body["text"]["format"]["schema"]["additionalProperties"] is False


@pytest.mark.parametrize(
    "options",
    [{}, {"instructions": "Use the final total.", "media_type": "text/markdown"}],
    ids=["simple", "customized"],
)
async def test_options_do_not_change_the_result_type(run, provider, options):
    result = await run(Invoice, configured(), b"Total: 42", **options)
    assert type(result) is Invoice and result.total == 42.0
    assert len(provider.requests) == 1
    assert provider.bodies[0].get("instructions") == options.get("instructions")
    document = provider.bodies[0]["input"][0]["content"][-1]
    assert document == {"type": "input_text", "text": "Total: 42"}


def test_aliases_defaults_and_nested_models_validate(provider):
    class Item(BaseModel):
        amount: float = Field(gt=0)

    class Receipt(BaseModel):
        items: list[Item]
        currency: str = Field(default="USD", alias="currency_code")

    provider.reply(output(value={"items": [{"amount": 4.2}], "currency_code": "EUR"}))
    result = extract(Receipt, configured(), b"text")
    assert result.items[0].amount == 4.2
    assert result.currency == "EUR"


async def test_caller_owned_client_stays_open():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=output()))
    with httpx.Client(transport=transport) as client:
        assert extract(Invoice, configured(), b"text", client=client).total == 42
        assert not client.is_closed
    async with httpx.AsyncClient(transport=transport) as client:
        assert (await extract_async(Invoice, configured(), b"text", client=client)).total == 42
        assert not client.is_closed


async def test_invalid_output_raises_schema_validation_error(run, provider):
    provider.reply(output(value={"total": "lots"}))
    with pytest.raises(SchemaValidationError) as error:
        await run(Invoice, configured(), b"text")
    message = str(error.value)
    assert isinstance(error.value.__cause__, ValidationError)
    assert "after 1 validation attempt" in message
    assert "- total:" in message
    assert "Next step:" in message
    assert "lots" not in message


async def test_validation_retry_sends_feedback_and_can_recover(run, provider):
    provider.reply(output(value={"total": "private-invalid-value"}), output())
    model = configured(validation_retries=1)
    assert (await run(Invoice, model, b"text", "Use EUR.")).total == 42
    assert len(provider.requests) == 2
    feedback = provider.bodies[1]["instructions"]
    assert feedback.startswith("Use EUR.")
    assert "total" in feedback
    assert "private-invalid-value" not in feedback


async def test_validation_diagnostics_are_bounded_and_name_nested_fields(run, provider):
    class Item(BaseModel):
        quantity: int

    class Receipt(BaseModel):
        items: list[Item]

    provider.reply(output(value={"items": [{"quantity": "private-invalid-value"}] * 7}))
    with pytest.raises(SchemaValidationError) as error:
        await run(Receipt, configured(), b"Receipt")
    message = str(error.value)
    assert "after 1 validation attempt" in message
    assert "items.0.quantity: Input should be a valid integer" in message
    assert "items.4.quantity" in message and "items.5.quantity" not in message
    assert "2 more validation errors omitted" in message
    assert "private-invalid-value" not in message
    assert error.value.__cause__.error_count() == 7


def test_validator_messages_are_not_echoed_in_diagnostics_or_feedback(provider):
    class Receipt(BaseModel):
        account: str

        @field_validator("account")
        @classmethod
        def reject_account(cls, value):
            raise ValueError(f"Rejected private account: {value}")

    provider.reply(output(value={"account": "private-account-number"}))
    with pytest.raises(SchemaValidationError) as error:
        extract(Receipt, configured(validation_retries=1), b"Receipt")
    message = str(error.value)
    assert "account: Value did not satisfy a schema validator" in message
    assert "after 2 validation attempts" in message
    assert "private-account-number" not in message
    assert "private-account-number" not in provider.bodies[1]["instructions"]


def test_model_validator_diagnostics_name_the_root(provider):
    class Receipt(BaseModel):
        total: float

        @model_validator(mode="after")
        def reject_total(self):
            assert self.total < 0, f"Invalid private total: {self.total}"
            return self

    with pytest.raises(SchemaValidationError) as error:
        extract(Receipt, configured(), b"Receipt")
    assert "<root>: Value did not satisfy a schema assertion" in str(error.value)
    assert "42" not in str(error.value)


@pytest.mark.parametrize(
    "payload", [None, [], "text", {"output": []}, {"status": "completed", "output": [None]}]
)
async def test_malformed_response_raises_schema_validation_error(run, provider, payload):
    provider.reply(payload)
    with pytest.raises(SchemaValidationError):
        await run(Invoice, configured(), b"text")


async def test_http_failure_raises_model_error_without_the_body(run, provider):
    provider.reply(lambda request: httpx.Response(503, text="private server details"))
    with pytest.raises(ModelError, match="HTTP 503") as error:
        await run(Invoice, configured(), b"text")
    assert isinstance(error.value.__cause__, httpx.HTTPStatusError)
    assert "private" not in str(error.value)
