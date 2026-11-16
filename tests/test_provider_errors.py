"""Provider failures become a ModelError with the facts, a next step, and no secrets."""

import httpx
import pytest

from openextract import Model, ModelError, extract
from tests.helpers import configured
from tests.test_extract import Invoice


@pytest.mark.parametrize("mode", ["tool", "json_schema"])
async def test_failure_reported_with_http_200_is_not_retried(run, provider, mode):
    choice = {
        "finish_reason": "error",
        "native_finish_reason": "MALFORMED_FUNCTION_CALL",
        "message": {"tool_calls": []},
    }
    provider.reply({"provider": "Google AI Studio", "choices": [choice]})
    model = Model("openrouter:google/gemini-2.5-flash", api_key="test", output_mode=mode)
    with pytest.raises(ModelError, match="MALFORMED_FUNCTION_CALL") as error:
        await run(Invoice, model, b"Invoice")
    assert len(provider.requests) == 1
    assert error.value.status_code == 200
    assert error.value.provider == "Google AI Studio"
    assert error.value.finish_reason == "error"
    assert error.value.native_finish_reason == "MALFORMED_FUNCTION_CALL"


@pytest.mark.parametrize("status", [200, 400])
@pytest.mark.parametrize(
    "raw",
    [
        '{"error": {"message": "Unsupported schema keyword: minItems"}}',
        "Unsupported schema keyword: minItems",
    ],
)
def test_http_and_application_errors_preserve_the_upstream_message(provider, status, raw):
    metadata = {"provider_name": "Google", "raw": raw}
    rejection = {"code": 400, "message": "Provider returned error", "metadata": metadata}
    provider.reply({"error": rejection}, status=status)
    with pytest.raises(ModelError, match="Unsupported schema keyword: minItems") as error:
        extract(Invoice, configured(), b"Invoice")
    assert error.value.code == 400
    assert error.value.provider == "Google"
    assert error.value.status_code == status
    assert isinstance(error.value.__cause__, httpx.HTTPStatusError) == (status == 400)


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"error": "Model is unavailable"}, "Model is unavailable"),
        (
            {"error": {"type": "invalid_request_error", "message": "Invalid schema"}},
            "invalid_request_error",
        ),
        ({"error": {"code": {"unexpected": "shape"}}}, "Model provider failed"),
        ({"error": {"message": "Error", "metadata": {"raw": "[]"}}}, "Error"),
        (
            {
                "choices": [
                    {
                        "finish_reason": "error",
                        "error": {"message": "Generation failed", "code": "upstream_error"},
                    }
                ]
            },
            "upstream_error",
        ),
        ({"choices": [{"finish_reason": "error"}]}, "error"),
    ],
)
def test_other_provider_error_shapes(provider, payload, expected):
    provider.reply(payload)
    with pytest.raises(ModelError, match=expected):
        extract(Invoice, configured(), b"Invoice")


@pytest.mark.parametrize("status", [200, 401])
def test_credentials_are_redacted_from_the_message_and_fields(provider, status):
    metadata = {"provider_name": "provider-private-key", "raw": "upstream rejected private-header"}
    rejection = {"code": "private-key", "message": "Invalid credential private-key"}
    provider.reply({"error": rejection | {"metadata": metadata}}, status=status)
    model = configured(api_key="private-key", headers={"X-Custom-Key": "private-header"})
    with pytest.raises(ModelError) as error:
        extract(Invoice, model, b"Invoice")
    assert provider.requests[0].headers["authorization"] == "Bearer private-key"
    assert "private-key" not in str(error.value)
    assert "private-header" not in str(error.value)
    assert "[REDACTED]" in str(error.value)
    assert error.value.code == "[REDACTED]"
    assert error.value.provider == "provider-[REDACTED]"


@pytest.mark.parametrize(
    "headers", [{"X-API-Key": "client-secret"}, {"Authorization": "Bearer client-secret"}]
)
def test_failure_reason_redacts_credentials_from_the_callers_client(provider, headers):
    choice = {"finish_reason": "error", "native_finish_reason": "failure-client-secret"}
    provider.reply({"choices": [choice]})
    with httpx.Client(headers=headers) as client, pytest.raises(ModelError) as error:
        extract(Invoice, configured(api_key=""), b"Invoice", client=client)
    assert error.value.native_finish_reason == "failure-[REDACTED]"
    assert "client-secret" not in str(error.value)


def test_http_error_without_an_error_object_keeps_the_status_and_drops_the_body(provider):
    provider.reply(["private body"], status=502)
    with pytest.raises(ModelError, match="HTTP 502") as error:
        extract(Invoice, configured(), b"Invoice")
    assert error.value.status_code == 502
    assert "private body" not in str(error.value)


def test_top_level_message_is_preserved_without_echoing_the_whole_body(provider):
    payload = {"message": "Model access is disabled", "private": "source contents"}
    provider.reply(payload, status=403)
    with pytest.raises(ModelError) as error:
        extract(Invoice, configured(), b"Invoice")
    assert "Details: Model access is disabled" in str(error.value)
    assert "source contents" not in str(error.value)


def test_large_message_cannot_hide_the_next_step_or_inject_extra_lines(provider):
    provider.reply({"error": {"message": "private-key\n" * 1000}}, status=401)
    with pytest.raises(ModelError) as error:
        extract(Invoice, configured(api_key="private-key"), b"Invoice")
    message = str(error.value)
    assert "private-key" not in message
    assert len(message) < 2300
    assert len(message.splitlines()) == 3
    assert message.splitlines()[-1].startswith("Next step:")


@pytest.mark.parametrize(
    "status,code,native,hint",
    [
        (400, "invalid_json_schema", None, "supported JSON Schema subset"),
        (200, None, "MALFORMED_FUNCTION_CALL", "output_mode='json_schema'"),
        (307, None, None, "final API endpoint"),
        (400, None, None, "Model(settings=...)"),
        (401, None, None, "API key or authentication headers"),
        (403, None, None, "permission"),
        (404, None, None, "model identifier"),
        (408, None, None, "provider or gateway timed out"),
        (413, None, None, "document size"),
        (429, None, None, "rate limits, quota, and billing"),
        (500, None, None, "provider's service status"),
        (504, None, None, "provider or gateway timed out"),
    ],
)
def test_error_explains_the_failure_and_the_next_step(provider, status, code, native, hint):
    payload = {
        "error": {"message": "Request was rejected", "code": code},
        "provider": "Example Provider",
        "choices": [{"finish_reason": "error", "native_finish_reason": native}],
    }
    provider.reply(payload, status=status)
    with pytest.raises(ModelError) as error:
        extract(Invoice, configured(), b"Invoice")
    message = str(error.value)
    assert "\nProvider: Example Provider\n" in message
    assert "\nDetails: Request was rejected\n" in message
    assert "\nNext step: " in message
    assert hint in message
    assert error.value.status_code == status
    assert error.value.code == code
    assert error.value.native_finish_reason == native
