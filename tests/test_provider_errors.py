import json

import httpx
import pytest

from openextract import Model, ModelError, extract, extract_async
from tests.helpers import configured
from tests.test_extract import Invoice


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mode", ["tool", "json_schema"])
async def test_native_provider_failure_with_http_200_surfaces_as_model_error(asynchronous, mode):
    seen = []
    payload = {
        "provider": "Google AI Studio",
        "choices": [
            {
                "finish_reason": "error",
                "native_finish_reason": "MALFORMED_FUNCTION_CALL",
                "message": {"tool_calls": []},
            }
        ],
    }

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=payload)

    model = Model("openrouter:google/gemini-2.5-flash", api_key="test", output_mode=mode)
    transport = httpx.MockTransport(handler)
    with pytest.raises(ModelError, match="MALFORMED_FUNCTION_CALL") as error:
        if asynchronous:
            async with httpx.AsyncClient(transport=transport) as client:
                await extract_async(Invoice, model, b"Invoice", client=client)
        else:
            with httpx.Client(transport=transport) as client:
                extract(Invoice, model, b"Invoice", client=client)
    assert len(seen) == 1
    assert error.value.status_code == 200
    assert error.value.provider == "Google AI Studio"
    assert error.value.finish_reason == "error"
    assert error.value.native_finish_reason == "MALFORMED_FUNCTION_CALL"


@pytest.mark.parametrize("status", [200, 400])
@pytest.mark.parametrize(
    "raw",
    [
        json.dumps({"error": {"message": "Unsupported schema keyword: minItems"}}),
        "Unsupported schema keyword: minItems",
    ],
)
def test_http_and_application_errors_preserve_upstream_message(status, raw):
    payload = {
        "error": {
            "code": 400,
            "message": "Provider returned error",
            "metadata": {"provider_name": "Google", "raw": raw},
        }
    }
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(status, json=payload))
        ) as client,
        pytest.raises(ModelError, match="Unsupported schema keyword: minItems") as error,
    ):
        extract(Invoice, configured(), b"Invoice", client=client)
    assert error.value.code == 400
    assert error.value.provider == "Google"
    assert error.value.status_code == status
    if status == 400:
        assert isinstance(error.value.__cause__, httpx.HTTPStatusError)


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
def test_other_provider_error_shapes(payload, expected):
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
        ) as client,
        pytest.raises(ModelError, match=expected),
    ):
        extract(Invoice, configured(), b"Invoice", client=client)


@pytest.mark.parametrize("status", [200, 401])
def test_error_details_redact_credentials_in_messages_and_fields(status):
    def handler(request):
        secret = request.headers["authorization"].removeprefix("Bearer ")
        payload = {
            "error": {
                "code": secret,
                "message": "Invalid credential " + secret,
                "metadata": {
                    "provider_name": "provider-" + secret,
                    "raw": "upstream rejected private-header",
                },
            }
        }
        return httpx.Response(status, json=payload)

    model = Model(
        "openai:test",
        api_key="private-key",
        headers={"X-Custom-Key": "private-header"},
        max_retries=0,
    )
    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ModelError) as error,
    ):
        extract(Invoice, model, b"Invoice", client=client)
    assert "private-key" not in str(error.value)
    assert "private-header" not in str(error.value)
    assert "[REDACTED]" in str(error.value)
    assert error.value.code == "[REDACTED]"
    assert error.value.provider == "provider-[REDACTED]"


def test_failure_reason_fields_redact_caller_client_credentials():
    payload = {
        "choices": [{"finish_reason": "error", "native_finish_reason": "failure-client-secret"}]
    }
    with (
        httpx.Client(
            headers={"X-API-Key": "client-secret"},
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)),
        ) as client,
        pytest.raises(ModelError) as error,
    ):
        extract(Invoice, configured(), b"Invoice", client=client)
    assert error.value.native_finish_reason == "failure-[REDACTED]"
    assert "client-secret" not in str(error.value)


def test_non_json_http_error_keeps_status_without_body():
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(502, json=["private body"]))
        ) as client,
        pytest.raises(ModelError, match="HTTP 502") as error,
    ):
        extract(Invoice, configured(), b"Invoice", client=client)
    assert error.value.status_code == 502
    assert "private body" not in str(error.value)
