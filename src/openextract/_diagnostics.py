"""Actionable error messages that never dump response bodies, model output, or credentials."""

import json
from collections.abc import Iterable
from typing import Any

import httpx
from pydantic import ValidationError

from .exceptions import ModelError

_MAX_VALIDATION_DETAILS = 5
# Validator messages and invalid union tags can embed raw input values.
_PRIVATE_VALIDATION_MESSAGES = {
    "value_error": "Value did not satisfy a schema validator.",
    "assertion_error": "Value did not satisfy a schema assertion.",
    "union_tag_invalid": "Value does not match an allowed discriminator tag.",
}
_TIMED_OUT = (
    "The provider or gateway timed out. Retry later or reduce "
    "the document size or output token budget."
)
_NEXT_STEPS = {
    401: "Check the API key or authentication headers for the selected provider.",
    403: "Check that the account and credentials have permission to use the selected model.",
    404: "Check the model identifier and Model(base_url=...), including the API version.",
    408: _TIMED_OUT,
    413: "Reduce the source document size or split it into smaller inputs.",
    429: "Wait before retrying; check the provider's rate limits, quota, and billing.",
    504: _TIMED_OUT,
}


def validation_details(error: ValidationError) -> str:
    errors = error.errors(include_input=False, include_url=False, include_context=False)
    lines = []
    for detail in errors[:_MAX_VALIDATION_DETAILS]:
        path = ".".join(str(part) for part in detail["loc"]) or "<root>"
        message = _PRIVATE_VALIDATION_MESSAGES.get(detail["type"], detail["msg"])
        lines.append(f"- {path}: {message} ({detail['type']})")
    if len(errors) > _MAX_VALIDATION_DETAILS:
        omitted = len(errors) - _MAX_VALIDATION_DETAILS
        lines.append(f"- {omitted} more validation errors omitted.")
    return "\n".join(lines)


def validation_message(error: ValidationError, attempts: int) -> str:
    return (
        "The model output did not match the schema "
        f"after {_count(attempts, 'validation attempt')}.\n"
        f"{validation_details(error)}\n"
        "Next step: Check the listed field types, constraints, and validators. "
        "Clarify field descriptions or extraction instructions before retrying."
    )


def invalid_response_message(error: Exception, *, context: str = "model response") -> str:
    if isinstance(error, json.JSONDecodeError):
        detail = "The provider response body was not valid JSON."
    elif isinstance(error, (LookupError, TypeError, AttributeError)):
        detail = "The provider returned an unexpected response structure."
    else:
        detail = str(error)
    return (
        f"Invalid {context}: {detail}\n"
        "Next step: Check that Model(api=...) and Model(base_url=...) match the provider. "
        "Use a model that supports the requested schema and input, and check output token limits."
    )


def request_error(error: httpx.RequestError, attempts: int) -> ModelError:
    if isinstance(error, httpx.TimeoutException):
        hint = "Check provider availability or increase Model(timeout=...) for slow requests."
    elif isinstance(error, httpx.LocalProtocolError):
        hint = "Check Model(headers=...) and client authentication for invalid header values."
    elif isinstance(error, httpx.DecodingError):
        hint = "Check the provider or proxy's response encoding and compression settings."
    else:
        hint = (
            "Check Model(base_url=...), network connectivity, proxy settings, and TLS certificates."
        )
    return ModelError(
        f"The model request failed ({type(error).__name__}) "
        f"after {_count(attempts, 'HTTP attempt')}.\nNext step: {hint}"
    )


def provider_error(
    payload: dict[str, Any], status_code: int, secrets: Iterable[str] = ()
) -> ModelError | None:
    """Describe an HTTP failure or an error envelope; ``None`` means the response succeeded."""
    choices = payload.get("choices")
    choice = _object(choices[0]) if isinstance(choices, list) and choices else {}
    error = payload.get("error")
    if error is None:
        error = choice.get("error")
    finish = _text(choice.get("finish_reason"))
    native = _text(choice.get("native_finish_reason"))
    succeeded = 200 <= status_code < 300
    if succeeded and error is None and finish != "error":
        return None

    fields = _object(error)
    metadata = _object(fields.get("metadata"))
    provider = _text(metadata.get("provider_name")) or _text(payload.get("provider"))
    code = fields.get("code", fields.get("type"))
    if type(code) not in (str, int):
        code = None
    message = _text(fields.get("message")) or _text(error) or _text(payload.get("message"))
    if upstream := _upstream_message(metadata.get("raw")):
        message = f"{message}: {upstream}" if message else upstream

    secrets = sorted({secret for secret in secrets if secret}, key=len, reverse=True)
    facts = {
        "Provider": provider,
        "Code": code,
        "Finish reason": finish,
        "Native finish reason": native,
        "Details": message,
    }
    lines = [
        "Model provider failed." if succeeded else f"Model provider returned HTTP {status_code}.",
        *(
            f"{label}: {_redact(str(fact), secrets)}"
            for label, fact in facts.items()
            if fact is not None
        ),
        "Next step: " + _next_step(status_code, code, native),
    ]
    return ModelError(
        "\n".join(lines),
        status_code=status_code,
        code=_redact(code, secrets) if isinstance(code, str) else code,
        provider=_redact(provider, secrets),
        finish_reason=_redact(finish, secrets),
        native_finish_reason=_redact(native, secrets),
    )


def _upstream_message(raw: object) -> str | None:
    """Read the upstream error OpenRouter may place in a JSON string under ``metadata.raw``."""
    if not isinstance(raw, str):
        return None
    try:
        upstream = _object(json.loads(raw))
    except ValueError:
        return raw
    return _text(_object(upstream.get("error", upstream)).get("message"))


def _next_step(status: int, code: str | int | None, native: str | None) -> str:
    if code == "invalid_json_schema":
        return (
            "Simplify the schema to the provider's supported JSON Schema subset "
            "or change output mode."
        )
    if native == "MALFORMED_FUNCTION_CALL":
        return (
            "Simplify the extraction schema or use Model(output_mode='json_schema') "
            "with a Chat Completions model that supports it."
        )
    if 300 <= status < 400:
        return (
            "Set Model(base_url=...) to the final API endpoint; "
            "provider redirects are not followed."
        )
    if status in _NEXT_STEPS:
        return _NEXT_STEPS[status]
    if status >= 500:
        return "Retry later; check the provider's service status if failures continue."
    return (
        "Check the provider details, model, schema, media type, "
        "and Model(settings=...) before retrying."
    )


def _redact(value: str | None, secrets: Iterable[str]) -> str | None:
    """Mask known credentials and collapse the text to a single bounded line."""
    if value is None:
        return None
    for secret in secrets:
        value = value.replace(secret, "[REDACTED]")
    return " ".join(value.split())[:2000]


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _object(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
