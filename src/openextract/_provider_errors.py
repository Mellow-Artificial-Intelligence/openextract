"""Extract actionable provider errors without dumping response bodies or credentials."""

import json
from collections.abc import Iterable
from typing import Any

from .exceptions import ModelError


def _object(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _redact(value: str, secrets: Iterable[str]) -> str:
    for secret in sorted(set(secrets), key=len, reverse=True):
        if secret:
            value = value.replace(secret, "[REDACTED]")
    return " ".join(value.split())[:2000]


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
    if status == 401:
        return "Check the API key or authentication headers for the selected provider."
    if status == 403:
        return "Check that the account and credentials have permission to use the selected model."
    if status == 404:
        return "Check the model identifier and Model(base_url=...), including the API version."
    if status in (408, 504):
        return (
            "The provider or gateway timed out. Retry later or reduce "
            "the document size or output token budget."
        )
    if status == 413:
        return "Reduce the source document size or split it into smaller inputs."
    if status == 429:
        return "Wait before retrying; check the provider's rate limits, quota, and billing."
    if status >= 500:
        return "Retry later; check the provider's service status if failures continue."
    return (
        "Check the provider details, model, schema, media type, "
        "and Model(settings=...) before retrying."
    )


def provider_error(
    payload: dict[str, Any], status_code: int, secrets: Iterable[str] = ()
) -> ModelError | None:
    secrets = tuple(secrets)
    choices = payload.get("choices")
    choice = _object(choices[0]) if isinstance(choices, list) and choices else {}
    error = payload.get("error")
    if error is None:
        error = choice.get("error")
    finish = _text(choice.get("finish_reason"))
    native = _text(choice.get("native_finish_reason"))
    if 200 <= status_code < 300 and error is None and finish != "error":
        return None

    fields = _object(error)
    metadata = _object(fields.get("metadata"))
    provider = _text(metadata.get("provider_name")) or _text(payload.get("provider"))
    code = fields.get("code", fields.get("type"))
    if type(code) not in (str, int):
        code = None
    message = _text(fields.get("message")) or _text(error) or _text(payload.get("message"))
    # OpenRouter may place the upstream error in a JSON string under metadata.raw.
    raw = metadata.get("raw")
    if isinstance(raw, str):
        try:
            upstream = _object(json.loads(raw))
        except ValueError:
            detail = raw
        else:
            upstream = _object(upstream.get("error", upstream))
            detail = _text(upstream.get("message"))
        if detail:
            message = f"{message}: {detail}" if message else detail

    summary = (
        f"Model provider returned HTTP {status_code}"
        if not 200 <= status_code < 300
        else "Model provider failed"
    )
    details = [summary + "."]
    for label, value in (
        ("Provider", provider),
        ("Code", code),
        ("Finish reason", finish),
        ("Native finish reason", native),
        ("Details", message),
    ):
        if value is not None:
            details.append(f"{label}: {_redact(str(value), secrets)}")
    details.append("Next step: " + _next_step(status_code, code, native))
    return ModelError(
        "\n".join(details),
        status_code=status_code,
        code=_redact(code, secrets) if isinstance(code, str) else code,
        provider=_redact(provider, secrets) if provider else None,
        finish_reason=_redact(finish, secrets) if finish else None,
        native_finish_reason=_redact(native, secrets) if native else None,
    )
