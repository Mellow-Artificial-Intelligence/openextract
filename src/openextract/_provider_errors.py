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
    message = _text(fields.get("message")) or _text(error)
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

    details = [
        str(value) for value in (provider, code, native or finish, message) if value is not None
    ]
    summary = (
        f"Model provider returned HTTP {status_code}"
        if not 200 <= status_code < 300
        else "Model provider failed"
    )
    if details:
        summary += ": " + "; ".join(details)
    return ModelError(
        _redact(summary, secrets),
        status_code=status_code,
        code=_redact(code, secrets) if isinstance(code, str) else code,
        provider=_redact(provider, secrets) if provider else None,
        finish_reason=_redact(finish, secrets) if finish else None,
        native_finish_reason=_redact(native, secrets) if native else None,
    )
