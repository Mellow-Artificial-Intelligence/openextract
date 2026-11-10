"""Readable validation diagnostics without dumping model output."""

import json

from pydantic import ValidationError

_MAX_DETAILS = 5
_PRIVATE_MESSAGES = {
    "value_error": "Value did not satisfy a schema validator.",
    "assertion_error": "Value did not satisfy a schema assertion.",
    "union_tag_invalid": "Value does not match an allowed discriminator tag.",
}


def validation_details(error: ValidationError) -> str:
    errors = error.errors(include_input=False, include_url=False, include_context=False)
    lines = []
    for detail in errors[:_MAX_DETAILS]:
        path = ".".join(str(part) for part in detail["loc"]) or "<root>"
        # Validator messages and invalid union tags can embed raw input values.
        message = _PRIVATE_MESSAGES.get(detail["type"], detail["msg"])
        lines.append(f"- {path}: {message} ({detail['type']})")
    if len(errors) > _MAX_DETAILS:
        lines.append(f"- {len(errors) - _MAX_DETAILS} more validation errors omitted.")
    return "\n".join(lines)


def validation_message(error: ValidationError, attempts: int) -> str:
    unit = "attempt" if attempts == 1 else "attempts"
    return (
        f"The model output did not match the schema after {attempts} validation {unit}.\n"
        f"{validation_details(error)}\n"
        "Next step: Check the listed field types, constraints, and validators. "
        "Clarify field descriptions or extraction instructions before retrying."
    )


def invalid_response_message(error: Exception, *, context: str = "model response") -> str:
    if isinstance(error, json.JSONDecodeError):
        detail = "The provider response body was not valid JSON."
    elif isinstance(error, (KeyError, IndexError, TypeError, AttributeError)):
        detail = "The provider returned an unexpected response structure."
    else:
        detail = str(error)
    return (
        f"Invalid {context}: {detail}\n"
        "Next step: Check that Model(api=...) and Model(base_url=...) match the provider. "
        "Use a model that supports the requested schema and input, and check output token limits."
    )
