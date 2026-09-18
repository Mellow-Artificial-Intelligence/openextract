"""Map provider and I/O failures onto the public ``ExtractionError`` hierarchy."""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from pydantic import ValidationError

from .exceptions import ExtractionError, ModelError, SchemaValidationError, UrlFetchError

_TRANSIENT_HTTP_STATUSES = frozenset((408, 409, 425, 429))
_PROVIDER_MODULE_NAMES: tuple[tuple[str, str], ...] = (
    ("openai", "openai"),
    ("anthropic", "anthropic"),
    ("google", "google"),
    ("botocore", "bedrock"),
    ("cohere", "cohere"),
    ("huggingface_hub", "huggingface"),
    ("groq", "groq"),
    ("mistralai", "mistral"),
    ("grpc", "xai"),
)
_TRANSIENT_ERROR_NAMES = (
    "timeout",
    "connection",
    "ratelimit",
    "throttl",
    "resourceexhausted",
    "servererror",
    "internalserver",
    "serviceunavailable",
)
_PERMANENT_ERROR_NAMES = (
    "authentication",
    "permission",
    "forbidden",
    "badrequest",
    "invalidrequest",
    "validation",
    "unprocessable",
    "notfound",
    "accessdenied",
    "tokenlimit",
    "contextwindow",
    "contextlength",
    "maximumcontext",
)
_TOKEN_LIMIT_MARKERS = (
    "token limit",
    "context window",
    "context length",
    "maximum context",
    "prompt is too long",
    "exceeded before any response",
)
_OUTPUT_RETRY_TYPES = frozenset({"UnexpectedModelBehavior", "ToolRetryError"})
_OUTPUT_RETRY_MARKERS = (
    "output retries",
    "return text or include your response in a tool call",
)

# Exact (module, class) signatures for provider/model error bases we classify
# as ``ModelError``. The classifier checks the exception's existing MRO instead
# of importing provider SDKs. ``openai.APIError`` also covers OpenRouter and
# Cerebras since both go through the openai SDK.
_MODEL_ERROR_SIGNATURES = frozenset(
    {
        ("pydantic_ai.exceptions", "ModelAPIError"),
        ("openai", "APIError"),
        ("anthropic", "APIError"),
        ("google.genai.errors", "APIError"),
        ("botocore.exceptions", "ClientError"),
        ("botocore.exceptions", "BotoCoreError"),
        ("cohere.core.api_error", "ApiError"),
        ("huggingface_hub.errors", "HfHubHTTPError"),
        ("groq", "APIError"),
        ("mistralai.client.errors.mistralerror", "MistralError"),
        ("grpc", "RpcError"),  # xAI SDK uses gRPC; pydantic-ai may surface this directly
    }
)


def _is_model_exception(exc: BaseException) -> bool:
    """Return whether an exception inherits from a supported model error base.

    Provider SDKs have already created the exception by the time this runs, so
    their classes are present in its MRO. Comparing exact module/class
    signatures preserves subclass handling without importing unrelated SDKs.
    """
    return any(
        (error_type.__module__, error_type.__name__) in _MODEL_ERROR_SIGNATURES
        for error_type in type(exc).__mro__
    )


def _model_provider(exc: BaseException) -> str | None:
    """Return a stable provider identifier when the exception exposes one."""
    model_name = getattr(exc, "model_name", None)
    if isinstance(model_name, str) and ":" in model_name:
        return model_name.split(":", 1)[0]

    for error_type in type(exc).__mro__:
        module = error_type.__module__
        for prefix, provider in _PROVIDER_MODULE_NAMES:
            if module == prefix or module.startswith(f"{prefix}."):
                return provider
    return None


def _exception_response(exc: BaseException) -> Any:
    """Return the provider response object carried by ``exc``, when it has one."""
    response = getattr(exc, "response", None)
    return getattr(exc, "raw_response", None) if response is None else response


def _response_metadata(response: Any, key: str) -> Any:
    """Read ``key`` out of a botocore-style ``ResponseMetadata`` mapping."""
    if not isinstance(response, Mapping):
        return None
    metadata = response.get("ResponseMetadata")
    return metadata.get(key) if isinstance(metadata, Mapping) else None


def _plain_int(value: object) -> int | None:
    """Return ``value`` when it is a real ``int`` rather than a ``bool``."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _model_status_code(exc: BaseException) -> int | None:
    """Extract an HTTP status code from common provider exception shapes."""
    response = _exception_response(exc)
    candidates = (
        getattr(exc, "status_code", None),
        getattr(response, "status_code", None),
        _response_metadata(response, "HTTPStatusCode"),
        getattr(exc, "code", None),
    )
    for candidate in candidates:
        status_code = _plain_int(candidate)
        if status_code is not None:
            return status_code
    return None


def _parse_retry_after(value: object) -> float | None:
    """Parse Retry-After seconds or an HTTP date into a non-negative duration."""
    if not isinstance(value, str | int | float) or isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(str(value))
        except (TypeError, ValueError, OverflowError):
            return None
        seconds = retry_at.timestamp() - time.time()
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


def _header_mappings(exc: BaseException) -> list[Mapping]:
    """Collect header-like mappings from common provider exception shapes."""
    header_sets: list[Mapping] = []
    headers = getattr(exc, "headers", None)
    if isinstance(headers, Mapping):
        header_sets.append(headers)

    response = _exception_response(exc)
    response_headers = getattr(response, "headers", None)
    if isinstance(response_headers, Mapping):
        header_sets.append(response_headers)
    metadata_headers = _response_metadata(response, "HTTPHeaders")
    if isinstance(metadata_headers, Mapping):
        header_sets.append(metadata_headers)
    return header_sets


def _model_retry_after(exc: BaseException) -> float | None:
    """Extract and parse Retry-After from common provider header containers."""
    direct_value = getattr(exc, "retry_after", None)
    if direct_value is not None:
        parsed_value = _parse_retry_after(direct_value)
        if parsed_value is not None:
            return parsed_value

    for header_set in _header_mappings(exc):
        for key, value in header_set.items():
            if str(key).lower() == "retry-after":
                return _parse_retry_after(value)
    return None


def _model_error_name(exc: BaseException) -> str:
    """Normalize provider class names and structured error codes for policy checks."""
    names = "".join(error_type.__name__ for error_type in type(exc).__mro__).lower()
    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        error = response.get("Error", {})
        if isinstance(error, Mapping):
            names += str(error.get("Code", "")).lower()
    return names.replace("_", "")


def _is_transient_model_exception(exc: BaseException, status_code: int | None) -> bool:
    """Classify only known transient provider failures as retryable."""
    if status_code is not None:
        return status_code in _TRANSIENT_HTTP_STATUSES or 500 <= status_code <= 599

    code = getattr(exc, "code", None)
    if callable(code):
        grpc_name = getattr(code(), "name", "").lower().replace("_", "")
        if grpc_name in {
            "deadlineexceeded",
            "resourceexhausted",
            "aborted",
            "unavailable",
            "internal",
        }:
            return True
        if grpc_name:
            return False

    error_name = _model_error_name(exc)
    if any(name in error_name for name in _PERMANENT_ERROR_NAMES):
        return False
    return any(name in error_name for name in _TRANSIENT_ERROR_NAMES)


def _is_token_limit_error(exc: BaseException) -> bool:
    """True when the provider or pydantic-ai rejected the prompt as too large."""
    message = str(exc).lower()
    return any(marker in message for marker in _TOKEN_LIMIT_MARKERS)


def _is_output_retry_error(exc: BaseException) -> bool:
    """True for empty-tool-call / exhausted pydantic-ai output retries."""
    for current in (exc, exc.__cause__, exc.__context__):
        if current is None:
            continue
        if type(current).__name__ in _OUTPUT_RETRY_TYPES:
            return True
        message = str(current).lower()
        if any(marker in message for marker in _OUTPUT_RETRY_MARKERS):
            return True
    return False


_EXPECTED_TYPES: dict[str, str] = {
    "bool_parsing": "bool",
    "bool_type": "bool",
    "bytes_type": "bytes",
    "dataclass_type": "object",
    "date_from_datetime_parsing": "date",
    "date_type": "date",
    "datetime_from_date_parsing": "datetime",
    "datetime_parsing": "datetime",
    "datetime_type": "datetime",
    "decimal_parsing": "Decimal",
    "decimal_type": "Decimal",
    "dict_type": "dict",
    "float_parsing": "float",
    "float_type": "float",
    "frozenset_type": "frozenset",
    "int_from_float": "int",
    "int_parsing": "int",
    "int_type": "int",
    "list_type": "list",
    "model_type": "object",
    "none_required": "None",
    "set_type": "set",
    "string_type": "str",
    "time_parsing": "time",
    "time_type": "time",
    "timedelta_parsing": "timedelta",
    "timedelta_type": "timedelta",
    "tuple_type": "tuple",
    "url_parsing": "URL",
    "url_type": "URL",
    "uuid_parsing": "UUID",
    "uuid_type": "UUID",
}


def _validation_field_path(loc: tuple[object, ...]) -> str:
    """Render a Pydantic error location as a dotted/indexed path (``lines[0].qty``)."""
    parts: list[str] = []
    for item in loc:
        if isinstance(item, int):
            if parts:
                parts[-1] += f"[{item}]"
            else:
                parts.append(f"[{item}]")
            continue
        parts.append(str(item))
    return ".".join(parts) or "<root>"


def _received_type_name(error: Mapping[str, Any]) -> str:
    """Return a short type name for the value that failed validation."""
    if "input" not in error:
        return "missing"
    value = error["input"]
    return "None" if value is None else type(value).__name__


def _format_validation_issue(error: Mapping[str, Any]) -> tuple[dict[str, str], str]:
    """Return structured detail and a ``field: expected T, got U`` line."""
    path = _validation_field_path(tuple(error.get("loc", ())))
    error_type = str(error.get("type", ""))
    received = _received_type_name(error)
    if error_type == "missing":
        detail = {"field": path, "expected": "required field", "received": "missing"}
        return detail, f"{path}: missing required field"
    if error_type == "extra_forbidden":
        detail = {"field": path, "expected": "absent", "received": "unexpected field"}
        return detail, f"{path}: unexpected field"
    expected = _EXPECTED_TYPES.get(error_type)
    if expected is not None:
        detail = {"field": path, "expected": expected, "received": received}
        return detail, f"{path}: expected {expected}, got {received}"
    message = str(error.get("msg", "invalid value"))
    detail = {"field": path, "expected": message, "received": received}
    return detail, f"{path}: {message}"


def _schema_validation_error(exc: ValidationError) -> SchemaValidationError:
    """Map a Pydantic ``ValidationError`` to a field-path ``SchemaValidationError``."""
    details: list[dict[str, str]] = []
    parts: list[str] = []
    for error in exc.errors(include_url=False):
        detail, part = _format_validation_issue(error)
        details.append(detail)
        parts.append(part)
    prefix = "Model output did not match schema"
    errors = tuple(details)
    if not parts:
        return SchemaValidationError(f"{prefix}: {str(exc).rstrip()}", errors=errors)
    if len(parts) == 1:
        return SchemaValidationError(f"{prefix}: {parts[0]}", errors=errors)
    joined = "; ".join(parts)
    return SchemaValidationError(f"{prefix} ({len(parts)} errors): {joined}", errors=errors)


def _map_exception(exc: BaseException) -> ExtractionError:
    """Translate a low-level exception into the appropriate ExtractionError subclass."""
    if isinstance(exc, httpx.HTTPStatusError):
        return UrlFetchError(f"Failed to fetch URL: {exc.response.status_code}")
    if isinstance(exc, httpx.RequestError):
        return UrlFetchError(f"Failed to fetch URL: {exc}")
    if isinstance(exc, ValidationError):
        return _schema_validation_error(exc)
    if _is_token_limit_error(exc):
        return ModelError(
            f"Model token limit exceeded: {exc}",
            provider=_model_provider(exc),
            status_code=_model_status_code(exc),
            retryable=False,
        )
    if _is_output_retry_error(exc):
        return ModelError(
            f"Model output retry exhausted: {exc}",
            provider=_model_provider(exc),
            status_code=_model_status_code(exc),
            retryable=True,
        )
    if _is_model_exception(exc):
        status_code = _model_status_code(exc)
        return ModelError(
            f"Model API error: {exc}",
            provider=_model_provider(exc),
            status_code=status_code,
            retryable=_is_transient_model_exception(exc, status_code),
            retry_after=_model_retry_after(exc),
        )
    return ExtractionError(f"Extraction failed: {exc}")


@contextmanager
def _extraction_errors() -> Iterator[None]:
    """Map low-level extraction failures to the ``ExtractionError`` hierarchy.

    ``TypeError`` (bad call) and already-mapped ``ExtractionError`` subclasses
    pass through unchanged; everything else is routed through
    :func:`_map_exception`.
    """
    try:
        yield
    except (TypeError, ExtractionError):
        raise
    except Exception as e:
        raise _map_exception(e) from e
