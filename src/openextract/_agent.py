"""Pydantic AI agent construction and one-shot run helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, cast

from ._config import _validate_timeout
from ._errors import _extraction_errors
from ._types import T, Usage
from .exceptions import ProviderNotInstalledError

if TYPE_CHECKING:
    from pydantic_ai import Agent as PydanticAgent
    from pydantic_ai.models import Model
    from pydantic_ai.models.instrumented import InstrumentationSettings
    from pydantic_ai.settings import ModelSettings

    def Agent(*args: object, **kwargs: object) -> PydanticAgent: ...
else:

    def Agent(*args, **kwargs):
        """Construct a Pydantic AI agent without loading it at package import time."""
        from pydantic_ai import Agent as PydanticAgent

        return PydanticAgent(*args, **kwargs)


_OPENAI_PREFIX = "openai:"
_OPENAI_RESPONSES_PREFIX = "openai-responses:"

# pydantic-ai model-id prefix -> the openextract optional-dependency extra that
# installs that provider's SDK. OpenAI-compatible providers (Cerebras, Ollama)
# ship through the openai SDK, so they map to the ``openai`` extra. Prefixes not
# listed here fall back to suggesting ``openextract[all]``.
_PROVIDER_EXTRAS: dict[str, str] = {
    "openai": "openai",
    "openai-chat": "openai",
    "openai-responses": "openai",
    "anthropic": "anthropic",
    "google-gla": "google",
    "google-vertex": "google",
    "bedrock": "bedrock",
    "cohere": "cohere",
    "groq": "groq",
    "huggingface": "huggingface",
    "mistral": "mistral",
    "openrouter": "openrouter",
    "xai": "xai",
    "cerebras": "openai",
    "ollama": "openai",
}


def _install_hint(model: str) -> str:
    """Return the ``pip install`` command that provides ``model``'s provider."""
    prefix = model.split(":", 1)[0]
    extra = _PROVIDER_EXTRAS.get(prefix)
    target = f"openextract[{extra}]" if extra else "openextract[all]"
    return f"pip install '{target}'"


def _route_model(model: str) -> str:
    """Route shorthand OpenAI identifiers through the Responses API."""
    if model.startswith(_OPENAI_PREFIX):
        return f"{_OPENAI_RESPONSES_PREFIX}{model.removeprefix(_OPENAI_PREFIX)}"
    return model


def _instrumentation_capabilities(
    instrument: bool | InstrumentationSettings,
) -> tuple[list[object], dict[str, object]]:
    """Return version-compatible Pydantic AI instrumentation arguments."""
    if instrument is False:
        return [], {}

    from pydantic_ai.models.instrumented import InstrumentationSettings

    if instrument is True:
        settings = InstrumentationSettings()
    elif isinstance(instrument, InstrumentationSettings):
        settings = instrument
    else:
        raise TypeError("instrument must be a bool or InstrumentationSettings instance.")

    try:
        from pydantic_ai.capabilities import Instrumentation
    except ImportError:  # pragma: no cover - compatibility with older pydantic-ai
        return [], {"instrument": settings}
    return [Instrumentation(settings)], {}


def _build_agent(
    schema: type[T],
    model: str | Model,
    instructions: str | None,
    *,
    model_settings: ModelSettings | None = None,
    instrument: bool | InstrumentationSettings = False,
    extra_capabilities: Sequence[object] = (),
) -> PydanticAgent:
    """Construct the pydantic_ai Agent, handling the ollama output-type quirk.

    A missing provider SDK surfaces here as ``ImportError`` (pydantic-ai infers
    the model eagerly); translate it into an actionable
    :class:`ProviderNotInstalledError`.
    """
    try:
        output_type = schema
        if isinstance(model, str) and model.startswith("ollama"):
            from pydantic_ai.output import NativeOutput

            output_type = NativeOutput(schema)
        capabilities, compatibility_kwargs = _instrumentation_capabilities(instrument)
        capabilities = [*capabilities, *extra_capabilities]
        _ensure_openai_usage_fallback()
        routed_model = _route_model(model) if isinstance(model, str) else model
        agent_kwargs: dict[str, object] = {
            "instructions": instructions,
            "output_type": output_type,
            "model_settings": _usage_model_settings(model, model_settings),
            **compatibility_kwargs,
        }
        if capabilities:
            agent_kwargs["capabilities"] = capabilities
        return Agent(
            routed_model,
            **agent_kwargs,
        )
    except ImportError as exc:
        if isinstance(model, str):
            message = (
                f"Model {model!r} needs a provider SDK that is not installed. "
                f"Install it with: {_install_hint(model)} "
                f"(or 'pip install openextract[all]'). Original error: {exc}"
            )
        else:
            message = f"The configured model needs a provider SDK that is not installed: {exc}"
        raise ProviderNotInstalledError(message) from exc


def _build_run_inputs(file_bytes: bytes, file_type: str) -> list:
    """Build the prompt inputs passed to the agent run."""
    from pydantic_ai import BinaryContent

    return [
        "Extract the requested information from this document.",
        BinaryContent(data=file_bytes, media_type=file_type),
    ]


def _resolve_run_inputs(
    file_bytes: bytes,
    file_type: str,
    style_inputs: list[str] | None,
) -> list:
    """Use style-specific prompts when present; otherwise pass media directly."""
    if style_inputs is None:
        return _build_run_inputs(file_bytes, file_type)
    return style_inputs


_INPUT_TOKEN_KEYS = (
    "input_tokens",
    "request_tokens",
    "prompt_tokens",
    "inputTokens",
    "promptTokens",
    "native_tokens_prompt",
    "native_tokens_input",
    "tokens_prompt",
)
_OUTPUT_TOKEN_KEYS = (
    "output_tokens",
    "response_tokens",
    "completion_tokens",
    "outputTokens",
    "completionTokens",
    "native_tokens_completion",
    "native_tokens_output",
    "tokens_completion",
)
_TOTAL_TOKEN_KEYS = ("total_tokens", "totalTokens", "native_tokens_total", "tokens_total")
_USAGE_NEST_KEYS = (
    "details",
    "usage",
    "provider_details",
    "provider_response",
    "request_usage",
    "extra",
    "raw",
    "body",
)
_OPENAI_USAGE_PATCHED = False


def _ensure_openai_usage_fallback() -> None:
    """Copy provider ``prompt_tokens`` when genai-prices extract leaves zeros.

    pydantic-ai ``RequestUsage.extract`` drops OpenRouter ``prompt_tokens`` /
    ``completion_tokens`` unless genai-prices recognizes the model. Live
    GLM-5.3-flash then records 0/0/0 even though the provider sent counts.
    """
    global _OPENAI_USAGE_PATCHED
    if _OPENAI_USAGE_PATCHED:
        return
    try:
        from pydantic_ai.models import openai as openai_mod
    except ImportError:
        return
    original = openai_mod._map_usage

    def _map_usage_with_fallback(response, provider, provider_url, model):  # noqa: ANN001
        mapped = original(response, provider, provider_url, model)
        if mapped.input_tokens or mapped.output_tokens:
            return mapped
        found = _usage_from_raw(getattr(response, "usage", None))
        if not (found.input_tokens or found.output_tokens):
            return mapped
        mapped.input_tokens = found.input_tokens
        mapped.output_tokens = found.output_tokens
        return mapped

    openai_mod._map_usage = _map_usage_with_fallback  # ty: ignore[invalid-assignment]
    _OPENAI_USAGE_PATCHED = True


def _usage_model_settings(
    model: str | Model, model_settings: ModelSettings | None
) -> ModelSettings | None:
    """Ask OpenRouter to include native usage so token counts are not zero.

    ``openrouter_usage`` is the pydantic-ai setting. ``extra_body.usage`` is
    the raw OpenRouter request field, used when the model wrapper does not
    translate ``openrouter_usage`` (the live GLM-5.3-flash miss).
    """
    if not (isinstance(model, str) and model.startswith("openrouter")):
        return model_settings
    _ensure_openai_usage_fallback()
    merged = dict(model_settings) if model_settings is not None else {}
    merged.setdefault("openrouter_usage", {"include": True})
    extra = dict(merged["extra_body"]) if isinstance(merged.get("extra_body"), dict) else {}
    extra.setdefault("usage", {"include": True})
    merged["extra_body"] = extra
    return cast("ModelSettings", merged)


def _positive_int(value: object) -> int:
    """Return a real positive token count, or ``0``."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    if isinstance(value, str) and value.isdigit():
        parsed = int(value)
        return parsed if parsed > 0 else 0
    return 0


def _token_count(raw: object, names: tuple[str, ...]) -> int:
    """Return the first positive alias, skipping default-zero placeholders.

    pydantic-ai ``RunUsage.input_tokens`` defaults to ``0``, and its deprecated
    ``request_tokens`` property just aliases that zero. Live OpenRouter counts
    may sit on ``prompt_tokens``, ``native_tokens_*``, or a nested container.
    """
    if isinstance(raw, dict):
        mapping = cast(dict[str, object], raw)
        values = (mapping.get(name) for name in names)
    else:
        values = (getattr(raw, name, None) for name in names)
    for value in values:
        count = _positive_int(value)
        if count:
            return count
    return 0


def _defined_attr(raw: object, key: str) -> object:
    """Return ``raw.key`` only when the attribute is actually defined.

    ``getattr`` on ``unittest.mock.Mock`` auto-creates children. Walking those
    as usage nests is exponential and hung CI under coverage (15m job timeout).
    """
    typ = type(raw)
    if hasattr(typ, key):
        return getattr(raw, key, None)
    namespace = getattr(raw, "__dict__", None)
    if isinstance(namespace, dict) and key in namespace:
        return namespace[key]
    return None


def _nested_usage_container(raw: object, key: str) -> object:
    if isinstance(raw, dict):
        mapping = cast(dict[str, object], raw)
        return mapping.get(key)
    return _defined_attr(raw, key)


def _usage_from_raw(raw: object) -> Usage:
    """Read token counts from a provider or pydantic-ai usage object."""
    return _usage_from_raw_depth(raw, 0)


def _usage_from_raw_depth(raw: object, depth: int) -> Usage:
    if raw is None or depth > 4:
        return Usage(0, 0, 0)
    if isinstance(raw, Usage):
        return raw
    dumped = _maybe_model_dump(raw)
    input_tokens = _token_count(raw, _INPUT_TOKEN_KEYS)
    output_tokens = _token_count(raw, _OUTPUT_TOKEN_KEYS)
    total_tokens = _token_count(raw, _TOTAL_TOKEN_KEYS)
    if input_tokens == 0 and output_tokens == 0 and dumped is not None:
        input_tokens = _token_count(dumped, _INPUT_TOKEN_KEYS)
        output_tokens = _token_count(dumped, _OUTPUT_TOKEN_KEYS)
        total_tokens = _token_count(dumped, _TOTAL_TOKEN_KEYS)
    if input_tokens == 0 and output_tokens == 0:
        for key in _USAGE_NEST_KEYS:
            nested = _nested_usage_container(raw, key)
            if nested is None or nested is raw:
                continue
            found = _usage_from_raw_depth(nested, depth + 1)
            if found.input_tokens or found.output_tokens or found.total_tokens:
                return found
        if dumped is not None:
            found = _usage_from_raw_depth(dumped, depth + 1)
            if found.input_tokens or found.output_tokens or found.total_tokens:
                return found
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return Usage(input_tokens, output_tokens, total_tokens)


def _maybe_model_dump(raw: object) -> dict[str, object] | None:
    if not callable(getattr(type(raw), "model_dump", None)):
        return None
    dumper = raw.model_dump  # ty: ignore[unresolved-attribute]
    try:
        dumped = dumper(exclude_none=True)
    except TypeError:
        try:
            dumped = dumper()
        except TypeError:
            return None
    return dumped if isinstance(dumped, dict) else None


def _maybe_call(value: object) -> object:
    if not callable(value):
        return value
    invoke = cast(Callable[[], object], value)
    try:
        return invoke()
    except TypeError:
        return value


def _extract_usage_object(result: object) -> object:
    """Unwrap ``result.usage`` whether it is a property, method, or mapping."""
    if result is None:
        return None
    usage = getattr(result, "usage", None)
    descriptor = getattr(type(result), "usage", None)
    if usage is not None and descriptor is not None and not callable(descriptor):
        return usage
    if callable(usage):
        return _maybe_call(usage)
    if usage is not None:
        return usage
    return result


def _message_usage_parts(message: object) -> list[object]:
    return [
        getattr(message, "usage", None),
        getattr(message, "provider_details", None),
        getattr(message, "provider_response", None),
    ]


def _usage_candidates(result: object) -> list[object]:
    """Collect usage-bearing objects from a pydantic-ai run result."""
    found: list[object] = [_extract_usage_object(result)]
    found.append(getattr(result, "provider_response", None))
    try:
        response = _maybe_call(getattr(result, "response", None))
    except (ValueError, AttributeError):
        response = None
    if response is not None and response is not result:
        found.append(getattr(response, "usage", None))
        found.append(getattr(response, "provider_details", None))
        found.append(getattr(response, "provider_response", None))
    for getter in ("all_messages", "new_messages"):
        messages = _maybe_call(getattr(result, getter, None))
        if isinstance(messages, list):
            for message in messages:
                found.extend(_message_usage_parts(message))
    return found


def _usage_from_result(result) -> Usage:
    """Build a ``Usage`` from a pydantic-ai run result or provider usage object."""
    for candidate in _usage_candidates(result):
        usage = _usage_from_raw(candidate)
        if usage.input_tokens or usage.output_tokens or usage.total_tokens:
            return usage
    return Usage(0, 0, 0)


def _model_identifier(model: str | Model | None, agent: object) -> str | None:
    """Return a stable model identifier for result diagnostics, when known."""
    if isinstance(model, str):
        return _route_model(model)
    if model is not None:
        name = getattr(model, "model_name", None)
        if isinstance(name, str):
            return name
    agent_model = getattr(agent, "model", None)
    name = getattr(agent_model, "model_name", None)
    return name if isinstance(name, str) else None


def _run_extraction(agent: PydanticAgent, inputs: list):
    """Run a prepared sync extraction and return the raw pydantic-ai result."""
    with _extraction_errors():
        return agent.run_sync(inputs)


async def _run_extraction_async(agent: PydanticAgent, inputs: list):
    """Run a prepared async extraction with public exception mapping."""
    with _extraction_errors():
        return await agent.run(inputs)


def _session_model_settings(
    model_settings: ModelSettings | None,
    timeout: float | None,
) -> ModelSettings | None:
    settings = dict(model_settings) if model_settings is not None else {}
    if timeout is not None:
        settings["timeout"] = _validate_timeout(timeout, name="timeout")
    return cast("ModelSettings | None", settings or None)
