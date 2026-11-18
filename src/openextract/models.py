"""Model configuration and provider defaults, without provider SDKs."""

import math
import os
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

import httpx

from .exceptions import ExtractionError

type API = Literal[
    "responses",
    "chat",
    "anthropic",
    "gemini",
    "ollama",
    "cohere",
    "bedrock",
    "mistral-ocr",
    "gemini-decisions",
    "openai-decisions",
    "openrouter-decisions",
    "openrouter-extractions",
]


@dataclass(frozen=True)
class Model:
    """Configure a model, optionally overriding its HTTP API and connection settings.

    ``base_url`` includes the API version (for example, ``https://host/v1``).
    ``settings`` contains provider-native generation parameters. Request structure
    and structured-output fields are managed by openextract and cannot be overridden.
    """

    name: str
    api: API | None = None
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    settings: Mapping[str, Any] = field(default_factory=dict)
    timeout: float = 60.0
    max_retries: int = 2
    validation_retries: int = 1
    output_mode: Literal["tool", "json_schema"] | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be a positive, finite number of seconds.")
        for name in ("max_retries", "validation_retries"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")


@dataclass(frozen=True)
class ResolvedModel:
    """A model configuration combined with its provider's defaults and the environment."""

    config: Model
    model_name: str
    api: API
    base_url: str
    headers: httpx.Headers = field(repr=False)


type _Provider = tuple[API, str, str | None, str]

_APIS = frozenset(get_args(API.__value__))
_OPENAI = ("https://api.openai.com/v1", "OPENAI_API_KEY", "OPENAI_BASE_URL")
_GOOGLE = ("https://generativelanguage.googleapis.com/v1beta", "GOOGLE_API_KEY", "GOOGLE_BASE_URL")
_OPENROUTER = ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL")
# Model prefix: default API, base URL, API key variable, and base URL variable.
_PROVIDERS: dict[str, _Provider] = {
    "openai": ("responses", *_OPENAI),
    "openai-chat": ("chat", *_OPENAI),
    "openai-decisions": ("openai-decisions", *_OPENAI),
    "anthropic": (
        "anthropic",
        "https://api.anthropic.com/v1",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
    ),
    "google": ("gemini", *_GOOGLE),
    "gemini-decisions": ("gemini-decisions", *_GOOGLE),
    "openrouter": ("chat", *_OPENROUTER),
    "openrouter-extractions": ("openrouter-extractions", *_OPENROUTER),
    "ollama": ("ollama", "http://localhost:11434", None, "OLLAMA_BASE_URL"),
    "groq": ("chat", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "GROQ_BASE_URL"),
    "mistral": ("chat", "https://api.mistral.ai/v1", "MISTRAL_API_KEY", "MISTRAL_BASE_URL"),
    "xai": ("chat", "https://api.x.ai/v1", "XAI_API_KEY", "XAI_BASE_URL"),
    "cerebras": ("chat", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", "CEREBRAS_BASE_URL"),
    "huggingface": ("chat", "https://router.huggingface.co/v1", "HF_TOKEN", "HF_BASE_URL"),
    "cohere": ("cohere", "https://api.cohere.com/v2", "CO_API_KEY", "COHERE_BASE_URL"),
    "bedrock": ("bedrock", "", "AWS_BEARER_TOKEN_BEDROCK", "BEDROCK_BASE_URL"),
}
_ALIASES = {"openai-responses": "openai", "google-gla": "google", "gemini": "google"}
_OPENROUTER_EXTRACTION_MODELS = {
    "cloudflare/clairic",
    "inception/mercury-decide",
    "levanto/sage-2",
    "scaledown/extract",
}
_KEY_HEADERS = {
    "anthropic": "x-api-key",
    "gemini": "x-goog-api-key",
    "gemini-decisions": "x-goog-api-key",
}


def resolve(model: str | Model) -> ResolvedModel:
    """Fill in a model's API, endpoint, and credentials from its provider prefix."""
    if isinstance(model, str):
        model = Model(model)
    elif not isinstance(model, Model):
        raise TypeError("model must be a provider:model string or an openextract.Model.")
    prefix, _, name = model.name.partition(":")
    if not prefix or not name:
        raise ValueError("Use a model identifier in the form 'provider:model'.")
    provider = _PROVIDERS.get(_ALIASES.get(prefix, prefix))
    if provider is None:
        if model.api is None or model.base_url is None:
            raise ValueError(
                f"Unknown provider {prefix!r}; configure Model(api=..., base_url=...)."
            )
        api = _resolve_api(model, prefix, name, model.api)
        return ResolvedModel(
            model, name, api, _checked_url(model.base_url), _headers(model, api, None)
        )
    default_api, default_url, key_variable, url_variable = provider
    api = _resolve_api(model, prefix, name, default_api)
    base_url = model.base_url
    if base_url is None:
        base_url = _default_url(api, os.environ.get(url_variable) or default_url)
    return ResolvedModel(
        model, name, api, _checked_url(base_url), _headers(model, api, key_variable)
    )


def merge_settings(
    target: dict[str, Any],
    settings: Mapping[str, Any],
    reserved: AbstractSet[str],
    kind: str = "request fields",
) -> None:
    """Copy caller generation settings into ``target``, rejecting managed keys."""
    if overridden := reserved & settings.keys():
        raise ValueError(f"settings cannot override {kind}: {sorted(overridden)}.")
    target.update(settings)


def _resolve_api(model: Model, prefix: str, name: str, default: API) -> API:
    if model.api is not None:
        api = model.api
    elif prefix == "openrouter" and name in _OPENROUTER_EXTRACTION_MODELS:
        api = "openrouter-extractions"
    elif prefix == "openrouter" and name.startswith(("typesafe/", "~typesafe/")):
        api = "openrouter-decisions"
    else:
        api = default
    if api not in _APIS:
        raise ValueError(f"Unsupported HTTP API: {api!r}.")
    if model.output_mode not in (None, "tool", "json_schema"):
        raise ValueError("output_mode must be 'tool' or 'json_schema'.")
    if model.output_mode is not None and api not in ("chat", "anthropic"):
        raise ValueError(
            "output_mode is only configurable for Chat Completions and Anthropic Messages."
        )
    return api


def _default_url(api: API, base_url: str) -> str:
    if api in ("openrouter-decisions", "openrouter-extractions"):
        # OpenRouter serves its alpha APIs beside /v1.
        return base_url.rstrip("/").removesuffix("/v1").removesuffix("/alpha") + "/alpha"
    if api == "bedrock" and not base_url:
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        if not region:
            raise ExtractionError("Set AWS_REGION or configure Model(base_url=...) for Bedrock.")
        return f"https://bedrock-runtime.{region}.amazonaws.com"
    return base_url


def _checked_url(base_url: str) -> str:
    try:
        url = httpx.URL(base_url)
    except httpx.InvalidURL as exc:
        raise ValueError("base_url must be a valid HTTP(S) URL.") from exc
    is_http = url.scheme in ("http", "https") and url.host
    if not is_http or url.userinfo or url.query or url.fragment:
        raise ValueError("base_url must be an HTTP(S) URL without credentials, query, or fragment.")
    return base_url.rstrip("/")


def _headers(model: Model, api: API, key_variable: str | None) -> httpx.Headers:
    headers = httpx.Headers(model.headers)
    key_header = _KEY_HEADERS.get(api, "authorization")
    key = model.api_key
    if key is None and key_variable:
        key = os.environ.get(key_variable)
        if not key and key_variable == "GOOGLE_API_KEY":
            key = os.environ.get("GEMINI_API_KEY")
        if not key and key_header not in headers:
            raise ExtractionError(
                f"Set {key_variable} or configure Model(api_key=..., headers=...)."
            )
    if key and key_header not in headers:
        headers[key_header] = f"Bearer {key}" if key_header == "authorization" else key
    if api == "anthropic":
        headers.setdefault("anthropic-version", "2023-06-01")
    return headers
