"""Model configuration without provider SDKs."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
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
class _Provider:
    api: API
    base_url: str
    key_env: str | None
    url_env: str


_PROVIDERS = {
    "openai": _Provider(
        "responses", "https://api.openai.com/v1", "OPENAI_API_KEY", "OPENAI_BASE_URL"
    ),
    "openai-chat": _Provider(
        "chat", "https://api.openai.com/v1", "OPENAI_API_KEY", "OPENAI_BASE_URL"
    ),
    "anthropic": _Provider(
        "anthropic", "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"
    ),
    "google": _Provider(
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta",
        "GOOGLE_API_KEY",
        "GOOGLE_BASE_URL",
    ),
    "ollama": _Provider("ollama", "http://localhost:11434", None, "OLLAMA_BASE_URL"),
    "openrouter": _Provider(
        "chat", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL"
    ),
    "openrouter-extractions": _Provider(
        "openrouter-extractions",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
    ),
    "groq": _Provider("chat", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "GROQ_BASE_URL"),
    "mistral": _Provider(
        "chat", "https://api.mistral.ai/v1", "MISTRAL_API_KEY", "MISTRAL_BASE_URL"
    ),
    "xai": _Provider("chat", "https://api.x.ai/v1", "XAI_API_KEY", "XAI_BASE_URL"),
    "cerebras": _Provider(
        "chat", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", "CEREBRAS_BASE_URL"
    ),
    "huggingface": _Provider("chat", "https://router.huggingface.co/v1", "HF_TOKEN", "HF_BASE_URL"),
    "cohere": _Provider("cohere", "https://api.cohere.com/v2", "CO_API_KEY", "COHERE_BASE_URL"),
    "bedrock": _Provider("bedrock", "", "AWS_BEARER_TOKEN_BEDROCK", "BEDROCK_BASE_URL"),
    "openai-decisions": _Provider(
        "openai-decisions", "https://api.openai.com/v1", "OPENAI_API_KEY", "OPENAI_BASE_URL"
    ),
}
_ALIASES = {"openai-responses": "openai", "google-gla": "google", "gemini": "google"}
_OPENROUTER_EXTRACTION_MODELS = {
    "cloudflare/clairic",
    "inception/mercury-decide",
    "levanto/sage-2",
    "scaledown/extract",
}


@dataclass(frozen=True)
class ResolvedModel:
    config: Model
    model_name: str
    api: API
    base_url: str
    headers: httpx.Headers = field(repr=False)


def resolve(model: str | Model) -> ResolvedModel:
    if isinstance(model, str):
        model = Model(model)
    if not isinstance(model, Model):
        raise TypeError("model must be a provider:model string or an openextract.Model.")
    prefix, separator, name = model.name.partition(":")
    if not separator or not prefix or not name:
        raise ValueError("Use a model identifier in the form 'provider:model'.")
    provider = _PROVIDERS.get(_ALIASES.get(prefix, prefix))
    if provider is None and (model.api is None or model.base_url is None):
        raise ValueError(f"Unknown provider {prefix!r}; configure Model(api=..., base_url=...).")
    if model.api is not None:
        api = model.api
    elif prefix == "openrouter" and name in _OPENROUTER_EXTRACTION_MODELS:
        api = "openrouter-extractions"
    elif prefix == "openrouter" and name.startswith(("typesafe/", "~typesafe/")):
        api = "openrouter-decisions"
    else:
        assert provider is not None
        api = provider.api
    if api not in get_args(API.__value__):
        raise ValueError(f"Unsupported HTTP API: {api!r}.")
    if model.output_mode is not None:
        if model.output_mode not in ("tool", "json_schema"):
            raise ValueError("output_mode must be 'tool' or 'json_schema'.")
        if api != "chat":
            raise ValueError("output_mode is only configurable for the Chat Completions API.")
    base_url = model.base_url
    if base_url is None:
        assert provider is not None
        base_url = os.environ.get(provider.url_env) or provider.base_url
        if api in ("openrouter-decisions", "openrouter-extractions"):
            base_url = base_url.rstrip("/").removesuffix("/v1").removesuffix("/alpha") + "/alpha"
        if api == "bedrock" and not base_url:
            region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
            if not region:
                raise ExtractionError(
                    "Set AWS_REGION or configure Model(base_url=...) for Bedrock."
                )
            base_url = f"https://bedrock-runtime.{region}.amazonaws.com"
    try:
        url = httpx.URL(base_url)
    except httpx.InvalidURL as exc:
        raise ValueError("base_url must be a valid HTTP(S) URL.") from exc
    if (
        url.scheme not in ("http", "https")
        or not url.host
        or url.userinfo
        or url.query
        or url.fragment
    ):
        raise ValueError("base_url must be an HTTP(S) URL without credentials, query, or fragment.")
    key = model.api_key
    if key is None and provider and provider.key_env:
        key = os.environ.get(provider.key_env)
        if not key and prefix in ("google", "google-gla", "gemini"):
            key = os.environ.get("GEMINI_API_KEY")
    headers = httpx.Headers(model.headers)
    auth_header = {"anthropic": "x-api-key", "gemini": "x-goog-api-key"}.get(api, "authorization")
    if key and auth_header not in headers:
        headers[auth_header] = f"Bearer {key}" if auth_header == "authorization" else key
    if provider and provider.key_env and model.api_key is None and auth_header not in headers:
        raise ExtractionError(
            f"Set {provider.key_env} or configure Model(api_key=..., headers=...)."
        )
    if api == "anthropic":
        headers.setdefault("anthropic-version", "2023-06-01")
    return ResolvedModel(model, name, api, base_url.rstrip("/"), headers)
