"""Model resolution: provider defaults, the environment, and explicit configuration."""

import httpx
import pytest

from openextract import ExtractionError, Model, extract
from openextract.models import resolve
from tests.helpers import configured, custom, output
from tests.test_extract import Invoice

OPENAI = "https://api.openai.com/v1"
GOOGLE = "https://generativelanguage.googleapis.com/v1beta"
OPENROUTER = "https://openrouter.ai/api/v1"
OPENROUTER_ALPHA = "https://openrouter.ai/api/alpha"
# Anthropic and Gemini take the bare key in their own header; the rest take a bearer token.
KEY_HEADERS = {"anthropic": ("x-api-key", "env-key"), "gemini": ("x-goog-api-key", "env-key")}


@pytest.mark.parametrize(
    "name,api,url,key_variable",
    [
        ("openai:x", "responses", OPENAI, "OPENAI_API_KEY"),
        ("openai-responses:x", "responses", OPENAI, "OPENAI_API_KEY"),
        ("openai-chat:x", "chat", OPENAI, "OPENAI_API_KEY"),
        ("openai-decisions:x", "openai-decisions", OPENAI, "OPENAI_API_KEY"),
        ("anthropic:x", "anthropic", "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
        ("google:x", "gemini", GOOGLE, "GOOGLE_API_KEY"),
        ("gemini:x", "gemini", GOOGLE, "GEMINI_API_KEY"),
        ("groq:x", "chat", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
        ("mistral:x", "chat", "https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
        ("xai:x", "chat", "https://api.x.ai/v1", "XAI_API_KEY"),
        ("cerebras:x", "chat", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
        ("huggingface:x", "chat", "https://router.huggingface.co/v1", "HF_TOKEN"),
        ("cohere:x", "cohere", "https://api.cohere.com/v2", "CO_API_KEY"),
    ],
)
def test_provider_defaults(monkeypatch, name, api, url, key_variable):
    monkeypatch.setenv(key_variable, "env-key")
    resolved = resolve(name)
    key_header, credential = KEY_HEADERS.get(api, ("authorization", "Bearer env-key"))
    assert (resolved.api, resolved.base_url) == (api, url)
    assert resolved.headers[key_header] == credential


@pytest.mark.parametrize(
    "name,api,url",
    [
        ("openrouter:x", "chat", OPENROUTER),
        ("openrouter:typesafe/jev-1.14", "openrouter-decisions", OPENROUTER_ALPHA),
        ("openrouter:~typesafe/jev-latest", "openrouter-decisions", OPENROUTER_ALPHA),
        ("openrouter:cloudflare/clairic", "openrouter-extractions", OPENROUTER_ALPHA),
        ("openrouter:inception/mercury-decide", "openrouter-extractions", OPENROUTER_ALPHA),
        ("openrouter:levanto/sage-2", "openrouter-extractions", OPENROUTER_ALPHA),
        ("openrouter:scaledown/extract", "openrouter-extractions", OPENROUTER_ALPHA),
        ("openrouter-extractions:custom/preview", "openrouter-extractions", OPENROUTER_ALPHA),
    ],
)
def test_openrouter_routes_by_model_name(monkeypatch, name, api, url):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    resolved = resolve(name)
    assert (resolved.api, resolved.base_url) == (api, url)
    assert resolved.model_name == name.partition(":")[2]
    assert resolved.headers["authorization"] == "Bearer env-key"


@pytest.mark.parametrize(
    "name,api,url",
    [
        ("openrouter:typesafe/model", "chat", OPENROUTER),
        ("openrouter:inception/mercury-decide", "openrouter-decisions", OPENROUTER_ALPHA),
        ("openrouter:custom/preview", "openrouter-extractions", OPENROUTER_ALPHA),
    ],
)
def test_explicit_api_overrides_openrouter_routing(name, api, url):
    resolved = resolve(configured(name, api=api))
    assert (resolved.api, resolved.base_url) == (api, url)


@pytest.mark.parametrize(
    "base_url", ["https://proxy.example/api/v1", "https://proxy.example/api/alpha/"]
)
def test_openrouter_alpha_apis_translate_the_environment_base_url(monkeypatch, base_url):
    monkeypatch.setenv("OPENROUTER_BASE_URL", base_url)
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    assert resolve("openrouter:cloudflare/clairic").base_url == "https://proxy.example/api/alpha"
    assert resolve("openrouter:x").base_url == base_url.rstrip("/")


@pytest.mark.parametrize("region_variable", ["AWS_REGION", "AWS_DEFAULT_REGION"])
def test_bedrock_region_and_key(monkeypatch, region_variable):
    monkeypatch.setenv(region_variable, "us-west-2")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-key")
    resolved = resolve("bedrock:us.anthropic.model")
    assert resolved.base_url == "https://bedrock-runtime.us-west-2.amazonaws.com"
    assert resolved.headers["authorization"] == "Bearer bedrock-key"


def test_environment_overrides_and_explicit_precedence(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1/")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    assert resolve("openai:test").base_url == "https://proxy.example/v1"
    model = Model("openai:test", base_url="https://explicit.example/v1/", api_key="explicit-key")
    resolved = resolve(model)
    assert resolved.base_url == "https://explicit.example/v1"
    assert resolved.headers["authorization"] == "Bearer explicit-key"


def test_custom_headers_take_precedence_and_secrets_are_not_in_repr():
    headers = {"X-API-Key": "header-secret", "Anthropic-Version": "custom-version"}
    model = Model("anthropic:test", api_key="secret-key", headers=headers)
    resolved = resolve(model)
    assert resolved.headers["x-api-key"] == "header-secret"
    assert resolved.headers["anthropic-version"] == "custom-version"
    assert "secret" not in repr(model)
    assert "secret" not in repr(resolved)


def test_custom_protocol_endpoint_without_auth():
    resolved = resolve(Model("local:model", api="chat", base_url="http://localhost:8000/v1"))
    assert resolved.model_name == "model"
    assert "authorization" not in resolved.headers


def test_caller_supplied_authentication(provider):
    class Signer(httpx.Auth):
        def auth_flow(self, request):
            request.headers["Authorization"] = "custom-signed-request"
            yield request

    provider.reply(output("bedrock"))
    model = Model("bedrock:model", base_url="https://example.com", api_key="")
    with httpx.Client(auth=Signer()) as client:
        assert extract(Invoice, model, b"text", client=client).total == 42
    assert provider.requests[0].headers["authorization"] == "custom-signed-request"


def test_custom_chat_endpoint_and_settings(provider):
    provider.reply(output("chat"))
    model = Model(
        "custom:tenant/model:latest",
        api="chat",
        base_url="https://example.com/custom/v1",
        headers={"X-Tenant": "tenant"},
        settings={"temperature": 0},
    )
    extract(Invoice, model, b"text")
    assert provider.requests[0].url == "https://example.com/custom/v1/chat/completions"
    assert provider.requests[0].headers["x-tenant"] == "tenant"
    assert provider.bodies[0]["model"] == "tenant/model:latest"


@pytest.mark.parametrize("mode", ["tool", "json_schema"])
def test_chat_output_mode_is_explicit(mode):
    assert resolve(configured("openai-chat:test", output_mode=mode)).config.output_mode == mode


@pytest.mark.parametrize(
    "name", ["openai:test", "google:test", "openai-decisions:test", "openrouter:typesafe/model"]
)
def test_output_mode_is_scoped_to_chat_and_anthropic(name):
    with pytest.raises(ValueError, match="only configurable"):
        resolve(configured(name, output_mode="json_schema"))


@pytest.mark.parametrize(
    "limit",
    [
        {"timeout": 0},
        {"timeout": -1},
        {"timeout": float("nan")},
        {"timeout": float("inf")},
        {"max_retries": -1},
        {"max_retries": 1.5},
        {"max_retries": True},
        {"validation_retries": -1},
    ],
)
def test_invalid_limits(limit):
    with pytest.raises(ValueError):
        Model("openai:test", **limit)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com",
        "https:///",
        "https://user:secret@example.com/v1",
        "https://example.com?token=secret",
        "https://example.com#fragment",
        "https://example.com:invalid",
    ],
)
def test_invalid_endpoint_url(url):
    with pytest.raises(ValueError, match="base_url"):
        resolve(Model("custom:model", api="chat", base_url=url))


@pytest.mark.parametrize(
    "model,error,message",
    [
        ("model", ValueError, "provider:model"),
        ("", ValueError, "provider:model"),
        (":model", ValueError, "provider:model"),
        ("openai:", ValueError, "provider:model"),
        (object(), TypeError, "openextract.Model"),  # Such as a provider SDK's model object.
        ("unknown:model", ValueError, "Unknown provider"),
        (Model("custom:model", api="chat"), ValueError, "Unknown provider"),
        (Model("custom:model", base_url="https://example.com"), ValueError, "Unknown provider"),
        (custom("invalid"), ValueError, "Unsupported HTTP API"),
        (configured("openai-chat:test", output_mode="invalid"), ValueError, "output_mode must"),
        ("openai:test", ExtractionError, "API_KEY"),
        ("anthropic:test", ExtractionError, "API_KEY"),
        ("gemini:test", ExtractionError, "API_KEY"),
        (Model("bedrock:model", api_key="key"), ExtractionError, "AWS_REGION"),
    ],
)
def test_unresolvable_model_is_rejected(model, error, message):
    with pytest.raises(error, match=message):
        resolve(model)
