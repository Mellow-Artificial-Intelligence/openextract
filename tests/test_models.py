import json

import httpx
import pytest

from openextract import ExtractionError, Model, extract
from openextract.models import resolve
from tests.helpers import output
from tests.test_extract import Invoice


@pytest.mark.parametrize(
    "name,api,url,key_env,key_header",
    [
        ("openai:x", "responses", "https://api.openai.com/v1", "OPENAI_API_KEY", "authorization"),
        (
            "openai-responses:x",
            "responses",
            "https://api.openai.com/v1",
            "OPENAI_API_KEY",
            "authorization",
        ),
        ("openai-chat:x", "chat", "https://api.openai.com/v1", "OPENAI_API_KEY", "authorization"),
        (
            "anthropic:x",
            "anthropic",
            "https://api.anthropic.com/v1",
            "ANTHROPIC_API_KEY",
            "x-api-key",
        ),
        (
            "google:x",
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta",
            "GOOGLE_API_KEY",
            "x-goog-api-key",
        ),
        (
            "gemini:x",
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta",
            "GEMINI_API_KEY",
            "x-goog-api-key",
        ),
        (
            "openrouter:x",
            "chat",
            "https://openrouter.ai/api/v1",
            "OPENROUTER_API_KEY",
            "authorization",
        ),
        (
            "openrouter:typesafe/jev-1.14",
            "openrouter-decisions",
            "https://openrouter.ai/api/alpha",
            "OPENROUTER_API_KEY",
            "authorization",
        ),
        (
            "openrouter:~typesafe/jev-latest",
            "openrouter-decisions",
            "https://openrouter.ai/api/alpha",
            "OPENROUTER_API_KEY",
            "authorization",
        ),
        ("groq:x", "chat", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "authorization"),
        ("mistral:x", "chat", "https://api.mistral.ai/v1", "MISTRAL_API_KEY", "authorization"),
        ("xai:x", "chat", "https://api.x.ai/v1", "XAI_API_KEY", "authorization"),
        ("cerebras:x", "chat", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", "authorization"),
        ("huggingface:x", "chat", "https://router.huggingface.co/v1", "HF_TOKEN", "authorization"),
        ("cohere:x", "cohere", "https://api.cohere.com/v2", "CO_API_KEY", "authorization"),
        (
            "openai-decisions:x",
            "openai-decisions",
            "https://api.openai.com/v1",
            "OPENAI_API_KEY",
            "authorization",
        ),
    ],
)
def test_provider_defaults(monkeypatch, name, api, url, key_env, key_header):
    monkeypatch.setenv(key_env, "env-key")
    result = resolve(name)
    assert result.api == api
    assert result.base_url == url
    assert result.headers[key_header] == (
        "env-key" if api in ("anthropic", "gemini") else "Bearer env-key"
    )


@pytest.mark.parametrize("region_env", ["AWS_REGION", "AWS_DEFAULT_REGION"])
def test_bedrock_region_and_key(monkeypatch, region_env):
    monkeypatch.setenv(region_env, "us-west-2")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-key")
    result = resolve("bedrock:us.anthropic.model")
    assert result.base_url == "https://bedrock-runtime.us-west-2.amazonaws.com"
    assert result.headers["authorization"] == "Bearer bedrock-key"


def test_bedrock_requires_region_or_url():
    with pytest.raises(ExtractionError, match="AWS_REGION"):
        resolve(Model("bedrock:model", api_key="key"))


@pytest.mark.parametrize(
    "kwargs",
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
def test_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        Model("openai:test", **kwargs)


@pytest.mark.parametrize("name", ["model", "", ":model", "openai:"])
def test_invalid_model_identifier(name):
    with pytest.raises(ValueError, match="provider:model"):
        resolve(name)


def test_sdk_model_objects_are_rejected():
    with pytest.raises(TypeError, match="openextract.Model"):
        resolve(object())


@pytest.mark.parametrize(
    "model",
    [
        "unknown:model",
        Model("custom:model", api="chat"),
        Model("custom:model", base_url="https://example.com"),
    ],
)
def test_unknown_provider_requires_explicit_protocol_and_url(model):
    with pytest.raises(ValueError, match="Unknown provider"):
        resolve(model)


def test_unknown_api_is_rejected():
    with pytest.raises(ValueError, match="Unsupported HTTP API"):
        resolve(Model("custom:model", api="invalid", base_url="https://example.com"))


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


@pytest.mark.parametrize("name", ["openai:test", "anthropic:test", "gemini:test"])
def test_missing_provider_key(name):
    with pytest.raises(ExtractionError, match="API_KEY"):
        resolve(name)


def test_environment_overrides_and_explicit_precedence(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1/")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    assert resolve("openai:test").base_url == "https://proxy.example/v1"
    model = Model("openai:test", base_url="https://explicit.example/v1/", api_key="explicit-key")
    resolved = resolve(model)
    assert resolved.base_url == "https://explicit.example/v1"
    assert resolved.headers["authorization"] == "Bearer explicit-key"


def test_custom_headers_take_precedence_and_secrets_are_not_in_repr():
    model = Model(
        "anthropic:test",
        api_key="secret-key",
        headers={"X-API-Key": "header-secret", "Anthropic-Version": "custom-version"},
    )
    resolved = resolve(model)
    assert resolved.headers["x-api-key"] == "header-secret"
    assert resolved.headers["anthropic-version"] == "custom-version"
    assert "secret" not in repr(model)
    assert "secret" not in repr(resolved)


def test_custom_protocol_endpoint_without_auth():
    resolved = resolve(Model("local:model", api="chat", base_url="http://localhost:8000/v1"))
    assert resolved.model_name == "model"
    assert "authorization" not in resolved.headers


def test_caller_supplied_authentication():
    class Signer(httpx.Auth):
        def auth_flow(self, request):
            request.headers["Authorization"] = "custom-signed-request"
            yield request

    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=output("bedrock"))

    with httpx.Client(auth=Signer(), transport=httpx.MockTransport(handler)) as client:
        result = extract(
            Invoice,
            Model("bedrock:model", base_url="https://example.com", api_key=""),
            b"text",
            client=client,
        )
    assert result.total == 42
    assert seen[0].headers["authorization"] == "custom-signed-request"


def test_custom_chat_endpoint_and_settings():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=output("chat"))

    model = Model(
        "custom:tenant/model:latest",
        api="chat",
        base_url="https://example.com/custom/v1",
        headers={"X-Tenant": "tenant"},
        settings={"temperature": 0},
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        extract(Invoice, model, b"text", client=client)
    assert seen[0].url == "https://example.com/custom/v1/chat/completions"
    assert seen[0].headers["x-tenant"] == "tenant"
    assert json.loads(seen[0].content)["model"] == "tenant/model:latest"


@pytest.mark.parametrize(
    "name",
    [
        "openrouter:cloudflare/clairic",
        "openrouter:inception/mercury-decide",
        "openrouter:levanto/sage-2",
        "openrouter:scaledown/extract",
        "openrouter-extractions:custom/preview",
    ],
)
def test_openrouter_extractions_route(monkeypatch, name):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    resolved = resolve(name)
    assert resolved.api == "openrouter-extractions"
    assert resolved.base_url == "https://openrouter.ai/api/alpha"
    assert resolved.model_name == name.split(":", 1)[1]
    assert resolved.headers["authorization"] == "Bearer env-key"


@pytest.mark.parametrize(
    "name,api,base_url",
    [
        ("openrouter:typesafe/model", "chat", "https://openrouter.ai/api/v1"),
        (
            "openrouter:inception/mercury-decide",
            "openrouter-decisions",
            "https://openrouter.ai/api/alpha",
        ),
        ("openrouter:custom/preview", "openrouter-extractions", "https://openrouter.ai/api/alpha"),
    ],
)
def test_openrouter_protocol_can_be_explicitly_overridden(name, api, base_url):
    resolved = resolve(Model(name, api=api, api_key="key"))
    assert resolved.api == api
    assert resolved.base_url == base_url


@pytest.mark.parametrize("mode", ["tool", "json_schema"])
def test_chat_output_mode_is_explicit(mode):
    assert (
        resolve(Model("openai-chat:test", api_key="test", output_mode=mode)).config.output_mode
        == mode
    )


def test_invalid_output_mode():
    with pytest.raises(ValueError, match="output_mode must"):
        resolve(Model("openai-chat:test", api_key="test", output_mode="invalid"))


@pytest.mark.parametrize(
    "name",
    [
        "openai:test",
        "anthropic:test",
        "google:test",
        "openai-decisions:test",
        "openrouter:typesafe/model",
    ],
)
def test_output_mode_is_scoped_to_chat(name):
    with pytest.raises(ValueError, match="only configurable"):
        resolve(Model(name, api_key="test", output_mode="json_schema"))
