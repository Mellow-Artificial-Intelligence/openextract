import httpx
import pytest

from openextract import _transport


def pytest_addoption(parser):
    parser.addoption(
        "--live", action="store_true", help="Run tests marked live against real provider APIs."
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    disabled = pytest.mark.skip(reason="Live API calls are disabled; pass --live to enable them.")
    for item in items:
        if item.get_closest_marker("live"):
            item.add_marker(disabled)


@pytest.fixture(autouse=True)
def block_unmarked_network_requests(request, monkeypatch):
    if request.node.get_closest_marker("live") and request.config.getoption("--live"):
        return

    def blocked(self, request):
        pytest.fail("Real HTTP requests require a live-marked test and --live.")

    async def blocked_async(self, request):
        pytest.fail("Real HTTP requests require a live-marked test and --live.")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", blocked_async)


@pytest.fixture(autouse=True)
def isolate_credentials_and_retry_delays(request, monkeypatch):
    if request.node.get_closest_marker("live"):
        return
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "OPENROUTER_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "XAI_API_KEY",
        "CEREBRAS_API_KEY",
        "HF_TOKEN",
        "CO_API_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "OPENAI_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "GOOGLE_BASE_URL",
        "OLLAMA_BASE_URL",
        "OPENROUTER_BASE_URL",
        "GROQ_BASE_URL",
        "MISTRAL_BASE_URL",
        "XAI_BASE_URL",
        "CEREBRAS_BASE_URL",
        "HF_BASE_URL",
        "COHERE_BASE_URL",
        "BEDROCK_BASE_URL",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_transport.time, "sleep", lambda seconds: None)

    async def sleep(seconds):
        pass

    monkeypatch.setattr(_transport.asyncio, "sleep", sleep)
