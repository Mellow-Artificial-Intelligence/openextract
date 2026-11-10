import httpx
import pytest


def test_offline_tests_block_real_http_requests():
    with (
        httpx.Client(trust_env=False) as client,
        pytest.raises(pytest.fail.Exception, match="Real HTTP requests require"),
    ):
        client.get("https://example.invalid")


async def test_offline_tests_block_real_async_http_requests():
    async with httpx.AsyncClient(trust_env=False) as client:
        with pytest.raises(pytest.fail.Exception, match="Real HTTP requests require"):
            await client.get("https://example.invalid")
