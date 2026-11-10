"""HTTP fixtures shared by extraction contract tests."""

import json

import httpx

from openextract import Model


def configured(name="openai:test", **kwargs):
    return Model(name, api_key="test-key", max_retries=0, validation_retries=0, **kwargs)


def output(api="responses", value=None):
    value = {"total": 42.0} if value is None else value
    text = json.dumps(value)
    if api == "responses":
        return {
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
        }
    if api == "chat":
        return {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "tool_calls": [
                            {"type": "function", "function": {"name": "extract", "arguments": text}}
                        ]
                    },
                }
            ]
        }
    if api == "anthropic":
        return {
            "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "name": "extract", "input": value}],
        }
    if api == "gemini":
        return {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": text}]}}]}
    if api == "ollama":
        return {"done": True, "done_reason": "stop", "message": {"content": text}}
    if api == "cohere":
        return {
            "finish_reason": "COMPLETE",
            "message": {"content": [{"type": "text", "text": text}]},
        }
    if api == "bedrock":
        return {
            "stopReason": "tool_use",
            "output": {"message": {"content": [{"toolUse": {"name": "extract", "input": value}}]}},
        }
    raise ValueError(api)


def serve(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    real_async_client = httpx.AsyncClient

    def sync_client(**kwargs):
        kwargs.setdefault("transport", transport)
        return real_client(**kwargs)

    def async_client(**kwargs):
        kwargs.setdefault("transport", transport)
        return real_async_client(**kwargs)

    def get(url, **kwargs):
        with sync_client(**kwargs) as client:
            return client.get(url)

    monkeypatch.setattr(httpx, "Client", sync_client)
    monkeypatch.setattr(httpx, "AsyncClient", async_client)
    monkeypatch.setattr(httpx, "get", get)
