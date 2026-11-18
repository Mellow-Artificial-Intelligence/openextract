"""Models and provider responses shared by the offline contract tests."""

import json

from pydantic import BaseModel

from openextract import Model


class Invoice(BaseModel):
    total: float


def configured(name="openai:test", **options):
    """A model with a key and no retries, so each test sees exactly one request."""
    return Model(
        name, **{"api_key": "test-key", "max_retries": 0, "validation_retries": 0} | options
    )


def custom(api, **options):
    """A model on a custom endpoint that speaks ``api``."""
    return configured("custom:test", api=api, base_url="https://example.com", **options)


def output(api="responses", value=None):
    """A successful response from ``api`` whose structured output is ``value``."""
    value = {"total": 42.0} if value is None else value
    text = json.dumps(value)
    call = {"type": "function", "function": {"name": "extract", "arguments": text}}
    tool_use = {"name": "extract", "input": value}
    return {
        "responses": {
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
        },
        "chat": {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [call]}}]},
        "chat-json": {"choices": [{"finish_reason": "stop", "message": {"content": text}}]},
        "anthropic": {"stop_reason": "tool_use", "content": [{"type": "tool_use", **tool_use}]},
        "anthropic-json": {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}]},
        "gemini": {
            "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": text}]}}]
        },
        "ollama": {"done": True, "done_reason": "stop", "message": {"content": text}},
        "cohere": {
            "finish_reason": "COMPLETE",
            "message": {"content": [{"type": "text", "text": text}]},
        },
        "bedrock": {
            "stopReason": "tool_use",
            "output": {"message": {"content": [{"toolUse": tool_use}]}},
        },
        "mistral-ocr": {"document_annotation": text},
    }[api]
