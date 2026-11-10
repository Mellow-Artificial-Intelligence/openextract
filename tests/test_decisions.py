import json
from enum import Enum
from typing import Literal

import httpx
import pytest
from pydantic import BaseModel, Field

import openextract._extract as extract_module
from openextract import (
    ExtractionError,
    Model,
    ModelError,
    SchemaValidationError,
    extract,
    extract_async,
)
from tests.helpers import serve

PDF_BYTES = b"%PDF-1.7 fake"


class Invoice(BaseModel):
    total: float


class Currency(Enum):
    USD = "usd"
    EUR = "eur"


class Triage(BaseModel):
    document_type: Literal["invoice", "receipt"] = Field(
        description="What kind of document is this?"
    )
    currency: Currency
    is_paid: bool


TRIAGE_ANSWERS = {
    "document_type": {"type": "choice", "choice": "receipt", "confidence": 0.9},
    "currency": {"type": "choice", "choice": "eur", "confidence": 0.8},
    "is_paid": {"type": "noul", "noul": 0.97},
}


def serve_decisions(monkeypatch, seen: list[httpx.Request], answers: dict = TRIAGE_ANSWERS) -> None:
    """Answer decisions requests with ``answers`` and record each request."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"answers": answers})

    serve(monkeypatch, handler)


def test_decisions_model_fills_choice_and_yes_no_fields(monkeypatch):
    seen: list[httpx.Request] = []
    serve_decisions(monkeypatch, seen)

    result = extract(Triage, "openrouter:typesafe/jev-1.14", b"Receipt. Paid in EUR.", "Be strict.")

    assert result == Triage(document_type="receipt", currency=Currency.EUR, is_paid=True)
    assert seen[0].url == "https://openrouter.ai/api/alpha/decisions"
    assert seen[0].headers["authorization"] == "Bearer sk-or-test"
    body = json.loads(seen[0].content)
    assert body["model"] == "typesafe/jev-1.14"
    assert body["state"] == [extract_module._PROMPT, "Be strict.", "Receipt. Paid in EUR."]
    assert body["questions"] == {
        "document_type": {
            "type": "choice",
            "instructions": "What kind of document is this?",
            "criteria": {"invoice": "invoice", "receipt": "receipt"},
        },
        "currency": {
            "type": "choice",
            "instructions": "currency",
            "criteria": {"usd": "usd", "eur": "eur"},
        },
        "is_paid": {"type": "noul", "instructions": "is paid"},
    }


async def test_decisions_model_reads_images(monkeypatch):
    seen: list[httpx.Request] = []
    serve_decisions(monkeypatch, seen)

    await extract_async(Triage, "openrouter:~typesafe/jev-latest", b"\x89PNG....")

    body = json.loads(seen[0].content)
    assert body["model"] == "~typesafe/jev-latest"
    assert body["state"][-1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,iVBORy4uLi4="},
    }


def test_decisions_model_rejects_free_form_fields(monkeypatch):
    serve_decisions(monkeypatch, [])

    with pytest.raises(TypeError, match="'total'"):
        extract(Invoice, "openrouter:typesafe/jev-1.14", b"Total: 42")


def test_decisions_model_rejects_unsupported_media(monkeypatch):
    serve_decisions(monkeypatch, [])

    with pytest.raises(ValueError, match="application/pdf"):
        extract(Triage, "openrouter:typesafe/jev-1.14", PDF_BYTES)


def test_decisions_model_requires_api_key(monkeypatch):
    serve_decisions(monkeypatch, [])
    monkeypatch.delenv("OPENROUTER_API_KEY")

    with pytest.raises(ExtractionError, match="OPENROUTER_API_KEY"):
        extract(Triage, "openrouter:typesafe/jev-1.14", b"text")


def test_decisions_error_response_raises_model_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    serve(
        monkeypatch,
        lambda request: httpx.Response(400, json={"error": {"message": "no such model"}}),
    )

    with pytest.raises(ModelError, match="HTTP 400"):
        extract(Triage, "openrouter:typesafe/nope", b"text")


async def test_decisions_network_failure_raises_model_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    serve(monkeypatch, handler)

    with pytest.raises(ModelError, match="ConnectError"):
        await extract_async(Triage, "openrouter:typesafe/jev-1.14", b"text")


def test_decisions_missing_answer_raises_schema_validation_error(monkeypatch):
    serve_decisions(monkeypatch, [], answers={"is_paid": {"type": "noul", "noul": 0.1}})

    with pytest.raises(SchemaValidationError, match="schema"):
        extract(Triage, "openrouter:typesafe/jev-1.14", b"text")


OPENAI_TRIAGE_ANSWERS = [
    {"type": "choice", "name": "currency", "choice": "eur"},
    {"type": "predicate", "name": "is_paid", "probability": 0.97},
    {"type": "choice", "name": "document_type", "choice": "receipt"},
]


def serve_openai_decisions(monkeypatch, seen, payload=None):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200, json=payload if payload is not None else {"answers": OPENAI_TRIAGE_ANSWERS}
        )

    serve(monkeypatch, handler)


def test_openai_decisions_fills_schema_and_uses_named_answers(monkeypatch):
    seen = []
    serve_openai_decisions(monkeypatch, seen)
    result = extract(Triage, "openai-decisions:gpt-6-luna", b"Paid receipt in EUR.", "Be strict.")

    assert result == Triage(document_type="receipt", currency=Currency.EUR, is_paid=True)
    assert seen[0].url == "https://api.openai.com/v1/decisions"
    assert seen[0].headers["authorization"] == "Bearer sk-test"
    assert json.loads(seen[0].content) == {
        "model": "gpt-6-luna",
        "input": [
            {"role": "developer", "content": "Be strict."},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": extract_module._PROMPT},
                    {"type": "input_text", "text": "Paid receipt in EUR."},
                ],
            },
        ],
        "questions": [
            {
                "name": "document_type",
                "type": "choice",
                "instructions": "What kind of document is this?",
                "choices": [
                    {"value": "invoice", "description": "invoice"},
                    {"value": "receipt", "description": "receipt"},
                ],
            },
            {
                "name": "currency",
                "type": "choice",
                "instructions": "currency",
                "choices": [
                    {"value": "usd", "description": "usd"},
                    {"value": "eur", "description": "eur"},
                ],
            },
            {"name": "is_paid", "type": "predicate", "instructions": "is paid"},
        ],
    }


async def test_openai_decisions_async_reads_images(monkeypatch):
    seen = []
    serve_openai_decisions(monkeypatch, seen)
    result = await extract_async(Triage, "openai-decisions:gpt-6-luna", b"\x89PNG....")
    assert result.is_paid is True
    body = json.loads(seen[0].content)
    assert body["input"][0] == {"role": "developer", "content": extract_module._PROMPT}
    assert body["input"][1]["content"][-1] == {
        "type": "input_image",
        "image_url": "data:image/png;base64,iVBORy4uLi4=",
    }


@pytest.mark.parametrize("probability,expected", [(0.49, False), (0.5, True)])
def test_openai_decisions_boolean_threshold(monkeypatch, probability, expected):
    answers = [dict(answer) for answer in OPENAI_TRIAGE_ANSWERS]
    answers[1]["probability"] = probability
    serve_openai_decisions(monkeypatch, [], {"answers": answers})
    assert extract(Triage, "openai-decisions:gpt-6-luna", b"text").is_paid is expected


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"answers": {}},
        {"answers": []},
        {"answers": OPENAI_TRIAGE_ANSWERS * 2},
        {"answers": [{"type": "refusal", "name": "is_paid"}]},
        {"answers": [{"type": "predicate", "name": "is_paid", "probability": 1.2}]},
        {"answers": [{"type": "predicate", "name": "is_paid", "probability": "yes"}]},
        {"answers": [{"type": "predicate", "name": "is_paid", "probability": True}]},
        {"answers": [{"type": "choice", "name": "is_paid", "probability": 0.5}]},
        {"answers": [{"type": "predicate", "name": "currency", "probability": 0.5}]},
        {"answers": [{"type": "choice", "name": "currency", "choice": "unknown"}]},
        {"answers": [{"type": "choice", "name": "unknown", "choice": "eur"}]},
        {"answers": [None]},
    ],
)
def test_openai_decisions_invalid_answers(monkeypatch, payload):
    serve_openai_decisions(monkeypatch, [], payload)
    with pytest.raises(SchemaValidationError):
        extract(Triage, "openai-decisions:gpt-6-luna", b"text")


def test_openai_decisions_invalid_json(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    serve(monkeypatch, lambda request: httpx.Response(200, content=b"not JSON"))
    with pytest.raises(SchemaValidationError):
        extract(Triage, "openai-decisions:gpt-6-luna", b"text")


def test_openai_decisions_validates_inputs_before_request(monkeypatch):
    seen = []
    serve_openai_decisions(monkeypatch, seen)
    with pytest.raises(TypeError, match="total"):
        extract(Invoice, "openai-decisions:gpt-6-luna", b"text")
    with pytest.raises(ValueError, match="application/pdf"):
        extract(Triage, "openai-decisions:gpt-6-luna", PDF_BYTES)
    with pytest.raises(ValueError, match="1 and 400"):
        extract(BaseModel, "openai-decisions:gpt-6-luna", b"text")
    assert seen == []


def test_openai_decisions_requires_own_api_key(monkeypatch):
    seen = []
    serve_openai_decisions(monkeypatch, seen)
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    with pytest.raises(ExtractionError, match="OPENAI_API_KEY"):
        extract(Triage, "openai-decisions:gpt-6-luna", b"text")
    assert seen == []


@pytest.mark.parametrize(
    "field,answer",
    [
        ("is_paid", {"type": "noul", "noul": 1.2}),
        ("is_paid", {"type": "noul", "noul": True}),
        ("is_paid", {"type": "choice", "noul": 0.5}),
        ("currency", {"type": "noul", "choice": "eur"}),
    ],
)
def test_openrouter_decisions_validates_answer_types(monkeypatch, field, answer):
    answers = {name: dict(value) for name, value in TRIAGE_ANSWERS.items()}
    answers[field] = answer
    serve_decisions(monkeypatch, [], answers)
    with pytest.raises(SchemaValidationError):
        extract(Triage, "openrouter:typesafe/jev-1.14", b"text")


@pytest.mark.parametrize("prefix", ["openai-decisions:gpt-6-luna", "openrouter:typesafe/jev-1.14"])
def test_decisions_reject_settings(monkeypatch, prefix):
    with pytest.raises(ValueError, match="settings"):
        extract(Triage, Model(prefix, api_key="test-key", settings={"temperature": 0}), b"text")


def test_decisions_aliases_are_validated_by_field_name(monkeypatch):
    class Aliased(BaseModel):
        is_paid: bool = Field(alias="paid")

    serve_openai_decisions(
        monkeypatch, [], {"answers": [{"name": "is_paid", "type": "predicate", "probability": 0.9}]}
    )
    assert extract(Aliased, "openai-decisions:gpt-6-luna", b"text").is_paid is True


def test_openai_decisions_provider_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    serve(monkeypatch, lambda request: httpx.Response(429, json={"error": "rate limited"}))
    with pytest.raises(ModelError, match="HTTP 429"):
        extract(Triage, "openai-decisions:gpt-6-luna", b"text")


async def test_openai_decisions_network_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def handler(request):
        raise httpx.ConnectError("offline", request=request)

    serve(monkeypatch, handler)
    with pytest.raises(ModelError, match="ConnectError"):
        await extract_async(Triage, "openai-decisions:gpt-6-luna", b"text")
