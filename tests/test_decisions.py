"""Decisions APIs fill bool, Literal, and Enum schemas through OpenRouter, OpenAI, and Gemini."""

import base64
from enum import Enum
from typing import Literal

import httpx
import pytest
from pydantic import BaseModel, Field, ValidationError, create_model, field_validator

from openextract import ExtractionError, Model, ModelError, SchemaValidationError, extract
from openextract._content import PROMPT
from tests.helpers import configured

OPENROUTER = "openrouter:typesafe/jev-1.14"
OPENAI = "openai-decisions:gpt-6-luna"
GEMINI = "gemini-decisions:gemini-decision-1"
KEY_VARIABLES = {
    OPENROUTER: "OPENROUTER_API_KEY",
    OPENAI: "OPENAI_API_KEY",
    GEMINI: "GOOGLE_API_KEY",
}
PDF_BYTES = b"%PDF-1.7 fake"
PNG_BYTES = b"\x89PNG...."
PNG_URI = "data:image/png;base64,iVBORy4uLi4="


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


TRIAGE = Triage(document_type="receipt", currency=Currency.EUR, is_paid=True)
# OpenRouter and Gemini share named questions and answers; OpenAI lists them.
NAMED_QUESTIONS = {
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
NAMED_ANSWERS = {
    "document_type": {"type": "choice", "choice": "receipt", "confidence": 0.9},
    "currency": {"type": "choice", "choice": "eur", "confidence": 0.8},
    "is_paid": {"type": "noul", "noul": 0.97},
}
OPENAI_ANSWERS = [
    {"type": "choice", "name": "currency", "choice": "eur"},
    {"type": "predicate", "name": "is_paid", "probability": 0.97},
    {"type": "choice", "name": "document_type", "choice": "receipt"},
]
ANSWERS = {
    OPENROUTER: {"answers": NAMED_ANSWERS},
    OPENAI: {"answers": OPENAI_ANSWERS},
    GEMINI: {"answers": NAMED_ANSWERS},
}


@pytest.fixture(autouse=True)
def provider_keys(monkeypatch):
    for variable in KEY_VARIABLES.values():
        monkeypatch.setenv(variable, "test-key")


def named(**replacements):
    return {"answers": {**NAMED_ANSWERS, **replacements}}


def listed(**answer):
    return {"answers": [answer]}


def test_openrouter_request(provider):
    provider.reply(ANSWERS[OPENROUTER])
    assert extract(Triage, OPENROUTER, "Receipt. Paid in EUR.", "Be strict.") == TRIAGE
    assert provider.requests[0].url == "https://openrouter.ai/api/alpha/decisions"
    assert provider.requests[0].headers["authorization"] == "Bearer test-key"
    assert provider.bodies[0] == {
        "model": "typesafe/jev-1.14",
        "state": [PROMPT, "Be strict.", "Receipt. Paid in EUR."],
        "questions": NAMED_QUESTIONS,
    }


async def test_openrouter_reads_images(run, provider):
    provider.reply(ANSWERS[OPENROUTER])
    await run(Triage, "openrouter:~typesafe/jev-latest", PNG_BYTES)
    assert provider.bodies[0]["model"] == "~typesafe/jev-latest"
    assert provider.bodies[0]["state"] == [
        PROMPT,
        {"type": "image_url", "image_url": {"url": PNG_URI}},
    ]


def test_openai_request(provider):
    provider.reply(ANSWERS[OPENAI])
    assert extract(Triage, OPENAI, "Paid receipt in EUR.", "Be strict.") == TRIAGE
    assert provider.requests[0].url == "https://api.openai.com/v1/decisions"
    assert provider.requests[0].headers["authorization"] == "Bearer test-key"
    choices = {
        name: [{"value": key, "description": key} for key in question["criteria"]]
        for name, question in NAMED_QUESTIONS.items()
        if question["type"] == "choice"
    }
    assert provider.bodies[0] == {
        "model": "gpt-6-luna",
        "input": [
            {"role": "developer", "content": "Be strict."},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT},
                    {"type": "input_text", "text": "Paid receipt in EUR."},
                ],
            },
        ],
        "questions": [
            {
                "name": "document_type",
                "type": "choice",
                "instructions": "What kind of document is this?",
                "choices": choices["document_type"],
            },
            {
                "name": "currency",
                "type": "choice",
                "instructions": "currency",
                "choices": choices["currency"],
            },
            {"name": "is_paid", "type": "predicate", "instructions": "is paid"},
        ],
    }


async def test_openai_reads_images(run, provider):
    provider.reply(ANSWERS[OPENAI])
    assert (await run(Triage, OPENAI, PNG_BYTES)).is_paid is True
    developer, user = provider.bodies[0]["input"]
    assert developer == {"role": "developer", "content": PROMPT}
    assert user["content"][-1] == {"type": "input_image", "image_url": PNG_URI}


@pytest.mark.parametrize(
    "source,media_type,kind",
    [
        ("Paid receipt", "text/plain", "text"),
        (PNG_BYTES, "image/png", "image"),
        (PDF_BYTES, "application/pdf", "document"),
    ],
)
@pytest.mark.parametrize("instructions", [None, "Check carefully."])
async def test_gemini_request(run, provider, monkeypatch, source, media_type, kind, instructions):
    monkeypatch.delenv("GOOGLE_API_KEY")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    provider.reply(ANSWERS[GEMINI])
    assert await run(Triage, GEMINI, source, instructions) == TRIAGE
    request = provider.requests[0]
    assert request.url == "https://generativelanguage.googleapis.com/v1beta/decisions"
    assert request.headers["x-goog-api-key"] == "gemini-key"
    assert "authorization" not in request.headers
    if kind == "text":
        document = {"type": "text", "text": source}
    else:
        data = base64.b64encode(source).decode()
        document = {"type": kind, "mime_type": media_type, "data": data}
    guidance = [PROMPT, instructions] if instructions else [PROMPT]
    assert provider.bodies[0] == {
        "model": "gemini-decision-1",
        "input": [*({"type": "text", "text": text + "\n"} for text in guidance), document],
        "questions": NAMED_QUESTIONS,
    }


def test_gemini_api_can_be_selected_for_a_google_model(provider, monkeypatch):
    monkeypatch.setenv("GOOGLE_BASE_URL", "https://proxy.example/v1beta")
    provider.reply({"error": {"code": "safety", "message": "Blocked"}}, status=400)
    with pytest.raises(ModelError, match="HTTP 400"):
        extract(Triage, Model("google:gemini-decision-1", api="gemini-decisions"), "text")
    assert provider.requests[0].url == "https://proxy.example/v1beta/decisions"
    assert provider.requests[0].headers["x-goog-api-key"] == "test-key"


@pytest.mark.parametrize("model", [OPENROUTER, OPENAI, GEMINI])
def test_inputs_are_validated_before_any_request(provider, monkeypatch, model):
    with pytest.raises(TypeError, match="only fill .* 'total'"):
        extract(Invoice, model, b"Total: 42")
    with pytest.raises(ValueError, match="settings"):
        extract(Triage, configured(model, settings={"temperature": 0}), b"text")
    monkeypatch.delenv(KEY_VARIABLES[model])
    with pytest.raises(ExtractionError, match=KEY_VARIABLES[model]):
        extract(Triage, model, b"text")
    assert provider.requests == []


@pytest.mark.parametrize(
    "model,media_type,message",
    [
        (OPENROUTER, "application/pdf", "text and images, not application/pdf"),
        (OPENAI, "application/pdf", "text and images, not application/pdf"),
        (GEMINI, "audio/wav", "supports text, images and PDF"),
        (GEMINI, "video/mp4", "supports text, images and PDF"),
        (GEMINI, "image/svg+xml", "supports text, images and PDF"),
    ],
)
def test_unsupported_media_is_rejected(model, media_type, message):
    with pytest.raises(ValueError, match=message):
        extract(Triage, model, b"data", media_type=media_type)


@pytest.mark.parametrize(
    "model,fields,message",
    [
        (OPENAI, {}, "1 and 400"),
        (OPENAI, {f"q{index}": (bool, ...) for index in range(401)}, "1 and 400"),
        (GEMINI, {}, "1 and 1000"),
        (GEMINI, {f"q{index}": (bool, ...) for index in range(1001)}, "1 and 1000"),
        (GEMINI, {"é": (bool, ...)}, "field names"),
        (GEMINI, {"a" * 65: (bool, ...)}, "field names"),
        (GEMINI, {"value": (Literal["only"], ...)}, "2 and 255"),
        (GEMINI, {"value": (Literal[tuple(map(str, range(256)))], ...)}, "2 and 255"),
        *(
            (GEMINI, {"value": (Literal[key, "ok"], ...)}, "option key")
            for key in ("", "x" * 129, " space", "space ", "new\nline", "no\x7f", "a b")
        ),
    ],
)
def test_provider_limits_are_enforced(model, fields, message):
    with pytest.raises(ValueError, match=message):
        extract(create_model("Questions", **fields), model, "text")


@pytest.mark.parametrize("model", [OPENROUTER, OPENAI, GEMINI])
async def test_provider_and_network_failures_raise_model_error(run, provider, model):
    def unreachable(request):
        raise httpx.ConnectError("offline", request=request)

    provider.reply({"error": {"message": "rate limited"}}, status=429)
    with pytest.raises(ModelError, match="HTTP 429"):
        await run(Triage, configured(model), b"text")
    provider.reply(unreachable)
    with pytest.raises(ModelError, match="ConnectError"):
        await run(Triage, configured(model), b"text")


@pytest.mark.parametrize("probability,expected", [(0.49, False), (0.5, True)])
def test_yes_means_a_probability_of_at_least_half(provider, probability, expected):
    provider.reply(named(is_paid={"type": "noul", "noul": probability}))
    assert extract(Triage, OPENROUTER, b"text").is_paid is expected
    answers = [dict(answer) for answer in OPENAI_ANSWERS]
    answers[1]["probability"] = probability
    provider.reply({"answers": answers})
    assert extract(Triage, OPENAI, b"text").is_paid is expected


def test_answers_are_validated_by_field_name_not_alias(provider):
    class Aliased(BaseModel):
        is_paid: bool = Field(alias="paid")

    provider.reply(listed(name="is_paid", type="predicate", probability=0.9))
    assert extract(Aliased, OPENAI, b"text").is_paid is True


@pytest.mark.parametrize(
    "model,payload",
    [
        *((model, {}) for model in (OPENROUTER, OPENAI, GEMINI)),
        *(
            (model, payload)
            for model in (OPENROUTER, GEMINI)
            for payload in (
                {"answers": []},
                {"answers": {}},
                {"answers": {"is_paid": {"type": "noul", "noul": 0.1}}},
                named(is_paid={"type": "noul", "noul": 1.2}),
                named(is_paid={"type": "noul", "noul": True}),
                named(is_paid={"type": "choice", "noul": 0.5}),
                named(currency={"type": "noul", "choice": "eur"}),
                named(currency={"type": "choice", "choice": "invalid"}),
                named(unknown={"type": "noul", "noul": 0.5}),
            )
        ),
        (OPENAI, {"answers": {}}),
        (OPENAI, {"answers": []}),
        (OPENAI, {"answers": OPENAI_ANSWERS * 2}),
        (OPENAI, {"answers": OPENAI_ANSWERS[:2]}),
        (OPENAI, {"answers": [None]}),
        (OPENAI, listed(type="refusal", name="is_paid")),
        (OPENAI, listed(type="predicate", name="is_paid", probability=1.2)),
        (OPENAI, listed(type="predicate", name="is_paid", probability="yes")),
        (OPENAI, listed(type="predicate", name="is_paid", probability=True)),
        (OPENAI, listed(type="choice", name="is_paid", probability=0.5)),
        (OPENAI, listed(type="predicate", name="currency", probability=0.5)),
        (OPENAI, listed(type="choice", name="currency", choice="unknown")),
        (OPENAI, listed(type="choice", name="unknown", choice="eur")),
    ],
)
async def test_invalid_answers_raise_schema_validation_error(run, provider, model, payload):
    provider.reply(payload)
    with pytest.raises(SchemaValidationError, match="Invalid decision answers"):
        await run(Triage, model, b"text")
    assert len(provider.requests) == 1


def test_non_json_response_raises_schema_validation_error(provider):
    provider.reply(lambda request: httpx.Response(200, content=b"not JSON"))
    with pytest.raises(SchemaValidationError):
        extract(Triage, OPENAI, b"text")


@pytest.mark.parametrize(
    "model,payload",
    [
        (OPENAI, listed(name="is_paid", type="predicate", probability=0.9)),
        (OPENROUTER, {"answers": {"is_paid": {"type": "noul", "noul": 0.9}}}),
    ],
)
async def test_validator_failures_name_the_field_without_echoing_input(
    run, provider, model, payload
):
    class Decision(BaseModel):
        is_paid: bool

        @field_validator("is_paid")
        @classmethod
        def reject_paid(cls, value):
            raise ValueError(f"Rejected private decision: {value}")

    provider.reply(payload)
    with pytest.raises(SchemaValidationError) as error:
        await run(Decision, model, b"Receipt")
    message = str(error.value)
    assert "is_paid: Value did not satisfy a schema validator" in message
    assert "after 1 validation attempt" in message
    assert "Next step:" in message
    assert "private decision" not in message
    assert isinstance(error.value.__cause__, ValidationError)
    assert len(provider.requests) == 1


@pytest.mark.parametrize("model", [OPENAI, OPENROUTER])
@pytest.mark.parametrize("choice", ["private-invalid-choice", []])
def test_invalid_choice_lists_the_options_without_echoing_the_answer(provider, model, choice):
    class Decision(BaseModel):
        currency: Literal["USD", "EUR"]

    answer = {"type": "choice", "choice": choice}
    named_format = model == OPENROUTER
    provider.reply(
        {"answers": {"currency": answer}} if named_format else listed(name="currency", **answer)
    )
    with pytest.raises(SchemaValidationError) as error:
        extract(Decision, model, b"Receipt")
    message = str(error.value)
    assert "Expected one of ['USD', 'EUR'] for field 'currency'" in message
    assert "Next step:" in message
    assert "private-invalid-choice" not in message
