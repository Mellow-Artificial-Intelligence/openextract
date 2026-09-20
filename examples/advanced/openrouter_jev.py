"""openextract + OpenRouter Decisions (Jev Latest).

Jev Latest (``~typesafe/jev-latest``) is a TypeSafe *decisions* model. It
cannot be used with chat/completions or openextract ``extract()`` — OpenRouter
returns 400 if you send it to ``/api/v1/chat/completions``.

Live path: ``POST https://openrouter.ai/api/alpha/decisions`` with
``Authorization: Bearer $OPENROUTER_API_KEY`` and body
``{model, state, questions}``. Questions use ``choice`` / ``noul`` / ``score``.
Docs: https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-questions-and-answers-request
TypeSafe / System One (``POST /api/v1/systemone``):
https://openrouter.ai/docs/guides/community/typesafe-sdk

This cookbook:

1. Uses openextract ``extract()`` + TestModel to pull structured fields from a memo.
2. Posts that object as Decisions ``state`` and maps answers onto a Pydantic model.

``--fixture`` (and the no-flag default) uses canned answers — no network.
Live Decisions smoke (needs ``OPENROUTER_API_KEY``)::

    OPENROUTER_API_KEY=... uv run python -m examples.advanced.openrouter_jev --live
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal, cast

import httpx
from pydantic import BaseModel, ValidationError
from pydantic_ai.models.test import TestModel

from openextract import extract

USAGE = "Usage: uv run python -m examples.advanced.openrouter_jev [--fixture|--live] [memo.txt]"
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "~typesafe/jev-latest"

SAMPLE_MEMO = (
    "Ticket #4412 from billing@acme.example. Customer was charged $4,200 twice "
    "after a failed refund. Policy: auto-refund duplicate charges under $5,000 "
    "when the original payment is confirmed on the card."
)

STATE_INSTRUCTIONS = "Extract ticket_id, customer, amount_usd, issue, and policy from the memo."

QUESTIONS: dict[str, dict[str, object]] = {
    "action": {
        "type": "choice",
        "instructions": "Should this ticket be approved, rejected, or escalated?",
        "criteria": {
            "approve": "Policy allows an automatic refund.",
            "reject": "The request is out of policy.",
            "escalate": "A human needs to review before acting.",
        },
    },
    "auto_refund": {
        "type": "noul",
        "instructions": "Is this ticket eligible for an automatic refund under the stated policy?",
        "criteria": {
            "true": "Duplicate charge under $5,000 with a confirmed original payment.",
            "false": "Amount, confirmation, or policy does not support an automatic refund.",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this billing ticket?",
        "criteria": [
            "Can wait for the next billing cycle",
            "Should be resolved this week",
            "Customer is blocked on a large charge right now",
        ],
    },
}

FIXTURE_STATE = {
    "ticket_id": "4412",
    "customer": "billing@acme.example",
    "amount_usd": 4200.0,
    "issue": "Charged twice after a failed refund",
    "policy": "Auto-refund duplicate charges under $5,000 when the original payment is confirmed.",
}

FIXTURE_RESPONSE: dict[str, object] = {
    "id": "gen-dec-fixture",
    "model": "typesafe/jev-1.13-20260917",
    "provider": "TypeSafe",
    "answers": {
        "action": {
            "type": "choice",
            "choice": "approve",
            "confidence": 0.91,
            "probabilities": {"approve": 0.91, "reject": 0.02, "escalate": 0.07},
        },
        "auto_refund": {"type": "noul", "noul": 0.94},
        "urgency": {
            "type": "score",
            "score": 1.8,
            "confidence": 0.88,
        },
    },
    "usage": {"input_tokens": 200, "output_tokens": 40, "cost": 0.00002},
}


class DecisionsError(RuntimeError):
    """Decisions API or answer-mapping failure (never includes credentials)."""


class TicketState(BaseModel):
    ticket_id: str
    customer: str
    amount_usd: float
    issue: str
    policy: str


class Decision(BaseModel):
    action: Literal["approve", "reject", "escalate"]
    action_confidence: float
    auto_refund: bool
    auto_refund_noul: float
    urgency: float
    model: str


class CookbookResult(BaseModel):
    state: TicketState
    decision: Decision


FIXTURE_OUTPUT = CookbookResult(
    state=TicketState.model_validate(FIXTURE_STATE),
    decision=Decision(
        action="approve",
        action_confidence=0.91,
        auto_refund=True,
        auto_refund_noul=0.94,
        urgency=1.8,
        model="typesafe/jev-1.13-20260917",
    ),
).model_dump()


def decisions_model() -> str:
    return os.environ.get("OPENROUTER_DECISIONS_MODEL", JEV_MODEL)


def require_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        print("OPENROUTER_API_KEY is required for --live.")
        raise SystemExit(1)
    return key


def extract_state(text: str) -> TicketState:
    return extract(
        schema=TicketState,
        model=TestModel(custom_output_args=FIXTURE_STATE),
        input_file=text.encode(),
        media_type="text/plain",
        instructions=STATE_INSTRUCTIONS,
    )


def _answer_field(answers: dict[str, object], name: str, expected: str) -> dict[str, object]:
    raw = answers.get(name)
    if not isinstance(raw, dict):
        raise DecisionsError(f"missing answer {name!r}")
    kind = raw.get("type")
    match kind:
        case "choice" | "noul" | "score":
            if kind != expected:
                raise DecisionsError(f"answer {name!r} type {kind!r} != {expected!r}")
        case _:
            raise DecisionsError(f"unsupported answer type: {kind!r}")
    return cast(dict[str, object], raw)


def decision_from_response(payload: dict[str, object]) -> Decision:
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise DecisionsError("Decisions response missing answers")
    action = _answer_field(answers, "action", "choice")
    refund = _answer_field(answers, "auto_refund", "noul")
    urgency = _answer_field(answers, "urgency", "score")
    match action.get("choice"):
        case "approve" | "reject" | "escalate" as action_choice:
            pass
        case other:
            raise DecisionsError(f"invalid action {other!r}")
    try:
        noul = float(refund["noul"])
        return Decision(
            action=action_choice,
            action_confidence=float(action["confidence"]),
            auto_refund=noul >= 0.5,
            auto_refund_noul=noul,
            urgency=float(urgency["score"]),
            model=str(payload.get("model") or JEV_MODEL),
        )
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise DecisionsError(f"Could not map Decisions answers: {exc}") from exc


def submit_decisions(
    state: TicketState | dict[str, object],
    *,
    api_key: str,
    model: str | None = None,
    questions: dict[str, dict[str, object]] | None = None,
    client: httpx.Client | None = None,
) -> dict[str, object]:
    payload = {
        "model": model or decisions_model(),
        "state": state.model_dump() if isinstance(state, TicketState) else state,
        "questions": questions or QUESTIONS,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    own_client = client is None
    http = client or httpx.Client(timeout=30.0)
    try:
        response = http.post(DECISIONS_URL, headers=headers, json=payload)
        if response.status_code >= 400:
            raise DecisionsError(f"Decisions API returned {response.status_code}: {response.text}")
        try:
            body = response.json()
        except ValueError as exc:
            raise DecisionsError("Decisions API returned non-JSON") from exc
        if not isinstance(body, dict):
            raise DecisionsError("Decisions API returned a non-object JSON body")
        return cast(dict[str, object], body)
    except httpx.HTTPError as exc:
        raise DecisionsError(f"Decisions API request failed: {exc}") from exc
    finally:
        if own_client:
            http.close()


def run_cookbook(
    text: str,
    *,
    live: bool = False,
    client: httpx.Client | None = None,
) -> CookbookResult:
    state = extract_state(text)
    payload = (
        submit_decisions(state, api_key=require_api_key(), client=client)
        if live
        else FIXTURE_RESPONSE
    )
    return CookbookResult(state=state, decision=decision_from_response(payload))


def load_memo(path: str | None) -> str:
    if path is None:
        return SAMPLE_MEMO
    memo = Path(path)
    if not memo.is_file():
        print(f"Missing file: {path}")
        sys.exit(1)
    return memo.read_text(encoding="utf-8")


def parse_argv(argv: list[str]) -> tuple[bool, str]:
    if "--live" in argv and "--fixture" in argv:
        print("Use either --fixture or --live, not both.")
        sys.exit(1)

    live = False
    path: str | None = None
    for arg in argv:
        if arg == "--live":
            live = True
        elif arg == "--fixture":
            continue
        elif arg.startswith("-") or path is not None:
            print(USAGE)
            sys.exit(1)
        else:
            path = arg
    return live, load_memo(path)


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    live, text = parse_argv(args)
    try:
        print(run_cookbook(text, live=live).model_dump_json(indent=2))
    except DecisionsError as exc:
        print(exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
