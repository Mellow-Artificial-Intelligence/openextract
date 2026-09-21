"""openextract + OpenRouter Decisions (Jev Latest) fraud-check pipeline.

Jev Latest (``~typesafe/jev-latest``) is a TypeSafe *decisions* model. It
cannot be used with chat/completions or openextract ``extract()`` — OpenRouter
returns 400 if you send it to ``/api/v1/chat/completions``.

This cookbook:

1. Uses openextract ``extract()`` + TestModel to pull document metadata,
   content, and risk signals from a memo, PDF, or image.
2. Posts that object as Decisions ``state`` (``POST /api/alpha/decisions``).
3. Maps Jev answers onto a result object. ``reasoning`` is composed here from
   the extracted signals and the selected choice label — Jev does not emit
   free-form prose.

``--fixture`` (and the no-flag default) uses canned answers — no network.
Pass a path to a memo, PDF, or image (including bundled fixtures). Live
Decisions smoke (needs ``OPENROUTER_API_KEY``)::

    OPENROUTER_API_KEY=... uv run python -m examples.advanced.openrouter_jev_fraud --live

Docs: https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-questions-and-answers-request
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

USAGE = (
    "Usage: uv run python -m examples.advanced.openrouter_jev_fraud "
    "[--fixture|--live] [memo.txt|document.pdf|scan.png]"
)
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "~typesafe/jev-latest"

SAMPLE_MEMO = (
    "Internal review memo — wire transfer W-8831. Source: vendor-invoice.pdf. "
    "Amount $18,750.00 from Northshore Holdings LLC to Apex Trading Co. "
    "(new vendor, onboarded 2 days ago) on 2026-09-12. Overnight wire requested "
    "by email from accounts@northshore.example. Bank details differ from the "
    "on-file W-9. Invoice PDF metadata shows a different seller name. Caller "
    "could not confirm the callback number."
)

STATE_INSTRUCTIONS = (
    "Extract source, amount_usd, payer, payee, date, a content summary, and "
    "risk_signals (short codes such as new_payee or bank_details_mismatch)."
)

QUESTIONS: dict[str, dict[str, object]] = {
    "verdict": {
        "type": "choice",
        "instructions": "Is this document fraudulent, not fraudulent, or does it need review?",
        "criteria": {
            "fraud": (
                "Strong evidence of fraud: mismatched parties, altered bank "
                "details, or forged signals."
            ),
            "not_fraud": "Routine transaction with consistent parties, amounts, and documents.",
            "review": "Ambiguous or incomplete evidence; a human should inspect before acting.",
        },
    },
    "severity": {
        "type": "score",
        "instructions": "How severe is the fraud risk on this document?",
        "criteria": [
            "Low: minor inconsistencies, no loss likely",
            "Medium: plausible fraud, limited exposure",
            "High: likely fraud with material financial exposure",
        ],
    },
    "needs_manual_review": {
        "type": "noul",
        "instructions": "Should a human reviewer inspect this before funds move?",
        "criteria": {
            "true": "Signals are incomplete or the verdict is review/fraud with material amount.",
            "false": "Evidence is conclusive enough to act without a human.",
        },
    },
}

FIXTURE_STATE = {
    "source": "vendor-invoice.pdf",
    "amount_usd": 18750.0,
    "payer": "Northshore Holdings LLC",
    "payee": "Apex Trading Co.",
    "date": "2026-09-12",
    "content": (
        "Overnight wire to a new vendor. Bank details differ from the on-file "
        "W-9. Invoice metadata seller name does not match the payee. Callback "
        "number could not be verified."
    ),
    "risk_signals": [
        "new_payee",
        "bank_details_mismatch",
        "rushed_wire",
        "unverified_callback",
    ],
}

FIXTURE_RESPONSE: dict[str, object] = {
    "id": "gen-dec-fraud-fixture",
    "model": "typesafe/jev-1.13-20260917",
    "provider": "TypeSafe",
    "answers": {
        "verdict": {
            "type": "choice",
            "choice": "review",
            "confidence": 0.78,
            "probabilities": {"fraud": 0.31, "not_fraud": 0.08, "review": 0.61},
        },
        "severity": {"type": "score", "score": 1.6, "confidence": 0.84},
        "needs_manual_review": {"type": "noul", "noul": 0.92},
    },
    "usage": {"input_tokens": 240, "output_tokens": 36, "cost": 0.00002},
}

Verdict = Literal["fraud", "not_fraud", "review"]


class DecisionsError(RuntimeError):
    """Decisions API or answer-mapping failure (never includes credentials)."""


class FraudDocState(BaseModel):
    source: str
    amount_usd: float
    payer: str
    payee: str
    date: str
    content: str
    risk_signals: list[str]


class CookbookResult(BaseModel):
    state: FraudDocState
    result: Verdict
    confidence: float
    reasoning: str
    severity: float
    needs_manual_review: bool
    model: str


def compose_reasoning(state: FraudDocState, verdict: Verdict) -> str:
    """Map extracted signals + the selected choice label into readable text.

    Jev answers are typed (choice / noul / score) and do not include a prose
    rationale; this string is assembled locally from ``state`` and ``QUESTIONS``.
    """
    criteria = QUESTIONS["verdict"]["criteria"]
    assert isinstance(criteria, dict)
    label = str(criteria[verdict])
    signals = ", ".join(state.risk_signals) or "none"
    return (
        f"Decisions choice={verdict!r} ({label}) with extract() signals "
        f"[{signals}]. {state.payer} → {state.payee} ${state.amount_usd:,.2f} "
        f"on {state.date} ({state.source})."
    )


FIXTURE_OUTPUT = CookbookResult(
    state=FraudDocState.model_validate(FIXTURE_STATE),
    result="review",
    confidence=0.78,
    reasoning=compose_reasoning(FraudDocState.model_validate(FIXTURE_STATE), "review"),
    severity=1.6,
    needs_manual_review=True,
    model="typesafe/jev-1.13-20260917",
).model_dump()


def decisions_model() -> str:
    return os.environ.get("OPENROUTER_DECISIONS_MODEL", JEV_MODEL)


def require_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        print("OPENROUTER_API_KEY is required for --live.")
        raise SystemExit(1)
    return key


def extract_state(
    input_file: str | bytes,
    *,
    media_type: str | None = None,
) -> FraudDocState:
    return extract(
        schema=FraudDocState,
        model=TestModel(custom_output_args=FIXTURE_STATE),
        input_file=input_file,
        media_type=media_type,
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


def _choice_confidence(answer: dict[str, object], choice: str) -> float:
    raw = answer.get("confidence")
    if raw is not None:
        return float(raw)
    probs = answer.get("probabilities")
    if isinstance(probs, dict) and choice in probs:
        return float(probs[choice])
    raise DecisionsError("missing verdict confidence")


def result_from_response(payload: dict[str, object], state: FraudDocState) -> CookbookResult:
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise DecisionsError("Decisions response missing answers")
    verdict = _answer_field(answers, "verdict", "choice")
    severity = _answer_field(answers, "severity", "score")
    review = _answer_field(answers, "needs_manual_review", "noul")
    match verdict.get("choice"):
        case "fraud" | "not_fraud" | "review" as result:
            pass
        case other:
            raise DecisionsError(f"invalid verdict {other!r}")
    try:
        noul = float(review["noul"])
        return CookbookResult(
            state=state,
            result=result,
            confidence=_choice_confidence(verdict, result),
            reasoning=compose_reasoning(state, result),
            severity=float(severity["score"]),
            needs_manual_review=noul >= 0.5,
            model=str(payload.get("model") or JEV_MODEL),
        )
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise DecisionsError(f"Could not map Decisions answers: {exc}") from exc


def submit_decisions(
    state: FraudDocState | dict[str, object],
    *,
    api_key: str,
    model: str | None = None,
    questions: dict[str, dict[str, object]] | None = None,
    client: httpx.Client | None = None,
) -> dict[str, object]:
    payload = {
        "model": model or decisions_model(),
        "state": state.model_dump() if isinstance(state, FraudDocState) else state,
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
    input_file: str | bytes = SAMPLE_MEMO.encode(),
    *,
    media_type: str | None = "text/plain",
    live: bool = False,
    client: httpx.Client | None = None,
) -> CookbookResult:
    state = extract_state(input_file, media_type=media_type)
    payload = (
        submit_decisions(state, api_key=require_api_key(), client=client)
        if live
        else FIXTURE_RESPONSE
    )
    return result_from_response(payload, state)


def resolve_input(path: str | None) -> tuple[bytes | str, str | None]:
    if path is None:
        return SAMPLE_MEMO.encode(), "text/plain"
    source = Path(path)
    if not source.is_file():
        print(f"Missing file: {path}")
        sys.exit(1)
    return str(source), None


def parse_argv(argv: list[str]) -> tuple[bool, bytes | str, str | None]:
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
    input_file, media_type = resolve_input(path)
    return live, input_file, media_type


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    live, input_file, media_type = parse_argv(args)
    try:
        print(run_cookbook(input_file, media_type=media_type, live=live).model_dump_json(indent=2))
    except DecisionsError as exc:
        print(exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
