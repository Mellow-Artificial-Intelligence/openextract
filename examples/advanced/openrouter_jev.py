"""Structured decisions from text via OpenRouter Jev Latest.

Jev Latest (``openrouter:~typesafe/jev-latest``) is a cheap text-in,
structured-decisions-out model (32k context). This cookbook uses
openextract's public ``extract`` API with a triage schema.

``--fixture`` (and the no-flag default) uses TestModel — no API key.
Live OpenRouter smoke (needs ``OPENROUTER_API_KEY``)::

    OPENROUTER_API_KEY=... uv run python -m examples.advanced.openrouter_jev --live
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pydantic_ai.models.test import TestModel

from examples._shared import openrouter_model
from openextract import extract

USAGE = "Usage: uv run python -m examples.advanced.openrouter_jev [--fixture|--live] [memo.txt]"

SAMPLE_MEMO = (
    "Ticket #4412 from billing@acme.example. Customer was charged $4,200 twice "
    "after a failed refund. Policy: auto-refund duplicate charges under $5,000 "
    "when the original payment is confirmed on the card."
)

INSTRUCTIONS = (
    "Read the memo and return a structured decision: approve, reject, or escalate. "
    "Include confidence, a short rationale, risk flags, and the next action."
)

FIXTURE_OUTPUT = {
    "decision": "approve",
    "confidence": 0.91,
    "rationale": "Duplicate charge is under the $5,000 auto-refund threshold.",
    "risk_flags": [],
    "next_action": "Issue the refund and email the customer.",
}


class Decision(BaseModel):
    decision: Literal["approve", "reject", "escalate"]
    confidence: float
    rationale: str
    risk_flags: list[str]
    next_action: str


def fixture_model() -> TestModel:
    return TestModel(custom_output_args=FIXTURE_OUTPUT)


def resolve_model(*, live: bool = False) -> str | TestModel:
    if live:
        return openrouter_model()
    return fixture_model()


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


def extract_decision(text: str, *, live: bool = False) -> Decision:
    return extract(
        schema=Decision,
        model=resolve_model(live=live),
        input_file=text.encode(),
        media_type="text/plain",
        instructions=INSTRUCTIONS,
    )


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    live, text = parse_argv(args)
    print(extract_decision(text, live=live).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
