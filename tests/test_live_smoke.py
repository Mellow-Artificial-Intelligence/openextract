"""Opt-in live provider smoke tests.

These make real network/model calls. Default pytest/CI runs skip them.

Enable with::

    OPENEXTRACT_LIVE_SMOKE=1 uv run pytest -m integration tests/test_live_smoke.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import BaseModel

import examples.advanced.openrouter_jev as jev
from openextract import extract

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples" / "fixtures" / "document_page.png"

# Record the identifiers exercised by this harness when opting in.
_DEFAULT_LIVE_MODEL = "openai:gpt-5"


class _SmokeInfo(BaseModel):
    summary: str
    language: str


def _live_enabled() -> bool:
    return os.environ.get("OPENEXTRACT_LIVE_SMOKE", "").lower() in ("1", "true", "yes")


@pytest.mark.integration
def test_live_openai_image_smoke() -> None:
    """Representative OpenAI image path; skipped unless explicitly enabled."""
    if not _live_enabled():
        pytest.skip("Set OPENEXTRACT_LIVE_SMOKE=1 to run live provider smoke tests")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required for the OpenAI live smoke test")
    if not FIXTURE.is_file():
        pytest.skip(f"missing fixture: {FIXTURE}")

    model = os.environ.get("OPENEXTRACT_LIVE_MODEL", _DEFAULT_LIVE_MODEL)
    result = extract(
        schema=_SmokeInfo,
        model=model,
        input_file=str(FIXTURE),
        instructions="Return a one-sentence summary and the primary language.",
    )
    assert isinstance(result.summary, str) and result.summary.strip()
    assert isinstance(result.language, str) and result.language.strip()


@pytest.mark.integration
def test_live_openrouter_jev_decisions_smoke() -> None:
    """OpenRouter Decisions (Jev Latest); skipped unless explicitly enabled."""
    if not _live_enabled():
        pytest.skip("Set OPENEXTRACT_LIVE_SMOKE=1 to run live provider smoke tests")
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY is required for the OpenRouter Jev live smoke test")

    result = jev.run_cookbook(jev.SAMPLE_MEMO, live=True)
    assert result.decision.action in {"approve", "reject", "escalate"}
    assert 0.0 <= result.decision.auto_refund_noul <= 1.0
    assert result.state.ticket_id.strip()
