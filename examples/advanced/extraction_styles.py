"""Choose how the model inspects a text document.

``direct`` sends the bytes to the LLM. ``table`` adds line-item guidance
and parses PDFs by page. ``form`` adds labeled-field guidance for forms
and receipts (same PDF parse-then-window path). ``search`` and ``code``
use the Pydantic AI harness (file tools or sandboxed Python) and need
``pydantic-ai-harness`` / ``pydantic-ai-harness[codemode]``.
"""

from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from openextract import ExtractionStyle, Extractor


class Note(BaseModel):
    title: str
    topic: str


def test_agent() -> Agent:
    return Agent(
        TestModel(custom_output_args={"title": "Q4 notes", "topic": "revenue"}),
        output_type=Note,
    )


def main() -> None:
    document = b"Q4 notes: revenue grew 12% year over year."
    with Extractor(Note, agent=test_agent(), style=ExtractionStyle.DIRECT) as extractor:
        result = extractor.extract(document, media_type="text/plain")
    print(result.model_dump_json(indent=2))
    print("Other styles: style='table' (line items), 'form' (receipts/forms), 'search', or 'code'.")


if __name__ == "__main__":
    main()
