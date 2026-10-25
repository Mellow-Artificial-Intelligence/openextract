<div align="center">

# openextract

**Extract structured data from documents, images, audio, and video using LLMs.**

[![PyPI version](https://img.shields.io/pypi/v/openextract.svg?logo=pypi&logoColor=white&color=4B8BBE)](https://pypi.org/project/openextract/)
[![Python versions](https://img.shields.io/pypi/pyversions/openextract.svg?logo=python&logoColor=white)](https://pypi.org/project/openextract/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/Mellow-Artificial-Intelligence/openextract/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Mellow-Artificial-Intelligence/openextract/actions/workflows/ci.yml)

[Website](https://mellow-artificial-intelligence.github.io/openextract/) &middot; [PyPI](https://pypi.org/project/openextract/) &middot; [Changelog](CHANGELOG.md)

</div>

One function: give it a Pydantic schema, a model, and a file. You get back a validated instance.

```python
from pydantic import BaseModel
from openextract import extract


class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str


invoice = extract(Invoice, "openai:gpt-5", source="./invoice.pdf")  # path to a file on disk
print(invoice.total)
```

## Install

```bash
pip install 'openextract[all]'   # or just your provider: [openai], [anthropic], [google], ...
```

Requires Python 3.12+. Set the API key your provider expects, such as `OPENAI_API_KEY`.

## Usage

```python
extract(schema, model, source, instructions=None, *, media_type=None)
await extract_async(schema, model, source, instructions=None, *, media_type=None)
```

- **`source`** is where the file is: a path on disk, a `pathlib.Path`, or an `http(s)://` URL. openextract opens it for you. You can also pass the file's raw `bytes`. A string is always treated as a path or URL, never as the document's text; to extract from text you already have, pass `text.encode()`.
- **`model`** is any [pydantic-ai](https://ai.pydantic.dev/models/) model string (`"anthropic:claude-sonnet-5"`, `"google-gla:gemini-2.5-pro"`, `"ollama:llama3"`, ...) or a configured pydantic-ai `Model`.
- **`instructions`** are optional extra guidance for the model.
- **`media_type`** is inferred for you. openextract checks the file extension, then the URL's `Content-Type`, then the file's leading bytes (PDF, PNG, JPEG, GIF, WebP, MP3, WAV, FLAC, OGG, MP4, UTF-8 text). Pass it only to override the result.

```python
from pathlib import Path

extract(Invoice, "openai:gpt-5", source="./invoices/2291.pdf")  # file path
extract(Invoice, "openai:gpt-5", source=Path("invoices") / "2291.pdf")  # pathlib.Path
extract(Invoice, "anthropic:claude-sonnet-5", source="https://example.com/2291.pdf")  # URL
extract(Invoice, "openai:gpt-5", source=pdf_bytes, instructions="Amounts are in EUR.")  # bytes
```

## Decision models

For schemas made of `bool`, `Literal`, and `Enum` fields, use a decision model
instead of generating free-form text. Both `extract` and `extract_async` support
text and images with these providers:

| Model string | API key | Endpoint |
| --- | --- | --- |
| `openai-decisions:gpt-6-luna` | `OPENAI_API_KEY` | [OpenAI Decisions](https://developers.openai.com/api/docs/guides/decisions) |
| `openrouter:typesafe/jev-1.14` | `OPENROUTER_API_KEY` | OpenRouter Decisions |

No provider extra is needed for either decision endpoint. The explicit
`openai-decisions:` prefix selects the Decisions API; `openai:` continues to use
Responses for general extraction. OpenAI Decisions accepts 1–400 schema fields.
PDFs, audio, video, and free-form fields are not supported by decision models.

For example, classify a receipt and check whether it is paid. [Jev](https://openrouter.ai/typesafe/jev-1.14) is also available as `openrouter:typesafe/jev-1.14` or `openrouter:~typesafe/jev-latest`.

```python
from typing import Literal

from pydantic import BaseModel, Field


class Triage(BaseModel):
    document_type: Literal["invoice", "receipt", "contract"] = Field(
        description="What kind of document is this?"
    )
    is_paid: bool = Field(description="Has the amount been paid in full?")


triage = extract(Triage, "openai-decisions:gpt-6-luna", source="./receipt.png")
# Or: extract(Triage, "openrouter:typesafe/jev-1.14", source="./receipt.png")
```

Each field becomes one question, phrased by its `description`. A `bool` field is `True` when the probability of yes is at least 0.5. Other field types raise a `TypeError`. Missing, invalid, or refused OpenAI answers raise `SchemaValidationError`.

## Errors

Every error subclasses `ExtractionError` and keeps the original exception as `__cause__`.

| Exception | Raised when |
| --- | --- |
| `ModelError` | The provider returns an error. |
| `SchemaValidationError` | The model's output doesn't match the schema. |
| `ProviderNotInstalledError` | The provider's SDK isn't installed. |
| `ExtractionError` | A URL can't be fetched. |

If openextract can't infer the media type, it raises a `ValueError` and asks you to pass `media_type`.

## License

MIT
