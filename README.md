<div align="center">

# openextract

**Extract structured data from documents, images, audio, and video using LLMs.**

[![PyPI version](https://img.shields.io/pypi/v/openextract.svg?logo=pypi&logoColor=white&color=4B8BBE)](https://pypi.org/project/openextract/)
[![Python versions](https://img.shields.io/pypi/pyversions/openextract.svg?logo=python&logoColor=white)](https://pypi.org/project/openextract/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/Mellow-Artificial-Intelligence/openextract/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Mellow-Artificial-Intelligence/openextract/actions/workflows/ci.yml)

[Website](https://mellow-artificial-intelligence.github.io/openextract/) &middot; [PyPI](https://pypi.org/project/openextract/) &middot; [Changelog](CHANGELOG.md) &middot; [Roadmap](https://github.com/Mellow-Artificial-Intelligence/openextract/wiki/Roadmap)

</div>

One function: give it a Pydantic schema, a model, and a source. You get back a validated instance.
Supported input formats depend on the chosen model.

## First extraction

Requires Python 3.12+. Install just the OpenAI provider for this example:

```bash
pip install 'openextract[openai]'
```

Set `OPENAI_API_KEY` to your OpenAI API key in the same terminal. Your account
needs access to `gpt-5`; running the example makes a paid provider request.

```bash
export OPENAI_API_KEY='your-api-key'  # macOS / Linux
```

On PowerShell, use `$env:OPENAI_API_KEY = 'your-api-key'` instead.

Save the following as `first_extraction.py`. The fictional invoice is included
as text bytes, so there is no document to download or missing file to create.

```python
from decimal import Decimal

from pydantic import BaseModel, Field

from openextract import extract


class Invoice(BaseModel):
    vendor: str = Field(description="Seller, not buyer.")
    total: Decimal = Field(description="Final amount due, including tax.")
    currency: str | None = Field(description="Three-letter currency code, or null if unclear.")


source = b"""INVOICE DEMO-001 (fictional sample)
Seller: Acme Paper Co.
Bill to: Northwind Studio
Paper supplies: 100.00 USD
Subtotal: 100.00 USD
Tax: 20.00 USD
Total due: 120.00 USD
"""

invoice = extract(Invoice, "openai:gpt-5", source)
print(invoice.vendor)  # str: Acme Paper Co.
print(invoice.total)  # Decimal: 120.00
print(invoice.currency)  # str | None: USD
```

```bash
python first_extraction.py
```

Expected fields for this fixture:

| Field | Python type | Expected value |
| --- | --- | --- |
| `vendor` | `str` | `"Acme Paper Co."` (seller, not Northwind Studio) |
| `total` | `Decimal` | `Decimal("120.00")` (final amount including tax, not the subtotal) |
| `currency` | `str \| None` | `"USD"` (or `None` when the source is unclear) |

Schema validation checks structure and types; it does **not** guarantee factual
accuracy. Check extracted values against your source before relying on them.

The same non-sensitive invoice is supplied in
[`examples/fixtures/invoice.txt`](examples/fixtures/invoice.txt). In a repository
checkout, after setting the key, run `uv run --extra openai python examples/basic.py`;
it finds the fixture automatically. An optional path or URL argument lets you use
your own document. See [example checks](examples/README.md) for offline validation
and the live-check record.

Continue with [fine-detail controls](examples/fine_details.md): field descriptions,
`Decimal` totals, nullable currency, validators, interpretation instructions, and
media-type overrides, using the same `Invoice` result type.

## Usage

```python
extract(schema, model, source, instructions=None, *, media_type=None)
await extract_async(schema, model, source, instructions=None, *, media_type=None)
```

- **`source`** is where the file is: a path on disk, a `pathlib.Path`, or an `http(s)://` URL. openextract opens it for you. You can also pass the file's raw `bytes`. A string is always treated as a path or URL, never as the document's text; to extract from text you already have, pass `text.encode()`.
- **`model`** is any [pydantic-ai](https://ai.pydantic.dev/models/) model string (`"anthropic:claude-sonnet-5"`, `"google-gla:gemini-2.5-pro"`, `"ollama:llama3"`, ...) or a configured pydantic-ai `Model`.
- **`instructions`** are optional extra guidance for the model.
- **`media_type`** is inferred for you. openextract checks the file extension, then the URL's `Content-Type`, then the file's leading bytes (PDF, PNG, JPEG, GIF, WebP, MP3, WAV, FLAC, OGG, MP4, UTF-8 text). Pass it only to override the result.

Media detection does not mean every model accepts every format. Choose a model
that supports your input and schema.

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
