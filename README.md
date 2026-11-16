<div align="center">

# openextract

**Extract structured data from documents, images, audio, and video using LLMs.**

[![PyPI version](https://img.shields.io/pypi/v/openextract.svg?logo=pypi&logoColor=white&color=4B8BBE)](https://pypi.org/project/openextract/)
[![Python versions](https://img.shields.io/pypi/pyversions/openextract.svg?logo=python&logoColor=white)](https://pypi.org/project/openextract/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/Mellow-Artificial-Intelligence/openextract/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Mellow-Artificial-Intelligence/openextract/actions/workflows/ci.yml)

[Website](https://mellow-artificial-intelligence.github.io/openextract/) &middot; [PyPI](https://pypi.org/project/openextract/) &middot; [Changelog](CHANGELOG.md)

</div>

One function: give it a Pydantic schema, a model, and a source. You get back a validated instance.

```bash
uv add openextract  # or: pip install openextract
```

Requires Python 3.12+. The only dependencies are HTTPX and Pydantic: every provider
is called over plain HTTP, with no provider SDKs or extras to install.

## First extraction

Set `OPENAI_API_KEY`, save this as `first_extraction.py`, and run it with
`uv run python first_extraction.py`. It makes one paid request.

```python
from decimal import Decimal

from pydantic import BaseModel, Field

from openextract import extract


class Invoice(BaseModel):
    vendor: str = Field(description="Seller, not buyer.")
    total: Decimal = Field(description="Final amount due, including tax.")
    currency: str | None = Field(description="Three-letter currency code, or null if unclear.")


source = """INVOICE DEMO-001 (fictional sample)
Seller: Acme Paper Co.
Bill to: Northwind Studio
Paper supplies: 100.00 USD
Subtotal: 100.00 USD
Tax: 20.00 USD
Total due: 120.00 USD
"""

invoice = extract(Invoice, "openai:gpt-6.1-sol", source)
print(invoice.vendor)  # Acme Paper Co.
print(invoice.total)  # 120.00
print(invoice.currency)  # USD
```

The result is an `Invoice`, validated by your schema. Validation checks structure
and types, not facts: review important values against the source.
[`examples/basic.py`](examples/basic.py) runs the same extraction on a file, a path,
or a URL.

## Usage

`extract` and `extract_async` take the same arguments:
`(schema, model, source, instructions=None, *, media_type=None, client=None, include_metadata=False)`.

| Argument | Meaning |
| --- | --- |
| `schema` | The Pydantic model to fill. Field descriptions, constraints, and validators all apply. |
| `model` | A `"provider:model"` string such as `"anthropic:claude-sonnet-5-5"`, or a configured [`Model`](#models-and-settings). |
| `source` | Text, a `pathlib.Path`, an `http(s)://` URL, or `bytes`. A string is always text unless it is a URL. |
| `instructions` | Optional guidance for this call. |
| `media_type` | Overrides detection. Text defaults to `text/plain`; otherwise the file extension, the URL's `Content-Type`, then the leading bytes decide. |
| `client` | Your own `httpx.Client` (or `httpx.AsyncClient`) for model requests. It stays open. Source URLs are fetched separately, without model credentials. |
| `include_metadata` | Returns an [`ExtractionResult`](#openrouter-extractions) with OpenRouter Extractions. |

```python
from pathlib import Path

extract(Invoice, "openai:gpt-6.1-sol", "Total due: 120.00")  # text
extract(Invoice, "openai:gpt-6.1-sol", Path("invoices/2291.pdf"))  # local file
extract(Invoice, "anthropic:claude-sonnet-5-5", "https://example.com/2291.pdf")  # URL
extract(Invoice, "google:gemini-2.5-flash", audio_bytes, media_type="audio/mpeg")  # bytes
```

Shape the result with ordinary Pydantic. A `Field(description=...)` tells the model
what a field means. A field typed `str | None` with no default must be answered but
may be `null`, which lets the model say a value is missing instead of inventing one.
Validators and constraints run on the output: a failure is sent back to the model
once as feedback, then raised as `SchemaValidationError`. Keep reusable meaning in
the schema and use `instructions` for guidance specific to one call.

## Providers

The model prefix selects an HTTP API and where its key is read from. The model you
choose must support the input type and the structured output its adapter uses.

| Prefix | HTTP API | API key variable | Inputs |
| --- | --- | --- | --- |
| `openai`, `openai-responses` | Responses, strict JSON Schema | `OPENAI_API_KEY` | Text, images, PDF, Word, RTF, ODT, PowerPoint, Excel |
| `openai-chat` | Chat Completions, tool or JSON Schema | `OPENAI_API_KEY` | Text, images, PDF, MP3, WAV |
| `anthropic` | Messages, tool or JSON Schema | `ANTHROPIC_API_KEY` | Text, PNG, JPEG, GIF, WebP, PDF |
| `google`, `google-gla`, `gemini` | `generateContent`, JSON Schema | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Text, images, PDF, audio, video, Office, OpenDocument |
| `openrouter` | Chat Completions, [Extractions](#openrouter-extractions), or [Decisions](#decision-models), by model | `OPENROUTER_API_KEY` | Depends on the API and model |
| `groq`, `mistral`, `xai`, `cerebras`, `huggingface` | Chat Completions, tool or JSON Schema | `GROQ_API_KEY`, `MISTRAL_API_KEY`, `XAI_API_KEY`, `CEREBRAS_API_KEY`, `HF_TOKEN` | Encoded as `openai-chat`; support varies by endpoint |
| `mistral` with `api="mistral-ocr"` | [Document annotations](#mistral-document-annotations) | `MISTRAL_API_KEY` | PDF, including scans |
| `cohere` | V2 Chat, JSON Schema | `CO_API_KEY` | Text, images |
| `bedrock` | Converse, tool | `AWS_BEARER_TOKEN_BEDROCK` | Text, PNG, JPEG, GIF, WebP, PDF |
| `ollama` | Local `/api/chat`, JSON Schema | None | Text, images |
| `openai-decisions`, `gemini-decisions` | [Decisions](#decision-models) | The provider's key | Text, images; Gemini also reads PDF |

An input type the adapter cannot encode raises `ValueError` before any request. A
model that rejects an encoded input reports it as `ModelError`.

- **Strict JSON Schema** (Responses, and Chat Completions in JSON Schema mode) asks
  for every declared field, including those with defaults, and rejects open-ended
  dictionaries. Use a tool-calling model with `Model(..., api="chat")` for those.
  Tagged unions are adapted automatically.
- **Gemini** reads only the first sheet of a spreadsheet, and Office files lose some
  layout. See Google's [document guide](https://aistudio.google.com/docs/document-processing).
- **Bedrock** needs `AWS_REGION` (or `AWS_DEFAULT_REGION`) unless you set `base_url`,
  and authenticates with a [bearer token](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys-use.html).
  SigV4 signing is not built in: pass your own authenticated `client` and
  `Model(..., api_key="")` to skip the key lookup.
- **Ollama** structured output works locally; Ollama Cloud does not support it yet.

## Models and settings

A `Model` configures a built-in provider, or any endpoint that speaks a supported
API. A provider name openextract does not know needs both `api` and `base_url`.

```python
import os

from openextract import Model

model = Model(
    "custom:my-model",
    api="chat",
    base_url="https://models.example.com/v1",
    api_key=os.environ["CUSTOM_API_KEY"],
    headers={"X-Tenant": "accounts"},
    settings={"temperature": 0},
    timeout=60,
    max_retries=2,
    validation_retries=1,
)
invoice = extract(Invoice, model, Path("invoice.pdf"))
```

| Option | Meaning |
| --- | --- |
| `api` | `responses`, `chat`, `anthropic`, `gemini`, `ollama`, `cohere`, `bedrock`, `mistral-ocr`, `openai-decisions`, `gemini-decisions`, `openrouter-decisions`, or `openrouter-extractions`. |
| `base_url` | Includes the API version, such as `https://host/v1`. Built-in providers also read `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `GOOGLE_BASE_URL`, `OPENROUTER_BASE_URL`, `GROQ_BASE_URL`, `MISTRAL_BASE_URL`, `XAI_BASE_URL`, `CEREBRAS_BASE_URL`, `HF_BASE_URL`, `COHERE_BASE_URL`, `BEDROCK_BASE_URL`, and `OLLAMA_BASE_URL`. |
| `api_key`, `headers` | Credentials belong here, never in the URL. Explicit values win over the environment, and your headers win over generated ones. Omit `api_key` for keyless endpoints. |
| `settings` | Generation parameters in the provider's own names. They go to the top level of the request, except Gemini (`generationConfig`), Ollama sampling (`settings["options"]`), and Bedrock inference (`settings["inferenceConfig"]`). Anthropic's `max_tokens` defaults to 4096. Request structure and schema fields cannot be replaced, streaming is not supported, and Decisions accept no settings. |
| `timeout` | Seconds allowed for each model request. Defaults to 60. |
| `max_retries` | Retries for transport failures and HTTP 408, 409, 429, 500, 502, 503, and 504, with exponential backoff or `Retry-After` capped at 30 seconds. Defaults to 2. Other errors and redirects fail immediately. |
| `validation_retries` | Retries after output that fails your schema, with feedback that omits raw values. Defaults to 1. Each attempt has its own HTTP retry budget. Refusals, truncated output, and malformed responses fail immediately. |
| `output_mode` | `"tool"` or `"json_schema"`, for Chat Completions and Anthropic Messages. Both default to a forced tool call. |

OpenRouter serves Decisions and Extractions from `/api/alpha`. An
`OPENROUTER_BASE_URL` ending in `/api/v1` is translated for them; an explicit
`base_url` should end in `/api/alpha`.

`output_mode="json_schema"` requests JSON text against your schema instead of a tool
call. The model and endpoint must support it; there is no automatic fallback.

```python
model = Model(
    "openrouter:google/gemini-2.5-flash",
    output_mode="json_schema",
    settings={"provider": {"require_parameters": True}},
)
```

With Anthropic, this mode uses [native JSON outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).
Objects are closed and optional fields keep their defaults. Numeric bounds, string
lengths and patterns, unsupported formats, and array limits move into field
descriptions and are still enforced by your schema. Open-ended dictionaries,
recursive references, tuples, and `allOf` raise `ValueError` before any request:
use tool mode for those.

## Mistral document annotations

`Model(api="mistral-ocr")` sends a PDF, including scans, to Mistral's
[`/v1/ocr`](https://docs.mistral.ai/api/endpoint/ocr) endpoint with your schema as
the document annotation format. `instructions` become the annotation prompt.

```python
model = Model("mistral:mistral-ocr-latest", api="mistral-ocr")
invoice = extract(Invoice, model, Path("scanned-invoice.pdf"))
```

Only PDFs are accepted, and plain `mistral:` strings still use Chat Completions.
Select pages with `settings={"pages": [0, 1]}`. Mistral annotates the OCR text plus
the first eight extracted images, so prefer text-focused scans, and enforces its own
[limits](https://docs.mistral.ai/studio/document-processing/annotations) of 50 MB
and 1,000 pages.

## OpenRouter Extractions

OpenRouter's alpha [Extractions API](https://openrouter.ai/docs/guides/community/extractions)
returns typed fields with confidence and source spans. It is selected automatically
for `cloudflare/clairic`, `inception/mercury-decide`, `levanto/sage-2`, and
`scaledown/extract`, or explicitly with `api="openrouter-extractions"`.

```python
from datetime import date
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, Field

from openextract import OpenRouterField, extract


class Invoice(BaseModel):
    invoice_number: str
    total: Annotated[Decimal, OpenRouterField(decimals=2, unit="USD")]
    due_date: date | None = Field(description="The payment due date, if stated.")


result = extract(
    Invoice,
    "openrouter:cloudflare/clairic",
    "Invoice INV-2291. Total due: 1080.00 USD. Pay by 2026-10-02.",
    include_metadata=True,
)
print(result.data.total)
print(result.answers["invoice_number"].get("span"))
```

Without `include_metadata`, the call returns the validated instance as usual. With
it, `ExtractionResult.data` holds the instance, `answers` the provider's answer for
each field, and `metadata` the rest of the response, such as usage. If OpenRouter
read only part of the source, the default call raises `SchemaValidationError`; with
metadata, check `result.metadata.get("truncated")` instead.

Fields may be `str`, `int`, `float`, `Decimal`, `date`, `time`, `datetime`, their
nullable forms, and lists of text, numbers, or dates. Use a structured-output
provider for nested records. `OpenRouterField` inside `Annotated` sets native
options or an explicit type, such as
`Annotated[float, OpenRouterField(type="estimate", min=0, max=90)]`. Text, PDF, PNG,
JPEG, and WebP are encoded; each model has its own limits. Pass request options such
as `reference_date` and `provider` through `Model(settings=...)`.

## Decision models

For schemas made only of `bool`, `Literal`, and `Enum` fields, a decision model
picks each answer from a closed set instead of generating text.

| Model string | API key variable | Inputs | Limits |
| --- | --- | --- | --- |
| `openai-decisions:gpt-6-luna` | `OPENAI_API_KEY` | Text, images | 1 to 400 fields |
| `gemini-decisions:gemini-decision-1` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Text, images, PDF | 1 to 1,000 fields, 2 to 255 options each |
| `openrouter:typesafe/jev-1.14` | `OPENROUTER_API_KEY` | Text, images | Set by the model |

```python
from typing import Literal


class Triage(BaseModel):
    document_type: Literal["invoice", "receipt", "contract"] = Field(
        description="What kind of document is this?"
    )
    is_paid: bool = Field(description="Has the amount been paid in full?")


triage = extract(Triage, "openai-decisions:gpt-6-luna", Path("receipt.png"))
```

Each field becomes one question, phrased by its description. A `bool` is `True` when
the probability of yes is at least 0.5. Other field types raise `TypeError`, and a
missing or invalid answer raises `SchemaValidationError` without a retry. The
`openai:` and `google:` prefixes keep their general APIs: use the prefixes above, or
`Model("google:gemini-decision-1", api="gemini-decisions")`.

## Errors

Every operational error subclasses `ExtractionError` and chains the original
exception as `__cause__` when there is one.

| Exception | Raised when |
| --- | --- |
| `ModelError` | A model request fails, or the provider reports an HTTP or generation error. Exposes `status_code`, `code`, `provider`, `finish_reason`, and `native_finish_reason` when known. |
| `SchemaValidationError` | The output does not match the schema, was refused or truncated, or the response is malformed. |
| `ExtractionError` | A source URL cannot be fetched, or a provider key or Bedrock region is missing. |

Invalid configuration and unsupported media raise `ValueError`; an unsupported model
object or Decisions field type raises `TypeError`. Local file errors keep their
standard Python exceptions. `ProviderNotInstalledError` remains importable for
compatibility and is never raised.

Messages say what failed and end with a next step:

```text
Model provider returned HTTP 401.
Code: authentication_error
Details: Invalid API key.
Next step: Check the API key or authentication headers for the selected provider.
```

Messages include selected provider details rather than whole response bodies, with
known credentials redacted. Schema errors list at most five field paths and never
echo the invalid values. Chained exceptions keep full diagnostics and can contain
sensitive data, so handle them with care in logs. Match on exception classes and
`ModelError` attributes, not message text.

## Development

```bash
uv sync --dev
uv run pytest
```

Tests run offline. See [CONTRIBUTING.md](CONTRIBUTING.md) for the opt-in live
provider checks and the release process, and [SECURITY.md](SECURITY.md) for how
URL sources and credentials are handled.

Upgrading from 1.x: install plain `openextract`, replace Pydantic AI model objects
with `openextract.Model`, and wrap local file names in `Path(...)`, because plain
strings are now document text.

## License

MIT
