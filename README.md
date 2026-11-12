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

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
All model requests use HTTPX directly. Pydantic is used for schemas and validation;
Pydantic AI and provider SDKs are not dependencies. No provider extras are needed.

Run `uv add` in your project directory. If you're starting from an empty
directory, run `uv init --python 3.12` first.

```bash
uv add openextract
```

Set `OPENAI_API_KEY` to your OpenAI API key in the same terminal. Your account
needs access to `gpt-5`; running the example makes a paid provider request.

```bash
export OPENAI_API_KEY='your-api-key'  # macOS / Linux
```

On PowerShell, use `$env:OPENAI_API_KEY = 'your-api-key'` instead.

Save the following as `first_extraction.py`. The fictional invoice is included
as plain text, so there is no document to download or missing file to create.

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

invoice = extract(Invoice, "openai:gpt-5", source)
print(invoice.vendor)  # str: Acme Paper Co.
print(invoice.total)  # Decimal: 120.00
print(invoice.currency)  # str | None: USD
```

```bash
uv run python first_extraction.py
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
checkout, after setting the key, run `uv run python examples/basic.py`;
it finds the fixture automatically. An optional path or URL argument lets you use
your own document. See [example checks](examples/README.md) for offline validation
and the live-check record.

Continue with [fine-detail controls](examples/fine_details.md): field descriptions,
`Decimal` totals, nullable currency, validators, interpretation instructions, and
media-type overrides, using the same `Invoice` result type.

## Usage

```python
extract(schema, model, source, instructions=None, *, media_type=None, client=None, include_metadata=False)
await extract_async(schema, model, source, instructions=None, *, media_type=None, client=None, include_metadata=False)
```

- **`source`** accepts plain text, a `pathlib.Path` (or another `os.PathLike`) for a local file, an `http(s)://` URL, or raw `bytes`. Strings are text unless they start with `http://` or `https://`. Text is encoded as UTF-8 automatically; use `Path("invoice.pdf")` to read a local file.
- **`model`** is a supported `"provider:model"` string (`"anthropic:claude-sonnet-5"`, `"google-gla:gemini-2.5-pro"`, `"ollama:llama3"`, ...) or an `openextract.Model` configuration.
- **`instructions`** are optional extra guidance for the model.
- **`media_type`** defaults to `text/plain` for plain strings. For files, URLs, and bytes, openextract checks the file extension, then the URL's `Content-Type`, then the leading bytes (PDF, PNG, JPEG, GIF, WebP, MP3, WAV, FLAC, OGG, MP4, UTF-8 text). Pass it to override the result.
- **`client`** is an optional `httpx.Client` for `extract`, or `httpx.AsyncClient` for `extract_async`. It is used only for model requests and remains open. Without it, openextract creates and closes a client for each extraction. Source URLs are fetched separately without model credentials.
- **`include_metadata`** returns an [`ExtractionResult`](#openrouter-extractions) when enabled for OpenRouter Extractions.

Media detection does not mean every model accepts every format. Choose a model
that supports your input and schema.

```python
from pathlib import Path

extract(Invoice, "openai:gpt-5", source="Total due: 120.00")  # plain text
extract(Invoice, "openai:gpt-5", source=Path("invoices") / "2291.pdf")  # pathlib.Path
extract(Invoice, "anthropic:claude-sonnet-5", source="https://example.com/2291.pdf")  # URL
extract(Invoice, "openai:gpt-5", source=pdf_bytes, instructions="Amounts are in EUR.")  # bytes
```

For provider generation settings, see [configured models](examples/configured_models.md)
for a complete example, sync/async checks, and provider compatibility limits.

## Providers

Adapters encode requests for these APIs. The selected model must support the
input type and structured output or tool calling used by its adapter.

| Prefix | HTTP API | API key environment variable | Inputs encoded by the adapter |
| --- | --- | --- | --- |
| `openai`, `openai-responses` | [Responses](https://developers.openai.com/api/docs/guides/structured-outputs) | `OPENAI_API_KEY` | Text, images, PDF, Word, RTF, ODT, PowerPoint, Excel |
| `openai-chat` | Chat Completions, tool or JSON Schema output | `OPENAI_API_KEY` | Text, images, PDF, MP3, WAV |
| `openrouter` | Chat Completions, [Extractions](#openrouter-extractions), or [Decisions](#decision-models), selected by model | `OPENROUTER_API_KEY` | Depends on the API and model |
| `groq`, `mistral`, `xai`, `cerebras`, `huggingface` | Chat Completions, tool or JSON Schema output | `GROQ_API_KEY`, `MISTRAL_API_KEY`, `XAI_API_KEY`, `CEREBRAS_API_KEY`, `HF_TOKEN`, respectively | Chat content supported by the endpoint and model |
| `anthropic` | Messages with a forced extraction tool | `ANTHROPIC_API_KEY` | Text, PNG, JPEG, GIF, WebP, PDF |
| `google`, `google-gla`, `gemini` | Gemini `generateContent` with JSON Schema | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Text, images, PDF, audio, video |
| `ollama` | Local `/api/chat` with JSON Schema | None | Text, images |
| `cohere` | V2 Chat with JSON Schema | `CO_API_KEY` | Text, images |
| `bedrock` | Converse with a forced extraction tool | `AWS_BEARER_TOKEN_BEDROCK` | Text, PNG, JPEG, GIF, WebP, PDF |

The Chat Completions adapter encodes text, images, PDF, MP3, and WAV; compatible
services do not necessarily accept every format. Unsupported adapter input types
raise `ValueError` before a model request. Model-specific restrictions are reported
by the provider as `ModelError`. Recognizing a file type does not imply every
provider can process it. For example, Responses does not accept audio or video;
use Gemini for those inputs, or an audio-capable Chat Completions model for MP3/WAV.

Responses and Chat Completions in JSON Schema mode use strict JSON Schema:
all declared fields, including fields with
defaults, are requested, and fixed objects disallow additional properties.
Open-ended dictionaries are rejected; select `Model("openai:gpt-5", api="chat")`
for tool-based output instead. Other schema restrictions depend on the provider.
The original Pydantic model always performs the final validation, including custom
validators and constraints. Tagged unions are adapted to the provider's strict
schema format. General adapters use JSON validation consistently, so strict
`date` and `Decimal` fields also work with decoded tool inputs.

Bedrock requires `AWS_REGION` (or `AWS_DEFAULT_REGION`) unless `base_url` is
provided. Authentication uses a [Bedrock bearer token](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys-use.html).
AWS credential discovery, profiles, role assumption, and automatic SigV4 signing
are not implemented. You can supply your own authenticated HTTPX client with
`Model(..., api_key="")` to disable environment-key lookup and the key requirement.
Local Ollama supports structured output; Ollama Cloud currently does not.

## Custom endpoints and settings

Use `Model` to configure an existing provider or any endpoint implementing a
supported API. Custom provider names require both `api` and `base_url`.

```python
import os

import httpx
from openextract import Model, extract

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

with httpx.Client() as client:
    invoice = extract(Invoice, model, "./invoice.pdf", client=client)
```

`api` selects `responses`, `chat`, `anthropic`, `gemini`, `ollama`, `cohere`,
`bedrock`, `openai-decisions`, `openrouter-decisions`, or `openrouter-extractions`.
`base_url` includes the API version. Use a base ending in `/v1` for OpenAI Decisions
and `/api/alpha` for OpenRouter Decisions or Extractions. Credentials go in `api_key` or
`headers`, not the URL. Custom headers take precedence over generated headers.
For keyless custom services, omit `api_key`.

Explicit configuration takes precedence over environment variables. Built-in
providers also read their base URL override: `OPENAI_BASE_URL`,
`ANTHROPIC_BASE_URL`, `GOOGLE_BASE_URL`, `OLLAMA_BASE_URL`,
`OPENROUTER_BASE_URL`, `GROQ_BASE_URL`, `MISTRAL_BASE_URL`, `XAI_BASE_URL`,
`CEREBRAS_BASE_URL`, `HF_BASE_URL`, `COHERE_BASE_URL`, or `BEDROCK_BASE_URL`.
For OpenRouter Decisions and Extractions, openextract translates an
`OPENROUTER_BASE_URL` ending in `/api/v1` to `/api/alpha`.

Chat Completions defaults to tool calling. For compatible endpoints, select
`output_mode="json_schema"` to request JSON text using `response_format`, with
the schema generated from your Pydantic model. For example, use Gemini through
[OpenRouter structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs):

```python
model = Model(
    "openrouter:google/gemini-2.5-flash",
    output_mode="json_schema",
    settings={"provider": {"require_parameters": True}},
)
invoice = extract(Invoice, model, "./invoice.pdf")
```

Set `OPENROUTER_API_KEY` in the environment. `require_parameters` restricts
OpenRouter routing to endpoints supporting the requested parameters. In this
mode, openextract sends the JSON Schema instead of extraction tools, reads JSON
from the response's message content, and validates it with the original Pydantic
model. `output_mode="tool"` explicitly selects tool calling. This option is only
configurable for Chat Completions; other adapters retain their own output formats.
The model and endpoint must support the selected mode.

`settings` uses the provider's native parameter names. Gemini settings go inside
`generationConfig`; Ollama sampling options go inside `settings["options"]`;
Bedrock inference options go inside `settings["inferenceConfig"]`. Anthropic's
default `max_tokens` is 4096 and can be overridden. Request structure and output
schema fields cannot be replaced. Decisions adapters do not accept settings.
Extraction does not stream responses.

By default, transport failures and HTTP 408, 409, 429, 500, 502, 503, and 504 are
retried twice, using exponential backoff or `Retry-After` capped at 30 seconds.
Other HTTP errors, redirects, and provider-reported generation failures fail
immediately. General extraction retries a
Pydantic validation failure once with feedback that omits raw input values.
Refusals, incomplete generations, malformed response envelopes, and invalid
Decisions answers fail immediately. The HTTP retry budget applies to each
validation attempt. Set both retry counts to zero for a single model request.

## OpenRouter Extractions

Set `OPENROUTER_API_KEY` and pass the model with the usual `openrouter:` prefix.
OpenExtract selects [OpenRouter's alpha Extractions API](https://openrouter.ai/docs/guides/community/extractions)
for `cloudflare/clairic`, `inception/mercury-decide`, `levanto/sage-2`, and
`scaledown/extract`. Use `Model("openrouter:provider/model", api="openrouter-extractions")`
to select it explicitly for another model.

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

Both sync and async calls return the validated Pydantic instance by default.
With `include_metadata=True`, `ExtractionResult.data` holds that instance,
`answers` holds the provider's original answer objects keyed by Python field name,
and `metadata` holds other response fields such as usage and truncation status.
Confidence, source spans, null probabilities, and distributions vary by model
and field. The default call raises `SchemaValidationError` if OpenRouter reports
that it read only part of the source. With metadata enabled, check
`result.metadata.get("truncated")` for partial results.

The adapter infers fields from `str`, `int`, `float`, `Decimal`, `date`, `time`,
`datetime`, their nullable forms, and lists of text, numbers, or dates. It sends
`Field(description=...)` and call-level instructions with each field while keeping
the source text intact for spans. Use `OpenRouterField` inside `Annotated` to set
native options or specify a type explicitly. For example,
`Annotated[float, OpenRouterField(type="estimate", min=0, max=90)]` extracts an
estimate's value as a float and keeps its distribution in the answer metadata.
Pydantic applies constraints and validators to the returned values. Use a
structured-output provider from the table above for nested records.

Each model has its own input and field limits. Clairic reads text and images and
supports all native field types. ScaleDown Extract reads PDFs and returns text
or lists of text. The library sends one source per call. Pass native request
options such as `reference_date` and `provider` through `Model(settings=...)`.

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
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class Triage(BaseModel):
    document_type: Literal["invoice", "receipt", "contract"] = Field(
        description="What kind of document is this?"
    )
    is_paid: bool = Field(description="Has the amount been paid in full?")


triage = extract(Triage, "openai-decisions:gpt-6-luna", source=Path("receipt.png"))
# Or: extract(Triage, "openrouter:typesafe/jev-1.14", source=Path("receipt.png"))
```

Each field becomes one question, phrased by its `description`. A `bool` field is `True` when the probability of yes is at least 0.5. Other field types raise a `TypeError`. Missing, invalid, or refused OpenAI answers raise `SchemaValidationError`.

## Errors

Operational errors subclass `ExtractionError`. The original exception is chained
as `__cause__` when available. `ModelError` includes selected provider error
messages and upstream details instead of dumping the entire response body.
Known credentials are redacted from those details; provider messages may still
contain input data. The chained `httpx.HTTPStatusError` retains the response for
HTTP failures.

`ModelError` exposes `status_code`, `code`, `provider`, `finish_reason`, and
`native_finish_reason` when available. An HTTP 200 response with an error envelope
or a Chat Completions `finish_reason="error"` raises `ModelError`, including the
native reason such as `MALFORMED_FUNCTION_CALL`. Invalid generated data, refusals,
and truncated output remain `SchemaValidationError`.

Messages explain what failed and end with a `Next step:` suggestion. For example:

```text
Model provider returned HTTP 401.
Code: authentication_error
Details: Invalid API key.
Next step: Check the API key or authentication headers for the selected provider.
```

Schema errors list field paths and validation reasons, with at most five details.
Raw invalid values are omitted; custom `ValueError`/assertion messages and invalid
union tags are summarized because they can embed input data. Request and validation
failures include attempt counts. Chained exceptions retain full diagnostics and
can contain sensitive data. For programmatic handling, use exception classes and
`ModelError` attributes rather than parsing message text.

| Exception | Raised when |
| --- | --- |
| `ModelError` | A model request fails, or the provider reports an HTTP or generation error. |
| `SchemaValidationError` | The model's output doesn't match the schema. |
| `ProviderNotInstalledError` | Deprecated compatibility export; never raised by HTTP adapters. |
| `ExtractionError` | A URL can't be fetched. |

Invalid configuration or unsupported media raises `ValueError`; unsupported model
objects or Decisions field types raise `TypeError`. Local file errors retain their
standard Python exceptions. If openextract can't infer the media type, it raises a
`ValueError` and asks you to pass `media_type`.

## Migrating from Pydantic AI

Install plain `openextract`; provider extras have been removed. Replace configured
Pydantic AI `Model` objects with `openextract.Model`. Supported model strings and
the Pydantic schema/validated-result interface remain available. Provider support
is defined by the adapters above rather than by Pydantic AI's model registry.
Vertex AI credential handling, SDK-specific configuration, and Logfire integration
are not bundled. HTTPX clients provide the transport customization boundary.

Plain string sources are now document text, with HTTP(S) URLs still fetched.
Wrap local file names in `Path(...)`: replace `source="invoice.pdf"` with
`source=Path("invoice.pdf")`. Missing `Path` inputs still raise `FileNotFoundError`.
Raw bytes remain supported.

## Tests

Default tests run offline and skip all live API calls. Saved synthetic PDF/image
fixtures cover single-page, multi-page, and scanned invoices. Run the provider
matrix explicitly with:

```bash
uv run --env-file .env pytest --live -m live --no-cov -v
```

See [Contributing](CONTRIBUTING.md#live-provider-checks) for credentials, provider
filters, and the offline CI command.

## License

MIT
