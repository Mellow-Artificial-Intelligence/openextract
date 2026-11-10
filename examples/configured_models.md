# Configure a model with its own settings

OpenExtract accepts an `openextract.Model` in place of a model string. Use
provider-native settings for generation behavior; keep field meanings in your
schema and interpretation guidance in `instructions`. The return value is still
your schema instance. All requests use HTTPX directly.

## A concrete provider setting

This example targets Anthropic's `claude-sonnet-4-5` through its Messages API and
sets an output-token limit. It requires Python 3.12+, `openextract`, and an
`ANTHROPIC_API_KEY` with access to that model. Running it against the provider
makes a paid request; the tests below use no real credentials or provider calls.

```bash
pip install openextract
export ANTHROPIC_API_KEY='your-api-key'
```

On PowerShell, set `$env:ANTHROPIC_API_KEY = 'your-api-key'` instead.

```python
from decimal import Decimal

from pydantic import BaseModel, Field

from openextract import Model, extract


class Invoice(BaseModel):
    vendor: str = Field(description="Seller, not buyer.")
    total: Decimal = Field(description="Final amount due, including tax.")
    currency: str | None = Field(description="Three-letter currency code, or null if unclear.")


model = Model("anthropic:claude-sonnet-4-5", settings={"max_tokens": 1024})
source = "Seller: Acme Paper Co.\nTotal due: 120.00 USD"

invoice = extract(Invoice, model, source)
print(invoice.total)
```

For async extraction, replace the final two lines with this block:

```python
import asyncio

from openextract import extract_async


async def main():
    invoice = await extract_async(Invoice, model, source)
    print(invoice.total)
    return invoice


invoice = asyncio.run(main())
```

Inside an existing event loop, use `await main()` instead of `asyncio.run`.

`max_tokens` caps generated output; it is not a document-size limit, a guarantee
of a complete response, or an accuracy setting. Too small a budget can truncate
structured output and cause validation failure. Provider limits still apply.
Settings use provider-native parameter names and are passed through without
SDK translation. Consult the provider's documentation for the chosen model.
Request structure, tools, and output schema fields are managed by OpenExtract
and cannot be replaced through settings. This example demonstrates only
`max_tokens`; it does not enable extended thinking or arbitrary sampling options.

## Routing and scope

`Model.name` uses the same `provider:model` format and routing as a model string.
For example, `openai:...` uses Responses and `ollama:...` uses native structured
output. Explicit `api` and `base_url` configure an alternative API or endpoint.
Use `Model("openai:...", api="chat")` for OpenAI Chat Completions.

Settings are applied in both sync and async calls. Construct another `Model`
for different defaults; there is no per-call `model_settings` argument.
`Model` stores configuration, not a network client. By default, extraction owns
and closes its HTTPX client; an explicitly supplied `client` remains caller-owned.

Decisions adapters reject generation settings and support narrower schemas and
media types. The free-form `Invoice` above is not a decision-model schema.
See the [HTTP adapter compatibility limits](model_compatibility.md) for migration
from Pydantic AI and for provider-specific settings placement.

## Offline verification

Run the focused offline checks in the locked development environment:

```bash
uv sync --dev --frozen
uv run --no-sync pytest tests/test_configured_models.py
```

The checks execute both published variants with a mocked provider boundary and
verify the configured token limit, model identifier, source, and typed result.
They also check configured defaults at the HTTP request boundary. Canned responses
verify configuration propagation, not extraction quality or live model access.
No live quality check was run.
