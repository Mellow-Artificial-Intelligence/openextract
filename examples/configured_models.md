# Configure a model with its own settings

OpenExtract accepts a configured Pydantic AI `Model` in place of a model string.
Use that model's typed settings for generation behavior; keep field meanings in
your schema and interpretation guidance in `instructions`. The return value is
still your schema instance. No OpenExtract settings object is needed.

## A concrete provider setting

This example targets Anthropic's `claude-sonnet-4-5` through its Messages API and
sets an output-token limit. It requires Python 3.12+, `openextract[anthropic]`, and
an `ANTHROPIC_API_KEY` with access to that model. Use the verified dependency pair
below: some combinations allowed by the broader package constraints fail before
a request (see the [compatibility record](model_compatibility.md)). Running it against the provider
makes a paid request; the tests below use no real credentials or provider calls.

```bash
pip install 'openextract[anthropic]' 'pydantic-ai-slim==2.56.0' 'anthropic==1.13.0'
export ANTHROPIC_API_KEY='your-api-key'
```

On PowerShell, set `$env:ANTHROPIC_API_KEY = 'your-api-key'` instead.

```python
from decimal import Decimal

from pydantic import BaseModel, Field
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings

from openextract import extract


class Invoice(BaseModel):
    vendor: str = Field(description="Seller, not buyer.")
    total: Decimal = Field(description="Final amount due, including tax.")
    currency: str | None = Field(description="Three-letter currency code, or null if unclear.")


settings = AnthropicModelSettings(max_tokens=1024)
model = AnthropicModel("claude-sonnet-4-5", settings=settings)
source = b"Seller: Acme Paper Co.\nTotal due: 120.00 USD"

invoice = extract(Invoice, model, source)
print(invoice.total)
```

For async extraction, replace the final two lines with this block (run either
version, not both sequentially with the same network client across event loops):

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
Do not assume every setting supported by Pydantic AI is accepted by every model:
for example, [Claude Sonnet 5 rejects `temperature`, `top_p`, and `top_k`](https://platform.claude.com/docs/en/models/sonnet-5/overview).
This example deliberately demonstrates only `max_tokens`; it does not enable
extended thinking or claim support for arbitrary sampling settings. Consult the
[typed Anthropic settings reference](https://pydantic.dev/docs/ai/api/models/anthropic/)
for the installed dependency version and the provider's model documentation.

## Routing and scope

A configured model chooses its own provider and endpoint. OpenExtract passes the
object to Pydantic AI's Agent unchanged, which uses the model's defaults in both
sync and async requests. For a different default, construct another configured
model; OpenExtract has no per-call `model_settings` argument.

String routing is separate: `"openai:..."` is rewritten to OpenAI Responses, while
`"ollama:..."` selects native structured output. Passing a configured model does
not apply those string rewrites: choose the appropriate model class yourself.
In particular, an OpenAI chat model object does not become a Responses model.

The explicit `openai-decisions:...` and supported OpenRouter Jev strings use
OpenExtract's separate HTTP adapters. Pydantic AI configured-model settings do
not configure those endpoints. They also have narrower schema and media support;
the free-form `Invoice` above is not a decision-model schema.

## Compatibility evidence

OpenExtract declares `pydantic-ai-slim>=1.37.0,<3`. Model-level `settings` and their
merge into request settings exist at the declared minimum, so the example does
not require a new OpenExtract keyword or a raised dependency floor. See the
[1.37.0 base model implementation](https://github.com/pydantic/pydantic-ai/blob/v1.37.0/pydantic_ai_slim/pydantic_ai/models/__init__.py)
and [Anthropic adapter](https://github.com/pydantic/pydantic-ai/blob/v1.37.0/pydantic_ai_slim/pydantic_ai/models/anthropic.py).

The [compatibility record and separate remedy proposal](model_compatibility.md)
include exact results and reproduction commands. The broad dependency range
does **not** guarantee that every allowed Anthropic SDK combination works. These are
representative version checks, not a claim that every intermediate or future
release below 3 has been executed. Provider-specific settings and transitive SDK
compatibility must still be checked when changing dependencies.

Run the focused offline checks in the locked development environment:

```bash
uv sync --dev --frozen
uv run --no-sync pytest tests/test_configured_models.py
```

The checks execute both published variants with a mocked provider boundary and
verify the configured token limit, model identifier, source, and typed result.
They also check default settings using a local recording model. Canned responses
verify configuration propagation, not extraction quality or live model access.
No live quality check was run.
