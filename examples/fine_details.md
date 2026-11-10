# Fine-detail controls

Start with the [first extraction](../README.md#first-extraction) for installation
and credentials. These examples use the same fictional invoice and general
`openai:gpt-5` route. Copy the Python blocks in order into one file; running it
makes three paid requests. Each section explains one control you can use on its
own. No configuration object is needed.

## Field meaning: seller, final total, and currency

Use Pydantic field descriptions to tell the model what each field means. This is
the quickstart's `Invoice` schema:

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
```

`vendor` asks for Acme Paper Co., not Northwind Studio. `total` asks for 120.00,
not the 100.00 subtotal. Descriptions guide the model; they do not validate that
it chose the right party or amount.

`Decimal` gives you decimal arithmetic on the returned amount. It does not force
two decimal places, round amounts, check totals against line items, or recover
precision lost before validation. For deterministic local values, construct it
from a string rather than a binary float:

```python
amount = Decimal("120.00")
assert amount + Decimal("0.10") == Decimal("120.10")
```

## Missing or ambiguous currency: required but nullable

`currency: str | None` with no default is **required but nullable**: the returned
object must include the field, but may explicitly supply `null` (Python `None`).
A missing currency should be represented as `null`, not an invented code. A bare
`$` is ambiguous without context; it does not establish USD.

These local examples illustrate the intended representation, not model output:

```python
missing_currency = Invoice.model_validate(
    {"vendor": "Acme Paper Co.", "total": "120.00", "currency": None}
)
assert missing_currency.currency is None


class InvoiceWithDefault(Invoice):
    currency: str | None = Field(
        default=None, description="Three-letter currency code, or null if unclear."
    )


omitted_currency = InvoiceWithDefault.model_validate(
    {"vendor": "Acme Paper Co.", "total": "120.00"}
)
assert omitted_currency.currency is None
```

Only the comparison schema has a default: local Pydantic validation allows its
currency key to be omitted. Omitting that key from `Invoice` raises a Pydantic
`ValidationError`. Defaults are not evidence about the source, and provider
structured-output requirements can differ from local Pydantic behavior.

## Validation: enforce a rule on the returned value

An ordinary Pydantic validator checks a value after type validation. For example,
this optional schema rejects malformed currency codes while still accepting
`None`:

```python
from pydantic import field_validator


class InvoiceWithCurrencyCheck(Invoice):
    @field_validator("currency")
    @classmethod
    def check_currency(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 3 or not value.isascii() or not value.isalpha() or not value.isupper()
        ):
            raise ValueError("Currency must be three uppercase ASCII letters or None")
        return value


checked = InvoiceWithCurrencyCheck.model_validate(
    {"vendor": "Acme Paper Co.", "total": "120.00", "currency": "USD"}
)
assert checked.currency == "USD"
```

Pass `InvoiceWithCurrencyCheck` as the schema to `extract` to apply this rule to
extraction results. It checks spelling shape, not ISO membership or whether USD
is actually stated in the invoice. A validator is not a model instruction;
keep the description to communicate the desired format. Local validation raises
`ValidationError`; extraction that cannot produce a valid result raises
OpenExtract's `SchemaValidationError` after the model's validation retries.

## Interpretation: instructions for this call

Keep reusable field meanings in the schema; use `instructions` for guidance
specific to this extraction. Here the source has a dollar sign but no country or
currency code. The intended currency is `None`, not a guessed USD:

```python
ambiguous_source = b"""INVOICE DEMO-002 (fictional sample)
Seller: Acme Paper Co.
Bill to: Northwind Studio
Total due: $120.00
"""

customized = extract(
    Invoice,
    "openai:gpt-5",
    ambiguous_source,
    instructions="Preserve names exactly. Do not guess missing or ambiguous currency.",
)
# Both the simple and customized calls return the same Invoice type.
assert isinstance(invoice, Invoice)
assert isinstance(customized, Invoice)
```

Instructions guide interpretation; they do not enforce correctness. The nullable
field permits an honest unknown, but cannot force the model to recognize one.
Review the returned currency against the source.

## Media type: explicitly identify the input

When you know the source format, `media_type` overrides inference from file
extensions, URL headers, and bytes. For known UTF-8 invoice text:

```python
explicit_media = extract(
    Invoice,
    "openai:gpt-5",
    source,
    media_type="text/plain",
)
assert isinstance(explicit_media, Invoice)
```

The override affects how the input is labeled, not how fields are interpreted.
Here it agrees with automatic detection; it can also correct a misleading file
extension or server header. It does not convert bytes, perform OCR, or add format
support to a model. Use the actual media type and a model that accepts it.

## Model limits and verification

These schemas are for general extraction. Decision routes
(`openai-decisions:...` and the supported Jev routes) accept narrower schemas:
`bool`, `Literal`, and `Enum` fields. Do not pass this `Invoice` to them:
free-form strings, `Decimal`, and nullable fields are not supported there.

Run the offline example checks from the repository root:

```bash
uv run --no-sync pytest tests/test_fine_details.py
```

The checks execute every Python block with provider requests disabled, using
canned responses through `httpx.MockTransport`. They check call wiring, typed
results, required/null/default behavior, and validator rejection. They do **not**
measure extraction quality. Expected values are read from the fictional sources;
no live provider quality check was run for this guide. See the
[manual live-check procedure](README.md#optional-live-quality-check) to assess
actual output. Schema validation never guarantees factual accuracy.
