# openextract (Rust)

A Rust port and standalone single-binary CLI of [openextract](../README.md) (`v2.4.0`), built so AI agents and Rust programs can download one self-contained binary and extract validated structured JSON from documents, images, audio, video, URLs, and text using LLMs.

Every provider is called over plain HTTP (using `rustls`, with zero provider SDKs or OpenSSL system dependencies) with full 1:1 parity across all 12 provider APIs, bounded retries, local JSON Schema validation, and credential redaction.

Source URLs are fetched by a separate client that does not send model credentials. It follows at most three redirects, and only to the same host and port (an `http` to `https` upgrade on the default ports is allowed). Downloads are capped at 50 MiB: `Content-Length` is checked before the body is read, and the body is streamed so a missing or short header cannot bypass the cap.

The crate version in `Cargo.toml` tracks the Python package version in `pyproject.toml`. CI compares them and fails when they differ. The minimum supported Rust version is 1.88 (`let` chains; 1.85 does not compile this crate or its `wiremock` dev-dependency). `rust-toolchain.toml` pins the toolchain used for formatting and Clippy.

## Build & Install

```bash
cd rust
cargo build --release
# Binary is at: rust/target/release/openextract
```

Or install locally into `~/.cargo/bin`:

```bash
cargo install --path rust
```

## CLI Usage (for AI Agents & Shell Workflows)

Stdout always emits **valid JSON only**; diagnostics and `Next step:` hints go to stderr.

Exit codes:
- `0`: Success
- `2`: Invalid CLI usage, config, schema, or unsupported media type
- `3`: Missing API key / Bedrock region, local file I/O error, or source URL fetch failure
- `4`: Provider HTTP, network, or generation error
- `5`: Schema validation failure, refusal, truncation, or malformed provider output

### Discover available providers and configured keys

```bash
openextract --list-providers
```

Returns a JSON array showing every built-in provider prefix, HTTP API, supported input formats, and whether its API key environment variable is currently set (without printing any secrets).

### Quick extraction with `--field` shorthand

Syntax: `-f NAME[:TYPE][?][=DESCRIPTION]`

Supported types: `string`, `number`, `integer`, `boolean`, `date`, `time`, `datetime`, `email`, `uri`, `uuid`, `enum(a,b,c)`, and arrays `T[]`. Append `?` for nullable fields (still required in strict JSON Schema so the model returns `null` rather than inventing values).

```bash
# Local PDF invoice
openextract -m openai:gpt-6.1-sol \
  -f 'vendor:string=Seller, not buyer.' \
  -f 'total:number=Final amount due, including tax.' \
  -f 'currency:string?=Three-letter currency code.' \
  invoice.pdf

# Pipe text or binary bytes from stdin
cat email.txt | openextract -m google:gemini-2.5-flash \
  -f sender:string \
  -f action_items:string[]

# Literal text (never treated as a file path)
openextract -m anthropic:claude-sonnet-5-5 \
  -f total:number \
  --text "Total due: 120.00 USD"

# Preview the generated JSON Schema without calling any model
openextract --print-schema -f vendor:string -f total:number -f currency:string?
```

### Full JSON Schema (`--schema`)

Pass an inline JSON object, a file path, `@path`, or `-` (stdin):

```bash
openextract -m anthropic:claude-sonnet-5-5 \
  --output-mode json_schema \
  --schema invoice.schema.json \
  https://example.com/2291.pdf
```

### Specialized APIs

```bash
# Closed-set Decisions API
openextract -m openai-decisions:gpt-6-luna \
  -f 'document_type:enum(invoice,receipt,contract)=What kind of document is this?' \
  -f 'is_paid:bool=Has the amount been paid in full?' \
  receipt.png

# OpenRouter Extractions with spans & metadata
openextract -m openrouter:cloudflare/clairic \
  --include-metadata \
  -f invoice_number:string \
  -f total:number \
  -f due_date:date? \
  invoice.pdf

# Gemini spreadsheet selection
openextract -m google:gemini-4-hydrogen \
  --settings '{"spreadsheetMetadata":{"sheetIndex":2,"visibleCells":"A1:H50","contentLayers":["DATA","FORMULAS"]}}' \
  -f total:number \
  financial_model.xlsx

# Mistral OCR document annotations
openextract -m mistral:mistral-ocr-latest \
  --api mistral-ocr \
  --settings '{"pages":[0,1]}' \
  -f vendor:string -f total:number \
  scanned-invoice.pdf
```

## Rust Library Usage

Add `openextract` to your `Cargo.toml` (disable default features to omit `clap` if you only need the library):

```toml
[dependencies]
openextract = { path = "./rust", default-features = false }
```

### Strongly-typed extraction (`schemars` + `serde`)

```rust
use openextract::extract;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct Invoice {
    /// Seller, not buyer.
    vendor: String,
    /// Final amount due, including tax.
    total: f64,
    /// Three-letter currency code, or null if unclear.
    currency: Option<String>,
}

#[tokio::main]
async fn main() -> openextract::Result<()> {
    let source = "Seller: Acme Paper Co.\nTotal due: 120.00 USD\n";
    let invoice: Invoice = extract("openai:gpt-6.1-sol", source, None).await?;
    println!("{invoice:?}");
    Ok(())
}
```

### Dynamic JSON Schema extraction

```rust
use openextract::{ExtractOptions, Source, extract_value};
use serde_json::json;

let schema = json!({
    "type": "object",
    "properties": {
        "vendor": {"type": "string"},
        "total": {"type": "number"}
    },
    "required": ["vendor", "total"]
});
let data = extract_value(
    &schema,
    "openai:gpt-6.1-sol",
    Source::Path("invoice.pdf".into()),
    ExtractOptions::new(),
).await?;
```

## Development

```bash
cd rust
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```
