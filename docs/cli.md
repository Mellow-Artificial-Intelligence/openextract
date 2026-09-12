---
layout: page
title: CLI contracts
---

# CLI stdout, stderr, and exit codes

This page is the contract for the `openextract` command. Successful results go
to **stdout**. Errors, warnings, and progress go to **stderr**. Exit codes are
stable for automation.

## Streams

| Stream | Contents |
| ------ | -------- |
| stdout | Successful extraction payloads (`json`, `jsonl`, or `repr`) |
| stderr | `error: ...` messages, a `warning: ...` line for partial batch failures, and `progress: ...` lines when `--progress` is set |

Never parse stderr for successful results. Never treat stdout as empty when
exit code `7` is returned — the batch output is still written.

## Exit codes

| Code | Meaning | Typical cause |
| ---- | ------- | ------------- |
| `0` | Success | Single-file or full-batch success |
| `1` | Usage / setup error | Missing or bad `--schema` / `--model`, stdin without `--media-type`, invalid manifest, invalid concurrency/retry/size/swarm options, unloadable `--agent`, argparse failures |
| `2` | URL fetch error | `UrlFetchError` (network failure, HTTP error, SSRF refusal) |
| `3` | Schema validation error | `SchemaValidationError` |
| `4` | Model API error | `ModelError` |
| `5` | Other extraction error | `InputTooLargeError` and other `ExtractionError` subclasses |
| `6` | Missing provider SDK | `ProviderNotInstalledError` |
| `7` | Partial batch failure | `--continue-on-error` with one or more per-item failures |
| `8` | Remote agent failure | `RemoteAgentError` from a `--agent` / `--agents` HTTP endpoint |
| `130` | Interrupted | Ctrl-C / SIGINT; outstanding batch work is cancelled |
| `141` | Broken pipe | stdout closed by the consumer (e.g. `head`); exits silently |

These mappings live in `src/openextract/_cli.py` and are covered by
`tests/test_cli.py`.

CLI option values are validated **before any model call**: invalid
`--max-concurrency`, `--max-retries`, `--retry-backoff`, `--retry-max-backoff`,
`--max-input-bytes`, or manifest contents exit `1` without contacting a
provider.

## Successful single-file output

```bash
openextract ./reports/q4.pdf \
  --schema mypkg.schemas:Invoice \
  --model xai:grok-4.3 \
  --output json
```

- Exit `0`.
- stdout: JSON object from `model_dump_json` (default `--output json`).
- With `--output repr`, stdout is `repr(payload)` instead of JSON.

## Successful batch output

```bash
openextract ./invoices/a.pdf ./invoices/b.pdf \
  --schema mypkg.schemas:Invoice \
  --model xai:grok-4.3 \
  --max-concurrency 8
```

- Exit `0` when every item succeeds.
- stdout: JSON array of per-item `model_dump()` objects, in input order.
- `--max-concurrency N` bounds in-flight extractions (default `5`); it must be
  a positive integer.

## JSONL output for large batches

```bash
openextract ./invoices/*.pdf \
  --schema mypkg.schemas:Invoice \
  --model xai:grok-4.3 \
  --output jsonl --continue-on-error
```

`--output jsonl` writes one JSON record per line as each input **completes**,
flushed immediately, so large batches emit useful output long before the batch
finishes. Records arrive in completion order; `index` is the zero-based input
position, so consumers can reorder or join back to their inputs.

```json
{"index": 1, "input": "./invoices/b.pdf", "result": {"total": 42.0}}
{"index": 0, "input": "./invoices/a.pdf", "error": "...", "error_type": "ModelError"}
```

- Success records: `{"index", "input", "result"}` (plus `"usage"` with
  `--usage`).
- Failure records (`--continue-on-error` only): `{"index", "input", "error",
  "error_type"}`.
- With `--usage`, a final `{"summary": {"inputs", "failed", "usage"}}` line
  carries aggregate usage; it is the only non-record line.
- A single input with `--output jsonl` uses the same batch semantics and record
  shapes.
- Without `--continue-on-error`, the first failure cancels outstanding work;
  records already written remain on stdout, the error goes to stderr, and the
  exit code maps as usual (`2`–`6`).

## Progress reporting

```bash
openextract ./invoices/*.pdf \
  --schema mypkg.schemas:Invoice \
  --model xai:grok-4.3 \
  --output jsonl --progress 2>progress.log
```

`--progress` writes one line per completed batch item to **stderr only**:

```
progress: 3/10 completed (1 failed): ./invoices/c.pdf
```

stdout stays machine-readable. Progress lines are human-oriented; do not parse
them. The flag is a no-op for single-input runs.

## Manifest input

```bash
openextract --manifest inputs.jsonl \
  --schema mypkg.schemas:Invoice \
  --model xai:grok-4.3 \
  --output jsonl
```

`--manifest FILE` reads inputs from a JSONL file instead of positional
arguments, so heterogeneous batches can set per-input media types:

```json
{"source": "./invoices/a.pdf", "media_type": "application/pdf", "name": "invoice-a"}
{"source": "https://example.com/report", "media_type": "text/html"}
{"source": "./notes.txt"}
```

- `source` (required): a path or `http(s)://` URL. Stdin (`-`) is not
  supported inside manifests.
- `media_type` (optional): per-item MIME type. Entries without one fall back
  to `--media-type`, then to inference.
- `name` (optional): safe display label used in JSONL records, progress lines,
  and error entries instead of the source.
- Blank lines are skipped; unknown keys are rejected.
- `--manifest` is mutually exclusive with positional inputs and always uses
  batch semantics (a one-entry manifest still emits a JSON array or JSONL
  records).
- Invalid manifests exit `1` with a `manifest line N: ...` error before any
  model call.

## `--usage` output

Single input:

```bash
openextract ./reports/q4.pdf \
  --schema mypkg.schemas:Invoice \
  --model xai:grok-4.3 \
  --usage
```

```json
{
  "result": { "...": "schema fields" },
  "usage": {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0
  }
}
```

Batches run through the richer result API and report per-item **and**
aggregate usage:

```bash
openextract ./invoices/a.pdf ./invoices/b.pdf \
  --schema mypkg.schemas:Invoice \
  --model xai:grok-4.3 \
  --usage
```

```json
{
  "results": [
    { "input": "./invoices/a.pdf", "result": { "...": "..." }, "usage": { "...": 0 } },
    { "input": "./invoices/b.pdf", "error": "...", "error_type": "ModelError" }
  ],
  "usage": { "input_tokens": 0, "output_tokens": 0, "total_tokens": 0 }
}
```

The aggregate sums successful items only. With `--output jsonl`, usage appears
on each success record and in the final `summary` line instead.

## `--cite` output

`--cite` asks the model for per-field source evidence, the same as
`cite=True` on the Python API. PDFs are parsed locally first when
`openextract[pdf]` is installed. Default is off; non-cite JSON shapes are
unchanged.

```bash
openextract ./reports/q4.pdf \
  --schema mypkg.schemas:Invoice \
  --model xai:grok-4.3 \
  --cite
```

```json
{
  "result": { "...": "schema fields" },
  "citations": [
    { "field": "vendor", "quote": "Acme", "page": 1 },
    { "field": "total", "quote": "12.50", "page": 1, "bbox": [0.1, 0.2, 0.3, 0.05] }
  ]
}
```

Each citation is `{field, quote, page}` with optional `bbox` (normalized COCO,
parser-backed only). `quote` / `page` may be `null`. Combine with `--usage` to
keep `{result, usage, citations}`.

- Batch JSON array: `[{ "result", "citations" }, ...]` in input order.
- JSONL success records add `"citations"`; failure records are unchanged.
- Swarms merge citations from successful agents.
- An injected Pydantic AI agent cannot be used with `--cite` (exit `1`, same
  `ValueError` as `Extractor(agent=..., cite=True)`).

## `--output json`, `--output jsonl`, and `--output repr`

| Flag | Behavior |
| ---- | -------- |
| `--output json` (default) | Pretty-printed JSON (`indent=2`), buffered until the run finishes. Single-file non-usage results use `model_dump_json`. Batch arrays are in input order. |
| `--output jsonl` | One compact JSON record per completed input, written incrementally in completion order. |
| `--output repr` | Python `repr(...)` of the same payload object as `json`. |

All formats write only to stdout on success.

## Stdin input

```bash
cat ./reports/q4.pdf | openextract - \
  --schema mypkg.schemas:Invoice \
  --model xai:grok-4.3 \
  --media-type application/pdf
```

- `-` reads raw bytes from stdin.
- Stdin is read in bounded chunks under the same input-size limit as files and URLs.
- `--media-type` is required.
- `-` cannot be combined with other input paths or manifests (exit `1`).

## `--continue-on-error` partial failures

```bash
openextract ./ok.pdf ./missing.pdf \
  --schema mypkg.schemas:Invoice \
  --model xai:grok-4.3 \
  --continue-on-error
```

- stdout still receives the full JSON array (or every JSONL record).
- Failed items appear as:

```json
{
  "input": "./missing.pdf",
  "error": "...",
  "error_type": "ModelError"
}
```

- Failures always preserve input identity: the `input` field is the positional
  argument as given, or the manifest `name`/`source`.
- stderr receives a warning: `warning: N of M input(s) failed; see output for details`.
- Exit `7` when any item failed; exit `0` when none failed.
- Without `--continue-on-error`, the first failure cancels outstanding work and
  maps to exit codes `2`–`6` / `1` as usual (no batch array; JSONL records
  already emitted remain).

## Cancellation and broken pipes

- **Ctrl-C / SIGINT**: outstanding batch work is cancelled and awaited, an
  `error: interrupted` line goes to stderr, and the exit code is `130`.
- **Broken pipe**: when the stdout consumer closes early (e.g.
  `openextract ... --output jsonl | head -5`), the CLI stops scheduling work,
  cancels what is in flight, and exits `141` without writing anything further.

## Retry policy

`--max-retries` enables retries for transient model failures only.
`--retry-backoff` controls exponential backoff with up to 25% additive jitter,
and `--retry-max-backoff` caps both calculated delays and provider
`Retry-After` values. Authentication, permission, and invalid-request failures
exit immediately with code `4`.

## Input size limit

`--max-input-bytes N` sets the maximum bytes loaded for each input. Without the
flag, the CLI uses `OPENEXTRACT_MAX_INPUT_BYTES` or the 50 MiB default. Values
must be positive integers. Oversized inputs fail before a model call and exit
`5`; URL bodies and stdin remain bounded even without a reliable length header.

## Swarms and agents

`--swarm N` runs N copies of one model over a single input. `--models a,b`
runs one agent per model. `--reduce` folds the outputs: `merge` (default),
`vote`, or `first`.

```bash
openextract ./invoices/q4.pdf \
  --schema mypkg.schemas:Invoice \
  --models openai:gpt-5,anthropic:claude-opus-4-8 \
  --reduce vote
```

`--agent SPEC` extracts with an [agent](agents.md) — a directory, a Python
file, or `module:attribute`. `--agents SPEC,SPEC` runs several. An agent that
declares an `output_schema` makes `--schema` optional:

```bash
openextract ./invoices/q4.pdf --agent ./agents/invoices
```

- These flags apply to a **single** positional input; combining them with
  several input files, `--manifest`, or `--output jsonl` exits `1`.
- `--swarm` may not contradict the length of `--models`, and `--model` and
  `--models` are mutually exclusive.
- A lone `--agent` is not itself a swarm, but an agent with subagents (or a
  remote endpoint) still fans out and its outputs are reduced.
- With `--usage`, a swarm prints `{"result": ..., "usage": ..., "agents": N,
  "reduce": "..."}` instead of the single-call shape.

## Extraction styles

`--style direct` (default) sends the resolved media to the model in one shot.
`--style search` and `--style code` are text-only: search gives the model
sandboxed file tools (read, regex search, glob), and code lets it write Python
against a workspace copy of the document via Pydantic AI Harness. Missing extras
exit `6` (`ProviderNotInstalledError`). Non-text inputs raise `ValueError`
(exit `1`).

## Related docs

- [Guide](guide.md)
- [For agents](agents.md)
- [Troubleshooting](troubleshooting.md)
- [API reference](api-reference.md)
- [URL security model](https://github.com/Mellow-Artificial-Intelligence/openextract/blob/main/SECURITY.md#url-input-security-model)
