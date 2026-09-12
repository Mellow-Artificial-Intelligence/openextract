---
layout: page
title: ExtractBench
---

# ExtractBench (any model)

[`scripts/extractbench.py`](https://github.com/Mellow-Artificial-Intelligence/openextract/blob/main/scripts/extractbench.py)
runs [LlamaIndex ExtractBench](https://github.com/run-llama/ExtractBench) through
openextract. Pass any `pydantic-ai` model identifier; the script registers an
openextract pipeline, downloads the dataset if needed, runs inference, and
scores with ExtractBench's official unified value F1 and grounding metrics.

This is a **quality** benchmark (schema-guided extraction accuracy and
source grounding). It is separate from [`scripts/bench.py`](benchmarking.md),
which only measures local CPU overhead with the model call mocked out.

## Quick start

From the repository root, with provider credentials in `.env` (the same keys
the CLI and examples use):

```bash
# 6 documents — good for trying a model (cents on a hosted API)
uv run python scripts/extractbench.py --model openai:gpt-5 --test

# One length split
uv run python scripts/extractbench.py --model xai:grok-4.3 --group short

# Full benchmark (370 documents / 4,869 pages — metered API usage)
uv run python scripts/extractbench.py --model anthropic:claude-sonnet-4
```

`--model` follows the same `provider:id` convention as `extract()`, for
example `openai:gpt-5`, `google-gla:gemini-2.5-pro`, `ollama:llama3`, or
`openrouter:anthropic/claude-sonnet-4`. You can also set `OPENEXTRACT_MODEL`
instead of passing `--model`.

The first run creates `.extractbench/venv`, installs ExtractBench from GitHub,
and editable-installs this repo with provider extras. Later runs reuse that
environment. Override the git source with `OPENEXTRACT_EXTRACTBENCH_GIT`.

## What it costs

A full run is 370 documents. Start with `--test`. ExtractBench's own guidance
is that hosted VLMs typically cost on the order of tens of dollars for a full
run; specialized APIs and coding agents cost more. This wrapper records token
usage; pass `--input-price-per-1m` and `--output-price-per-1m` if you want
ExtractBench to compute `cost_usd`. Reported usage and `cost_usd` cover the
successful attempt only; tokens spent on failed attempts that were retried
(`--max-retries`) are not included.

## Output

Results land under `.extractbench/output/<pipeline_name>/` (pipeline names are
`openextract_<slug>` of the model id, or `--pipeline-name`). After a run:

```bash
uv run python scripts/extractbench.py --serve openextract_openai_gpt_5
```

## Other commands

```bash
uv run python scripts/extractbench.py --install          # bootstrap only
uv run python scripts/extractbench.py --download-only --test
uv run python scripts/extractbench.py --status --test
uv run python scripts/extractbench.py --model openai:gpt-5 --skip-inference
```

`--max-concurrent` defaults to 4 (ExtractBench's own default is 20). Raise it
if your provider quota allows. `--max-input-bytes` defaults to 500 MiB so long
scans are not rejected by openextract's 50 MiB library default.

## Grounding

The runner asks the model for per-field citations by default (`--cite`, disable
with `--no-cite`). Raw `citations` in the result JSON use
`Citation.as_dict()` (`field`, `quote`, `page`, `bbox` as four floats or
`null`). ExtractBench scoring still uses `as_field_citation()` / `FieldCitation`:

| ExtractBench field | openextract `Citation` | When it scores |
| --- | --- | --- |
| `field_path` | `field` | Always, when a citation is emitted |
| `page` | `page` | **Page-level** grounding, stamped from the local parse index when the quote is found |
| `reference_text` | `quote` | Evidence text; not required to score. Short quotes are kept |
| `bbox` | `bbox` | **Word-level** grounding (IoU 0.5), only when the local parser has a matching word/span box |

The runner **parses PDFs locally** (pypdfium2, `openextract[pdf]` / `openextract[all]`)
and feeds page-indexed text (`--- Page N ---`) to the model before extraction.
Long documents (and oversized pages) are split into windows under a 12k-character
/ 1-page budget, extracted with bounded concurrency (`--window-concurrency`, default 4),
and merged — the whole PDF is not dumped as one prompt. `--timeout` (default 240s)
is per window; the document wall budget is
`timeout × (ceil(windows / concurrency) + 2)` so GLM variance (pueblo
past 960s) does not kill a finishing extract. When the pool budget
hits, finished windows and any in-flight that complete quickly are
merged into a result instead of discarded. ExtractBench's per-file
timeout is raised to at least that budget for a long document (never a
hard 1800s below the pool wait) and per-file timeout retries are off,
so Veralto / long are not run 3×1800s. A pool timeout does not wait
for cancelled workers (that hang is how long reached the 96-window file cap
with no result.json). Hung windows are a single attempt; empty-tool-call /
output-retry failures are retried at the window (`output_retries=3` plus
one extra attempt). Request timeouts and token-limit errors are not
retried at file level. Scanned / empty-text PDFs are rendered to compact grayscale page images
(long edge ≤ 768px) locally and are
never uploaded for provider document-parse. Citations are collected from
every window before reduce. One-page windows remap model `page=1` / missing
pages onto the document page in that window. When a window
returns values but no cites — or a cite whose quote is paraphrased, or whose
hinted page is a false match — pages are located from the extracted values in
the local parse (including numeric and punctuation variants such as
`$1,234.00` vs `1234`). Boxes are never invented and are never
taken from the model: if a parser span matches the quote or value (exact,
then simple fuzzy), a normalized COCO `[x, y, width, height]` in `[0, 1]` is
attached, preferring the value span over a long quote box; if
nothing matches, `bbox` is omitted. Model field paths that drop array
indexes (`qty` vs `lines[0].qty`) are rebound onto the extracted output.
Citations without a page cannot become
`FieldCitation` (ExtractBench requires `page >= 1`) and are dropped at the
mapping step.

`--no-cite` still runs parse-then-extract but does not ask for or emit
`field_citations`. Token `usage` and `cost_usd` come from the provider usage
object captured by `extract_with_usage` (OpenRouter native usage is requested
via `extra_body` as well as `openrouter_usage`).

`--test` is 6 documents and is the right first run (cents on a hosted API). A
full run is 370 documents / 4,869 pages and is metered API usage.

## Latest smoke

The last published 6-document `--test` (OpenRouter `z-ai/glm-5.3-flash`,
citations on, package `0.13.0`) finished 6/6 inference with successful-only
page F1 of 0.24. That is a **pre-fix** smoke from before the page/word
grounding work shipped in `0.13.1` ([#220](https://github.com/Mellow-Artificial-Intelligence/openextract/pull/220)):
window page remap, value backfill when the quote is paraphrased or the hinted
page is wrong, and parser boxes for numeric/punctuation variants. Those
numbers are not a new run.

Package `0.13.1+` includes those fixes. A fresh 6-doc smoke on that tree has
not been published yet. The maintainer gate (6/6 finish **and** successful-only
page F1 > 0.37) is still unmet until that run is posted. The full 370-document
run has not been run. Listing this pipeline on the official ExtractBench
leaderboard is a later change in
[run-llama/ExtractBench](https://github.com/run-llama/ExtractBench), not this
repository.

How to run: [Quick start](#quick-start).

## Dataset license

ExtractBench documents come from public records (see the upstream README).
The harness clones ExtractBench into `.extractbench/` (gitignored) and does
not vendor those files in this repository.
