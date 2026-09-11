"""Tests for scripts/extractbench.py (offline; no ExtractBench install required)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from openextract import Citation, ExtractionError, ModelError, SchemaValidationError, Usage

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import extractbench  # noqa: E402


def test_pipeline_name_for_model_slugs_provider_prefix():
    assert extractbench.pipeline_name_for_model("openai:gpt-5") == "openextract_openai_gpt_5"
    assert extractbench.pipeline_name_for_model("xai:grok-4.3") == "openextract_xai_grok_4_3"
    assert extractbench.pipeline_name_for_model("openai:gpt-5", "custom") == "custom"


def test_add_additional_properties_false_closes_nested_objects():
    schema = {
        "type": "object",
        "properties": {
            "vendor": {"type": "string"},
            "lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"qty": {"type": "integer"}},
                },
            },
        },
        "$defs": {"Addr": {"type": "object", "properties": {"city": {"type": "string"}}}},
    }
    out = extractbench.add_additional_properties_false(schema)
    assert out["additionalProperties"] is False
    assert out["properties"]["lines"]["items"]["additionalProperties"] is False
    assert out["$defs"]["Addr"]["additionalProperties"] is False
    assert schema.get("additionalProperties") is None


def test_inline_json_schema_defs_resolves_local_refs():
    schema = {
        "type": "object",
        "properties": {"addr": {"$ref": "#/$defs/Addr"}},
        "$defs": {
            "Addr": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            }
        },
    }
    out = extractbench.inline_json_schema_defs(schema)
    assert "$defs" not in out
    assert out["properties"]["addr"]["properties"]["city"]["type"] == "string"


def test_inline_json_schema_defs_leaves_recursive_refs():
    schema = {
        "type": "object",
        "properties": {"child": {"$ref": "#/$defs/Node"}},
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"child": {"$ref": "#/$defs/Node"}},
            }
        },
    }
    out = extractbench.inline_json_schema_defs(schema)
    assert out["properties"]["child"]["properties"]["child"]["$ref"] == "#/$defs/Node"


def test_as_extracted_dict_from_passthrough_model():
    output = extractbench.ExtractedDocument.model_validate({"vendor": "Acme", "total": 1.5})
    assert extractbench.as_extracted_dict(output) == {"vendor": "Acme", "total": 1.5}


def test_as_extracted_dict_rejects_non_objects():
    with pytest.raises(TypeError, match="Expected dict"):
        extractbench.as_extracted_dict("not-json")


def test_extract_document_with_citations_maps_extractbench_fields():
    schema = {
        "type": "object",
        "properties": {"vendor": {"type": "string"}, "total": {"type": "number"}},
        "required": ["vendor", "total"],
    }
    model = TestModel(
        custom_output_args={
            "output": {"vendor": "Acme", "total": 12.5},
            "citations": [
                {
                    "field": "vendor",
                    "quote": "Acme Co",
                    "page": 1,
                    "bbox": [0.1, 0.2, 0.3, 0.04],
                },
                {"field": "total", "quote": "12.50"},
            ],
        }
    )
    data, usage, citations = extractbench.extract_document_with_citations(
        b"%PDF-fixture",
        schema,
        model,
        media_type="application/pdf",
        max_retries=0,
        cite=True,
    )
    assert data == {"vendor": "Acme", "total": 12.5}
    assert usage.input_tokens >= 0
    assert citations[0] == Citation("vendor", "Acme Co", 1, None)
    mapped = extractbench.field_citations_for_extractbench(citations)
    data_only, _usage = extractbench.extract_document(
        b"%PDF-fixture",
        schema,
        model,
        media_type="application/pdf",
        max_retries=0,
        cite=True,
    )
    assert data_only == data
    assert mapped == [
        {
            "field_path": "vendor",
            "page": 1,
            "bbox": None,
            "reference_text": "Acme Co",
        }
    ]


def test_extract_document_parser_boxes_and_no_cite(tmp_path):
    from tests.pdf_fixture import synthetic_pdf

    schema = {
        "type": "object",
        "properties": {"vendor": {"type": "string"}},
        "required": ["vendor"],
    }
    pdf = synthetic_pdf("Acme Corp")
    path = tmp_path / "acme.pdf"
    path.write_bytes(pdf)
    cited_model = TestModel(
        custom_output_args={
            "output": {"vendor": "Acme"},
            "citations": [
                {"field": "vendor", "quote": "Acme Corp", "page": 3},
                {"field": "other", "quote": "not-in-document", "page": 1},
            ],
        }
    )
    data, usage, citations = extractbench.extract_document_with_citations(
        path,
        schema,
        cited_model,
        max_retries=0,
        cite=True,
    )
    assert data["vendor"] == "Acme"
    assert usage.input_tokens >= 0
    assert citations[0].page == 1
    assert citations[0].bbox is not None
    assert citations[1].bbox is None
    mapped = extractbench.field_citations_for_extractbench(citations)
    assert mapped[0]["page"] == 1
    assert mapped[0]["bbox"] == list(citations[0].bbox)

    plain, _usage = extractbench.extract_document(
        pdf,
        schema,
        TestModel(custom_output_args={"vendor": "Acme"}),
        media_type="application/pdf",
        max_retries=0,
        cite=False,
    )
    assert plain["vendor"] == "Acme"
    source, media_type, parsed = extractbench._parse_then_extract_source(pdf, "application/pdf")
    assert media_type == "text/plain"
    assert b"--- Page 1 ---" in source
    assert parsed is not None
    kept, kept_type, _parsed = extractbench._parse_then_extract_source(
        b"%PDF-not-a-pdf", "application/pdf"
    )
    assert kept == b"%PDF-not-a-pdf"
    assert kept_type == "application/pdf"


def test_extract_document_chunks_large_parse_and_keeps_boxes(monkeypatch, tmp_path):
    from tests.pdf_fixture import synthetic_pdf

    monkeypatch.setattr("openextract._parse.DEFAULT_PARSE_WINDOW_CHARS", 40)
    schema = {
        "type": "object",
        "properties": {"vendor": {"type": "string"}},
        "required": ["vendor"],
    }
    pdf = synthetic_pdf(pages=["AAAA " * 30, "Acme Corp " + "BBBB " * 30])
    path = tmp_path / "long.pdf"
    path.write_bytes(pdf)
    calls: list[int] = []
    original = extractbench.Extractor.extract_with_usage

    def _spy(self, source, *, media_type=None):
        payload = source if isinstance(source, bytes) else Path(source).read_bytes()
        calls.append(len(payload))
        return original(self, source, media_type=media_type)

    monkeypatch.setattr(extractbench.Extractor, "extract_with_usage", _spy)
    data, usage, citations = extractbench.extract_document_with_citations(
        path,
        schema,
        TestModel(
            custom_output_args={
                "output": {"vendor": "Acme Corp"},
                "citations": [{"field": "vendor", "quote": "Acme Corp", "page": 2}],
            }
        ),
        max_retries=0,
        cite=True,
    )
    assert data["vendor"] == "Acme Corp"
    assert usage.input_tokens >= 0
    assert len(calls) >= 2
    assert all(size < 500 for size in calls)
    assert citations[0].page == 2
    assert citations[0].bbox is not None
    sequential, sequential_usage, sequential_cites = extractbench.extract_document_with_citations(
        path,
        schema,
        TestModel(
            custom_output_args={
                "output": {"vendor": "Acme Corp"},
                "citations": [{"field": "vendor", "quote": "Acme Corp", "page": 2}],
            }
        ),
        max_retries=0,
        cite=True,
        window_concurrency=1,
        timeout=None,
    )
    assert sequential == data
    assert sequential_usage.input_tokens >= 0
    assert sequential_cites[0].page == 2


def test_unwrap_cited_document_flat_fallback():
    data, citations = extractbench._unwrap_cited_document(
        {"vendor": "Acme", "citations": [{"field": "vendor", "page": 1}]},
        cite=True,
    )
    assert data == {"vendor": "Acme"}
    assert citations[0].page == 1


def test_default_windows_split_short_slide_deck(monkeypatch, tmp_path):
    from tests.pdf_fixture import synthetic_pdf

    schema = {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    }
    pdf = synthetic_pdf(pages=[f"Slide {index}" for index in range(1, 21)])
    path = tmp_path / "deck.pdf"
    path.write_bytes(pdf)
    calls: list[int] = []
    original = extractbench.Extractor.extract_with_usage

    def _spy(self, source, *, media_type=None):
        payload = source if isinstance(source, bytes) else Path(source).read_bytes()
        calls.append(len(payload))
        return original(self, source, media_type=media_type)

    monkeypatch.setattr(extractbench.Extractor, "extract_with_usage", _spy)
    data, _usage, _citations = extractbench.extract_document_with_citations(
        path,
        schema,
        TestModel(custom_output_args={"output": {"title": "Slide 1"}, "citations": []}),
        max_retries=0,
        cite=True,
    )
    assert data["title"] == "Slide 1"
    assert len(calls) == 20


def test_window_reduce_backfills_field_citations_when_model_omits_them(tmp_path):
    from tests.pdf_fixture import synthetic_pdf

    schema = {
        "type": "object",
        "properties": {"vendor": {"type": "string"}, "total": {"type": "number"}},
        "required": ["vendor", "total"],
    }
    pdf = synthetic_pdf(pages=["Acme Corp", "Total 1,234"])
    path = tmp_path / "bianco.pdf"
    path.write_bytes(pdf)
    data, _usage, citations = extractbench.extract_document_with_citations(
        path,
        schema,
        TestModel(
            custom_output_args={
                "output": {"vendor": "Acme Corp", "total": 1234},
                "citations": [],
            }
        ),
        max_retries=0,
        cite=True,
    )
    assert data == {"vendor": "Acme Corp", "total": 1234}
    mapped = extractbench.field_citations_for_extractbench(citations)
    paths = {item["field_path"] for item in mapped}
    assert "vendor" in paths
    assert any(item["page"] >= 1 for item in mapped)


def test_extractbench_does_not_retry_timeouts():
    assert extractbench._extractbench_retryable(ModelError("rate limited", retryable=True))
    assert not extractbench._extractbench_retryable(ModelError("request timeout", retryable=True))
    assert not extractbench._extractbench_retryable(ModelError("token limit", retryable=False))
    assert not extractbench._extractbench_retryable(
        ModelError("Model output retry exhausted: Exceeded maximum output retries (1)")
    )


def test_window_pool_timeout_scales_with_batches():
    assert extractbench.window_pool_timeout(10, 4, 240.0) == 1200.0
    assert extractbench.window_pool_timeout(11, 4, 240.0) == 1200.0
    assert extractbench.window_pool_timeout(41, 4, 240.0) == 3120.0
    assert extractbench.window_pool_timeout(66, 4, 240.0) == 4560.0
    assert extractbench.window_pool_timeout(1, 4, 240.0) == 720.0
    assert extractbench.window_pool_timeout(0, 4, 240.0) == 240.0
    assert extractbench.window_pool_timeout(8, 4, None) is None


def test_window_pool_timeout_has_slack_for_pueblo():
    """Pueblo finished in 705.8s, then died at 720s and 960s; +1 batch was not enough."""
    pueblo = extractbench.window_pool_timeout(11, 4, 240.0)
    assert pueblo is not None
    assert pueblo > 960.0
    assert pueblo == 240.0 * (3 + extractbench.WINDOW_POOL_SLACK_BATCHES)


def test_window_pool_leftover_grace_scales_with_budget():
    assert extractbench.window_pool_leftover_grace(0.0) == 0.0
    assert extractbench.window_pool_leftover_grace(-1.0) == 0.0
    assert extractbench.window_pool_leftover_grace(0.15) == 0.15 * 0.05
    assert (
        extractbench.window_pool_leftover_grace(1200.0) == extractbench.WINDOW_POOL_LEFTOVER_GRACE
    )


def test_extractbench_file_timeout_covers_pool_and_keeps_floor():
    assert extractbench.extractbench_file_timeout(1, 4, 240.0) == 1800.0
    assert extractbench.extractbench_file_timeout(10, 4, 240.0) == 1800.0
    assert extractbench.extractbench_file_timeout(41, 4, 240.0) == 3120.0
    assert extractbench.extractbench_file_timeout(66, 4, 240.0) == 4560.0
    assert (
        extractbench.extractbench_file_timeout(extractbench.DEFAULT_FILE_TIMEOUT_WINDOWS, 4, 240.0)
        == 6240.0
    )
    assert extractbench.extractbench_file_timeout(66, 4, None) is None


def test_bind_inference_timeouts_disables_timeout_retries():
    def _run(self, pipeline, **kwargs):
        return kwargs

    bound = extractbench.bind_inference_timeouts(_run, 4080.0, timeout_retries=0)
    forced = bound(object(), "openextract", timeout_retries=2, per_file_timeout=1800.0)
    assert forced["per_file_timeout"] == 4080.0
    assert forced["timeout_retries"] == 0
    left = extractbench.bind_inference_timeouts(_run, None, timeout_retries=0)(object(), "x")
    assert left["timeout_retries"] == 0
    assert "per_file_timeout" not in left


def test_windowed_extract_uses_batch_budget_not_one_timeout(monkeypatch, tmp_path):
    import time

    from tests.pdf_fixture import synthetic_pdf

    schema = {
        "type": "object",
        "properties": {"vendor": {"type": "string"}},
        "required": ["vendor"],
    }
    pdf = synthetic_pdf(pages=["page one", "page two", "page three"])
    path = tmp_path / "goshen.pdf"
    path.write_bytes(pdf)
    original = extractbench.Extractor.extract_with_usage

    def _slow(self, source, *, media_type=None):
        time.sleep(0.12)
        return original(self, source, media_type=media_type)

    monkeypatch.setattr(extractbench.Extractor, "extract_with_usage", _slow)
    data, usage, _citations = extractbench.extract_document_with_citations(
        path,
        schema,
        TestModel(custom_output_args={"output": {"vendor": "Acme"}, "citations": []}),
        max_retries=0,
        cite=True,
        timeout=0.2,
        window_concurrency=2,
    )
    assert data["vendor"] == "Acme"
    assert usage.input_tokens >= 0


def test_scanned_pdf_does_not_upload_original(monkeypatch, tmp_path):
    from tests.pdf_fixture import synthetic_pdf

    schema = {
        "type": "object",
        "properties": {"vendor": {"type": "string"}},
        "required": ["vendor"],
    }
    pdf = synthetic_pdf(pages=["", ""])
    path = tmp_path / "bianco.pdf"
    path.write_bytes(pdf)
    media_types: list[str | None] = []
    original = extractbench.Extractor.extract_with_usage

    def _spy(self, source, *, media_type=None):
        media_types.append(media_type)
        payload = source if isinstance(source, bytes) else Path(source).read_bytes()
        assert not payload.startswith(b"%PDF")
        return original(self, source, media_type=media_type)

    monkeypatch.setattr(extractbench.Extractor, "extract_with_usage", _spy)
    data, usage, citations = extractbench.extract_document_with_citations(
        path,
        schema,
        TestModel(custom_output_args={"output": {"vendor": "Acme"}, "citations": []}),
        max_retries=0,
        cite=True,
    )
    assert data["vendor"] == "Acme"
    assert usage.input_tokens >= 0
    assert media_types
    assert all(kind == "image/png" for kind in media_types)
    assert all(citation.bbox is None for citation in citations)
    source, media_type, parsed = extractbench._parse_then_extract_source(pdf, "application/pdf")
    assert media_type == "image/png"
    assert isinstance(source, bytes) and source.startswith(b"\x89PNG")
    assert parsed is not None and not parsed.has_text()


def test_extractbench_records_provider_usage(monkeypatch, tmp_path):
    from openextract import Usage
    from tests.pdf_fixture import synthetic_pdf

    schema = {
        "type": "object",
        "properties": {"vendor": {"type": "string"}},
        "required": ["vendor"],
    }
    path = tmp_path / "w14.pdf"
    path.write_bytes(synthetic_pdf("W14"))

    def _usage(self, source, *, media_type=None):
        return extractbench.ExtractedDocument.model_validate(
            {"output": {"vendor": "Acme"}, "citations": []}
        ), Usage(1280, 64, 1344)

    monkeypatch.setattr(extractbench.Extractor, "extract_with_usage", _usage)
    data, usage, _citations = extractbench.extract_document_with_citations(
        path,
        schema,
        TestModel(custom_output_args={"output": {"vendor": "Acme"}, "citations": []}),
        max_retries=0,
        cite=True,
    )
    assert data["vendor"] == "Acme"
    assert usage == Usage(1280, 64, 1344)


def test_window_source_prefers_image_then_headers():
    from openextract._parse import ParsedDocument, ParsedPage

    text = ParsedDocument(pages=(ParsedPage(1, "Hello", 1, 1, ()),))
    assert extractbench._window_source(text)[1] == "text/plain"
    image = ParsedDocument(pages=(ParsedPage(1, "", 1, 1, (), image=b"\x89PNG-img"),))
    payload, media_type = extractbench._window_source(image)
    assert media_type == "image/png"
    assert payload == b"\x89PNG-img"
    headers = ParsedDocument(pages=(ParsedPage(1, "", 1, 1, ()),))
    payload, media_type = extractbench._window_source(headers)
    assert media_type == "text/plain"
    assert b"--- Page 1 ---" in payload


def test_window_pool_timeout_is_permanent(monkeypatch, tmp_path):
    import time

    from tests.pdf_fixture import synthetic_pdf

    schema = {
        "type": "object",
        "properties": {"vendor": {"type": "string"}},
        "required": ["vendor"],
    }
    pdf = synthetic_pdf(pages=["AAAA page one", "BBBB page two"])
    path = tmp_path / "slow.pdf"
    path.write_bytes(pdf)
    original = extractbench.Extractor.extract_with_usage

    def _slow(self, source, *, media_type=None):
        time.sleep(0.4)
        return original(self, source, media_type=media_type)

    monkeypatch.setattr(extractbench.Extractor, "extract_with_usage", _slow)
    with pytest.raises(ModelError, match="timed out") as exc:
        extractbench.extract_document_with_citations(
            path,
            schema,
            TestModel(custom_output_args={"output": {"vendor": "Acme"}, "citations": []}),
            max_retries=0,
            cite=True,
            timeout=0.05,
            window_concurrency=2,
        )
    assert exc.value.retryable is False


def test_window_pool_timeout_does_not_wait_on_cancelled_workers(monkeypatch, tmp_path):
    import time

    from tests.pdf_fixture import synthetic_pdf

    schema = {
        "type": "object",
        "properties": {"vendor": {"type": "string"}},
        "required": ["vendor"],
    }
    pdf = synthetic_pdf(pages=["AAAA page one", "BBBB page two"])
    path = tmp_path / "hang.pdf"
    path.write_bytes(pdf)
    original = extractbench.Extractor.extract_with_usage

    def _hang(self, source, *, media_type=None):
        time.sleep(2.0)
        return original(self, source, media_type=media_type)

    monkeypatch.setattr(extractbench.Extractor, "extract_with_usage", _hang)
    started = time.monotonic()
    with pytest.raises(ModelError, match="timed out"):
        extractbench.extract_document_with_citations(
            path,
            schema,
            TestModel(custom_output_args={"output": {"vendor": "Acme"}, "citations": []}),
            max_retries=0,
            cite=True,
            timeout=0.05,
            window_concurrency=2,
        )
    assert time.monotonic() - started < 1.0


def test_window_pool_timeout_merges_finished_windows(monkeypatch, tmp_path):
    import time

    from tests.pdf_fixture import synthetic_pdf

    schema = {
        "type": "object",
        "properties": {"vendor": {"type": "string"}},
        "required": ["vendor"],
    }
    pdf = synthetic_pdf(pages=["AAAA page one", "BBBB page two", "CCCC page three"])
    path = tmp_path / "pueblo.pdf"
    path.write_bytes(pdf)

    def _mixed(self, source, *, media_type=None):
        text = source.decode("utf-8", errors="ignore") if isinstance(source, bytes) else ""
        if "AAAA" in text:
            return extractbench.ExtractedDocument.model_validate(
                {"output": {"vendor": "Pueblo"}, "citations": []}
            ), Usage(1, 1, 2)
        time.sleep(2.0)
        raise AssertionError("hung window should be discarded")

    monkeypatch.setattr(extractbench.Extractor, "extract_with_usage", _mixed)
    started = time.monotonic()
    data, usage, _citations = extractbench.extract_document_with_citations(
        path,
        schema,
        TestModel(custom_output_args={"output": {"vendor": "Acme"}, "citations": []}),
        max_retries=0,
        cite=True,
        timeout=0.05,
        window_concurrency=2,
    )
    assert time.monotonic() - started < 1.0
    assert data["vendor"] == "Pueblo"
    assert usage == Usage(1, 1, 2)


def test_window_output_retry_is_retried_at_window(monkeypatch, tmp_path):
    from tests.pdf_fixture import synthetic_pdf

    schema = {
        "type": "object",
        "properties": {"vendor": {"type": "string"}},
        "required": ["vendor"],
    }
    pdf = synthetic_pdf(pages=["AAAA page one", "BBBB page two"])
    path = tmp_path / "veralto.pdf"
    path.write_bytes(pdf)
    calls = {"n": 0}
    original = extractbench.Extractor.extract_with_usage

    def _flaky(self, source, *, media_type=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ExtractionError("Exceeded maximum output retries (1)")
        return original(self, source, media_type=media_type)

    monkeypatch.setattr(extractbench.Extractor, "extract_with_usage", _flaky)
    data, _usage, _citations = extractbench.extract_document_with_citations(
        path,
        schema,
        TestModel(custom_output_args={"output": {"vendor": "Veralto"}, "citations": []}),
        max_retries=0,
        cite=True,
        timeout=2.0,
        window_concurrency=1,
    )
    assert data["vendor"] == "Veralto"
    assert calls["n"] >= 2


def test_window_output_retry_does_not_kill_other_windows(monkeypatch, tmp_path):
    from tests.pdf_fixture import synthetic_pdf

    schema = {
        "type": "object",
        "properties": {"vendor": {"type": "string"}},
        "required": ["vendor"],
    }
    pdf = synthetic_pdf(pages=["AAAA page one", "BBBB page two"])
    path = tmp_path / "veralto.pdf"
    path.write_bytes(pdf)
    calls = {"n": 0}

    def _one_bad(self, source, *, media_type=None):
        calls["n"] += 1
        text = source.decode("utf-8", errors="ignore") if isinstance(source, bytes) else ""
        if "AAAA" in text:
            raise ExtractionError("Exceeded maximum output retries (1)")
        return extractbench.ExtractedDocument.model_validate(
            {"output": {"vendor": "Veralto"}, "citations": []}
        ), Usage(2, 1, 3)

    monkeypatch.setattr(extractbench.Extractor, "extract_with_usage", _one_bad)
    data, usage, _citations = extractbench.extract_document_with_citations(
        path,
        schema,
        TestModel(custom_output_args={"output": {"vendor": "Acme"}, "citations": []}),
        max_retries=0,
        cite=True,
        timeout=2.0,
        window_concurrency=2,
    )
    assert data["vendor"] == "Veralto"
    assert usage == Usage(2, 1, 3)
    assert calls["n"] >= 3


def test_build_schema_agent_raises_output_retries():
    schema = {
        "type": "object",
        "properties": {"vendor": {"type": "string"}},
        "required": ["vendor"],
    }
    agent = extractbench._build_schema_agent(
        TestModel(custom_output_args={"output": {"vendor": "Acme"}, "citations": []}),
        extractbench.json_schema_with_citations(extractbench.prepare_schema(schema)),
        "extract",
    )
    assert agent._max_output_retries == extractbench.WINDOW_OUTPUT_RETRIES
    assert extractbench.WINDOW_OUTPUT_RETRIES > 1


def test_window_output_retry_raises_when_nothing_succeeds(monkeypatch, tmp_path):
    from tests.pdf_fixture import synthetic_pdf

    schema = {
        "type": "object",
        "properties": {"vendor": {"type": "string"}},
        "required": ["vendor"],
    }
    path = tmp_path / "empty.pdf"
    path.write_bytes(synthetic_pdf(pages=["AAAA page one", "BBBB page two"]))

    def _always_fail(self, source, *, media_type=None):
        raise ExtractionError("Exceeded maximum output retries (1)")

    monkeypatch.setattr(extractbench.Extractor, "extract_with_usage", _always_fail)
    with pytest.raises(ExtractionError, match="output retries"):
        extractbench.extract_document_with_citations(
            path,
            schema,
            TestModel(custom_output_args={"output": {"vendor": "Acme"}, "citations": []}),
            max_retries=0,
            cite=True,
            timeout=2.0,
            window_concurrency=2,
        )


def test_run_window_with_output_retries_reraises_other_errors():
    def _boom():
        raise ModelError("request timeout", retryable=True)

    with pytest.raises(ModelError, match="timeout"):
        extractbench._run_window_with_output_retries(_boom, attempts=2)


def test_window_page_one_citation_stamps_document_page(tmp_path):
    from tests.pdf_fixture import synthetic_pdf

    schema = {
        "type": "object",
        "properties": {"vendor": {"type": "string"}},
        "required": ["vendor"],
    }
    pdf = synthetic_pdf(pages=["AAAA noise", "Acme Corp"])
    path = tmp_path / "windowed.pdf"
    path.write_bytes(pdf)
    data, _usage, citations = extractbench.extract_document_with_citations(
        path,
        schema,
        TestModel(
            custom_output_args={
                "output": {"vendor": "Acme Corp"},
                "citations": [{"field": "vendor", "quote": "the supplier", "page": 1}],
            }
        ),
        max_retries=0,
        cite=True,
    )
    assert data["vendor"] == "Acme Corp"
    vendor = next(item for item in citations if item.field == "vendor")
    assert vendor.page == 2
    assert vendor.bbox is not None
    mapped = extractbench.field_citations_for_extractbench(citations)
    assert mapped[0]["page"] == 2
    assert mapped[0]["bbox"] == list(vendor.bbox)


def test_merge_window_payloads_aligns_single_page_window():
    from openextract._parse import ParsedDocument, ParsedPage

    window = ParsedDocument(pages=(ParsedPage(2, "Acme Corp", 1, 1, ()),))
    data, citations = extractbench._merge_window_payloads(
        [
            (
                {
                    "output": {"vendor": "Acme Corp"},
                    "citations": [{"field": "vendor", "quote": "Acme Corp", "page": 1}],
                },
                window,
            )
        ],
        cite=True,
    )
    assert data["vendor"] == "Acme Corp"
    assert citations[0].page == 2


def test_merge_window_payloads_keeps_later_citations():
    data, citations = extractbench._merge_window_payloads(
        [
            {"output": {"vendor": None}, "citations": []},
            {
                "output": {"vendor": "Acme Corp"},
                "citations": [{"field": "vendor", "quote": "Acme Corp", "page": 2}],
            },
        ],
        cite=True,
    )
    assert data["vendor"] == "Acme Corp"
    assert citations[0].page == 2
    assert citations[0].quote == "Acme Corp"
    empty, empty_cites = extractbench._merge_window_payloads([], cite=True)
    assert empty == {}
    assert empty_cites == ()


def test_parse_args_cite_defaults_on():
    args = extractbench.parse_args(["--model", "openai:gpt-5", "--test"])
    assert args.cite is True
    assert extractbench.parse_args(["--model", "openai:gpt-5", "--no-cite"]).cite is False
    timed = extractbench.parse_args(
        ["--model", "openai:gpt-5", "--timeout", "90", "--window-concurrency", "2"]
    )
    assert timed.timeout == 90
    assert timed.window_concurrency == 2


def test_extract_document_with_test_model():
    schema = {
        "type": "object",
        "properties": {
            "vendor": {"type": "string"},
            "total": {"type": "number"},
        },
        "required": ["vendor", "total"],
    }
    model = TestModel(custom_output_args={"vendor": "Acme", "total": 12.5})
    data, usage = extractbench.extract_document(
        b"%PDF-fixture",
        schema,
        model,
        media_type="application/pdf",
        max_retries=0,
    )
    assert data["vendor"] == "Acme"
    assert data["total"] == 12.5
    assert usage.input_tokens >= 0


def test_output_type_for_schema_raises_instead_of_degrading(capsys):
    schema = {"title": "InvoiceList", "type": "array", "items": {"type": "object"}}
    with pytest.raises(SchemaValidationError, match="InvoiceList"):
        extractbench._output_type_for_schema(schema, "openai:gpt-5")
    assert "InvoiceList" in capsys.readouterr().err


def test_extract_document_rejects_unconvertible_schema(capsys):
    model = TestModel(custom_output_args={"vendor": "Acme"})
    with pytest.raises(SchemaValidationError):
        extractbench.extract_document(
            b"%PDF-fixture",
            {"type": "array", "items": {"type": "string"}},
            model,
            media_type="application/pdf",
            max_retries=0,
        )
    assert "warning:" in capsys.readouterr().err


def _fake_venv(tmp_path: Path) -> tuple[Path, Path]:
    venv_dir = tmp_path / "venv"
    python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.touch()
    return venv_dir, python


def _patch_bootstrap(monkeypatch, tmp_path: Path, venv_dir: Path) -> list[list[str]]:
    monkeypatch.setattr(extractbench, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(extractbench, "VENV_DIR", venv_dir)
    monkeypatch.setattr(extractbench, "extract_bench_available", lambda: False)
    commands: list[list[str]] = []
    monkeypatch.setattr(extractbench, "_run", commands.append)
    return commands


def test_inside_benchmark_venv_compares_sys_prefix(monkeypatch, tmp_path):
    """Membership uses sys.prefix, not resolved executables (uv symlinks alias)."""
    venv_dir = tmp_path / "venv"
    monkeypatch.setattr(extractbench, "VENV_DIR", venv_dir)
    monkeypatch.setattr(sys, "prefix", str(venv_dir))
    assert extractbench._inside_benchmark_venv() is True
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "other-venv"))
    assert extractbench._inside_benchmark_venv() is False


def test_ensure_extract_bench_raises_inside_broken_benchmark_venv(monkeypatch, tmp_path):
    venv_dir, _python = _fake_venv(tmp_path)
    commands = _patch_bootstrap(monkeypatch, tmp_path, venv_dir)
    monkeypatch.setattr(sys, "prefix", str(venv_dir))
    with pytest.raises(RuntimeError, match="not importable"):
        extractbench.ensure_extract_bench()
    assert commands == []


def test_ensure_extract_bench_skips_install_when_already_present(monkeypatch, tmp_path):
    venv_dir, python = _fake_venv(tmp_path)
    commands = _patch_bootstrap(monkeypatch, tmp_path, venv_dir)
    monkeypatch.setattr(extractbench, "_venv_has_extract_bench", lambda _python: True)
    monkeypatch.setattr(sys, "argv", ["extractbench.py"])
    execs: list[tuple] = []
    monkeypatch.setattr(os, "execv", lambda *call: execs.append(call))
    extractbench.ensure_extract_bench()
    assert commands == []
    assert execs == [(str(python), [str(python), str(SCRIPTS / "extractbench.py")])]


def test_ensure_extract_bench_installs_when_probe_fails(monkeypatch, tmp_path):
    venv_dir, python = _fake_venv(tmp_path)
    commands = _patch_bootstrap(monkeypatch, tmp_path, venv_dir)
    monkeypatch.setattr(extractbench, "_venv_has_extract_bench", lambda _python: False)
    extractbench.ensure_extract_bench(reexec=False)
    assert len(commands) == 1
    assert extractbench.EXTRACTBENCH_GIT in commands[0]
    assert str(python) in commands[0]


def test_venv_has_extract_bench_false_when_python_missing(tmp_path):
    assert extractbench._venv_has_extract_bench(tmp_path / "missing") is False


def test_parse_args_test_split_and_model():
    args = extractbench.parse_args(["--model", "openai:gpt-5", "--test"])
    assert args.model == "openai:gpt-5"
    assert args.test is True
    assert args.max_concurrent == 4


def test_parse_args_serve_without_name():
    args = extractbench.parse_args(["--serve"])
    assert args.serve == ""


def test_require_model_uses_env(monkeypatch):
    monkeypatch.setenv("OPENEXTRACT_MODEL", "xai:grok-4.3")
    args = extractbench.parse_args(["--test"])
    assert extractbench._require_model(args) == "xai:grok-4.3"


def test_require_model_exits_without_model(monkeypatch, capsys):
    monkeypatch.delenv("OPENEXTRACT_MODEL", raising=False)
    args = extractbench.parse_args(["--test"])
    with pytest.raises(SystemExit) as exc:
        extractbench._require_model(args)
    assert exc.value.code == 2
    assert "--model is required" in capsys.readouterr().err


def test_main_install_bootstraps_without_model(monkeypatch, capsys):
    monkeypatch.setattr(extractbench, "ensure_extract_bench", lambda: None)
    assert extractbench.main(["--install"]) == 0
    assert "ExtractBench ready" in capsys.readouterr().out
