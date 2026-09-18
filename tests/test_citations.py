"""Tests for opt-in per-field provenance."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel
from pydantic_ai.models.test import TestModel

from openextract import (
    AsyncExtractor,
    Citation,
    ExtractionInput,
    ExtractionResult,
    Extractor,
    extract,
    extract_async,
    extract_many,
    extract_many_with_results,
    extract_swarm,
    extract_swarm_with_results,
    extract_with_usage,
    extract_with_usage_async,
)
from openextract._citations import (
    CITATION_INSTRUCTIONS,
    CitationDraft,
    citations_from_payload,
    cited_output_schema,
    field_citations_for_extractbench,
    filter_citations,
    json_schema_with_citations,
    prepare_cited_run,
    sanitize_citation,
    split_cited_output,
    with_citation_instructions,
)
from openextract._confidence import MATCH_SCORES, score_citation_match


class Person(BaseModel):
    name: str
    age: int


def _cited_model(**kwargs: object) -> TestModel:
    return TestModel(custom_output_args=kwargs)


class TestCitationMapping:
    def test_as_dict_is_json_stable(self):
        citation = Citation(
            field="lines[0].qty",
            quote="3",
            page=1,
            bbox=(0.1, 0.2, 0.3, 0.05),
        )
        dumped = citation.as_dict()
        assert dumped == {
            "field": "lines[0].qty",
            "quote": "3",
            "page": 1,
            "bbox": [0.1, 0.2, 0.3, 0.05],
            "confidence": None,
            "match": None,
        }
        assert json.loads(json.dumps(dumped)) == dumped

    def test_as_dict_includes_quote_only_without_inventing_bbox(self):
        quote_only = Citation(field="vendor", quote="Acme")
        assert quote_only.as_dict() == {
            "field": "vendor",
            "quote": "Acme",
            "page": None,
            "bbox": None,
            "confidence": None,
            "match": None,
        }
        assert quote_only.as_field_citation() is None
        assert field_citations_for_extractbench([quote_only]) == []

    def test_as_dict_includes_heuristic_confidence(self):
        citation = Citation(
            field="vendor",
            quote="Acme",
            page=1,
            bbox=(0.1, 0.2, 0.3, 0.05),
            confidence=0.95,
            match="exact",
        )
        dumped = citation.as_dict()
        assert dumped["confidence"] == 0.95
        assert dumped["match"] == "exact"
        mapped = citation.as_field_citation()
        assert mapped is not None
        assert "confidence" not in mapped
        assert "match" not in mapped
        assert mapped["field_path"] == "vendor"

    def test_model_supplied_confidence_is_ignored(self):
        citation = sanitize_citation(
            {"field": "vendor", "quote": "Acme", "page": 1, "confidence": 0.99, "match": "exact"}
        )
        assert citation == Citation("vendor", "Acme", 1, None)
        kept = sanitize_citation(
            Citation("vendor", "Acme", 1, None, confidence=0.95, match="exact")
        )
        assert kept == Citation("vendor", "Acme", 1, confidence=0.95, match="exact")
        quote_only = Citation(field="vendor", quote="Acme")
        assert quote_only.as_field_citation() is None
        assert field_citations_for_extractbench([quote_only]) == []

    def test_page_only_emits_extractbench_payload(self):
        citation = Citation(field="vendor", page=2)
        assert citation.as_field_citation() == {
            "field_path": "vendor",
            "page": 2,
            "bbox": None,
            "reference_text": None,
        }

    def test_normalized_bbox_is_passed_through(self):
        citation = Citation(
            field="lines[0].qty",
            quote="3",
            page=1,
            bbox=(0.1, 0.2, 0.3, 0.05),
        )
        payload = citation.as_field_citation()
        assert payload is not None
        assert payload["bbox"] == [0.1, 0.2, 0.3, 0.05]
        assert payload["field_path"] == "lines[0].qty"
        assert payload["reference_text"] == "3"


class TestSanitize:
    def test_credential_urls_and_data_uris_are_redacted(self):
        citation = sanitize_citation(
            {
                "field": "vendor",
                "quote": (
                    "see https://user:secret@example.com/x?token=1 and data:text/plain;base64,abcd"
                ),
                "page": 1,
            }
        )
        assert citation is not None
        assert "secret" not in citation.quote
        assert "token=" not in citation.quote
        assert "data:" not in citation.quote
        assert "[redacted]" in citation.quote

    def test_pixel_bbox_is_dropped_not_rescaled(self):
        citation = sanitize_citation(
            {"field": "vendor", "quote": "Acme", "page": 1, "bbox": [177, 82, 318, 43]}
        )
        assert citation is not None
        assert citation.bbox is None
        assert citation.page == 1

    def test_invalid_field_and_empty_evidence_are_dropped(self):
        assert sanitize_citation({"field": "../etc/passwd", "quote": "x", "page": 1}) is None
        assert sanitize_citation({"field": "vendor"}) is None
        assert sanitize_citation(object()) is None
        assert sanitize_citation({"field": "vendor", "quote": b"bytes", "page": True}) is None
        zero_page = sanitize_citation({"field": "vendor", "quote": "Acme", "page": 0})
        assert zero_page is not None and zero_page.page is None

    def test_draft_and_citation_round_trip(self):
        draft = CitationDraft(field="vendor", quote="Acme", page=1)
        citation = sanitize_citation(draft)
        assert citation == Citation("vendor", "Acme", 1, None)
        assert sanitize_citation(citation) == citation

    def test_short_quote_with_page_is_kept(self):
        citation = sanitize_citation({"field": "quarter", "quote": "Q1", "page": 1})
        assert citation == Citation("quarter", "Q1", 1, None)

    def test_payload_skips_non_sequences(self):
        assert citations_from_payload("vendor") == ()
        assert citations_from_payload(b"vendor") == ()
        assert citations_from_payload(None) == ()
        assert citations_from_payload([{"field_path": "vendor", "page": 1}]) == (
            Citation(field="vendor", page=1),
        )
        assert citations_from_payload([{"field": "../bad"}, object()]) == ()

    def test_bbox_edge_cases_are_dropped(self):
        assert (
            sanitize_citation({"field": "n", "page": 1, "bbox": [True, 0.0, 0.1, 0.1]}).bbox is None
        )
        assert sanitize_citation({"field": "n", "page": 1, "bbox": [0.1, 0.1, 0.1, 0]}).bbox is None
        assert (
            sanitize_citation({"field": "n", "page": 1, "bbox": [0.8, 0.1, 0.3, 0.1]}).bbox is None
        )
        assert sanitize_citation({"field": "n", "page": 1, "bbox": [0.1, 0.1, 0.1]}).bbox is None
        long = sanitize_citation({"field": "n", "quote": "x" * 3000, "page": 1})
        assert long is not None and len(long.quote) == 2000


class TestCitationConfidence:
    def test_scores_are_heuristic_not_model_provided(self):
        assert score_citation_match(None, parsed=True) is None
        assert score_citation_match("nope", parsed=True) is None
        assert score_citation_match("exact", parsed=True) == MATCH_SCORES["exact"]
        assert score_citation_match("numeric", parsed=True) == 0.85
        assert score_citation_match("value", parsed=True) == 0.80
        assert score_citation_match("fuzzy", parsed=True) == 0.65
        assert score_citation_match("page", parsed=True) == 0.40
        assert score_citation_match("quote", parsed=True) == 0.50
        assert score_citation_match("page", parsed=False) == 0.30
        assert score_citation_match("quote", parsed=False) == 0.45
        assert score_citation_match("quote", parsed=False, agrees=True) == 0.55
        assert score_citation_match("quote", parsed=False, agrees=False) == 0.25
        assert score_citation_match("exact", parsed=False) == MATCH_SCORES["exact"]
        assert score_citation_match("numeric", parsed=False) == MATCH_SCORES["numeric"]
        assert score_citation_match("fuzzy", parsed=False) == MATCH_SCORES["fuzzy"]
        assert score_citation_match("value", parsed=False) == MATCH_SCORES["value"]

    def test_split_stamps_unparsed_quote_agreement(self):
        output, citations = split_cited_output(
            {
                "output": {"name": "Ada", "age": 36},
                "citations": [
                    {"field": "name", "quote": "Ada Lovelace", "page": 1},
                    {"field": "age", "quote": "99", "page": 1},
                ],
            },
            Person,
            cite=True,
        )
        assert output == Person(name="Ada", age=36)
        assert citations[0].match == "quote"
        assert citations[0].confidence == 0.55
        assert citations[1].match == "quote"
        assert citations[1].confidence == 0.25

    def test_split_stamps_unparsed_page_only(self):
        output, citations = split_cited_output(
            {"output": {"name": "Ada", "age": 36}, "citations": [{"field": "name", "page": 1}]},
            Person,
            cite=True,
        )
        assert output.name == "Ada"
        assert citations[0].match == "page"
        assert citations[0].confidence == 0.30
        assert citations[0].bbox is None


class TestCiteMinConfidence:
    def test_filter_drops_weak_and_unstamped(self):
        exact = Citation("vendor", "Acme", 1, confidence=0.95, match="exact")
        quote = Citation("total", "12.50", 1, confidence=0.55, match="quote")
        weak = Citation("date", "Jan", 1, confidence=0.25, match="quote")
        unstamped = Citation("notes", "see source", 1)
        kept = filter_citations((exact, quote, weak, unstamped), 0.55)
        assert kept == (exact, quote)
        assert filter_citations((exact, unstamped), None) == (exact, unstamped)

    def test_split_filters_after_grounding(self):
        output, citations = split_cited_output(
            {
                "output": {"name": "Ada", "age": 36},
                "citations": [
                    {"field": "name", "quote": "Ada Lovelace", "page": 1},
                    {"field": "age", "quote": "99", "page": 1},
                ],
            },
            Person,
            cite=True,
            cite_min_confidence=0.5,
        )
        assert output == Person(name="Ada", age=36)
        assert [item.field for item in citations] == ["name"]
        assert citations[0].confidence == 0.55
        mapped = field_citations_for_extractbench(citations)
        assert mapped[0]["field_path"] == "name"
        assert "confidence" not in mapped[0]

    def test_results_api_applies_threshold(self):
        model = _cited_model(
            output={"name": "Ada", "age": 36},
            citations=[
                {"field": "name", "quote": "Ada", "page": 1},
                {"field": "age", "quote": "99", "page": 1},
            ],
        )
        results = extract_many_with_results(
            Person,
            model,
            [ExtractionInput(b"Ada is 36", media_type="text/plain")],
            cite=True,
            cite_min_confidence=0.5,
        )
        assert [item.field for item in results[0].citations] == ["name"]
        unfiltered = extract_many_with_results(
            Person,
            model,
            [ExtractionInput(b"Ada is 36", media_type="text/plain")],
            cite=True,
        )
        assert {item.field for item in unfiltered[0].citations} == {"name", "age"}

    def test_swarm_filters_agent_and_reduced_citations(self):
        model = _cited_model(
            output={"name": "Ada", "age": 36},
            citations=[
                {"field": "name", "quote": "Ada", "page": 1},
                {"field": "age", "quote": "99", "page": 1},
            ],
        )
        swarm = extract_swarm_with_results(
            Person,
            model,
            b"Ada is 36",
            media_type="text/plain",
            cite=True,
            cite_min_confidence=0.5,
        )
        assert [item.field for item in swarm.citations] == ["name"]
        assert [item.field for item in swarm.agents[0].citations] == ["name"]

    def test_default_cite_false_is_unchanged(self):
        model = TestModel(custom_output_args={"name": "Ada", "age": 36})
        result = extract(Person, model, b"x", media_type="text/plain", cite_min_confidence=0.9)
        assert result == Person(name="Ada", age=36)

    @pytest.mark.parametrize("value", [-0.01, 1.01, float("inf"), float("nan"), True])
    def test_invalid_threshold_raises_at_call_time(self, value):
        model = TestModel(custom_output_args={"name": "Ada", "age": 36})
        with pytest.raises(ValueError, match="cite_min_confidence must be a finite number in"):
            extract(Person, model, b"x", media_type="text/plain", cite_min_confidence=value)
        with pytest.raises(ValueError, match="cite_min_confidence must be a finite number in"):
            Extractor(Person, model, cite_min_confidence=value)
        with pytest.raises(ValueError, match="cite_min_confidence must be a finite number in"):
            extract_many(
                Person,
                model,
                [ExtractionInput(b"x", media_type="text/plain")],
                cite_min_confidence=value,
            )
        with pytest.raises(ValueError, match="cite_min_confidence must be a finite number in"):
            extract_swarm(Person, model, b"x", media_type="text/plain", cite_min_confidence=value)


class TestSchemaWrap:
    def test_default_prepare_is_unchanged(self):
        schema, instructions = prepare_cited_run(Person, "pull the person", False)
        assert schema is Person
        assert instructions == "pull the person"

    def test_cite_wraps_schema_and_appends_instructions(self):
        schema, instructions = prepare_cited_run(Person, "pull the person", True)
        assert schema is not Person
        assert schema is cited_output_schema(Person)
        assert instructions is not None
        assert instructions.startswith("pull the person")
        assert CITATION_INSTRUCTIONS in instructions
        assert with_citation_instructions(None) == CITATION_INSTRUCTIONS
        assert with_citation_instructions("   ") == CITATION_INSTRUCTIONS

    def test_json_schema_envelope_matches_extractbench_fields(self):
        wrapped = json_schema_with_citations(
            {"type": "object", "properties": {"n": {"type": "string"}}}
        )
        props = wrapped["properties"]["citations"]["items"]["properties"]
        assert set(props) == {"field", "quote", "page"}
        assert "bbox" not in props
        assert wrapped["required"] == ["output", "citations"]
        assert "bbox" not in CITATION_INSTRUCTIONS
        assert "MUST return a citation" in CITATION_INSTRUCTIONS

    def test_split_default_returns_raw(self):
        person = Person(name="Ada", age=36)
        output, citations = split_cited_output(person, Person, cite=False)
        assert output is person
        assert citations == ()

    def test_wrap_is_cached_and_split_accepts_dict(self):
        assert cited_output_schema(Person) is cited_output_schema(Person)
        output, citations = split_cited_output(
            {"output": {"name": "Ada", "age": 36}, "citations": [{"field": "name", "page": 1}]},
            Person,
            cite=True,
        )
        assert output == Person(name="Ada", age=36)
        assert citations[0].field == "name"

    def test_split_aligns_citations_to_one_page_window(self):
        from openextract._parse import ParsedDocument, ParsedPage

        window = ParsedDocument(pages=(ParsedPage(4, "Ada", 1, 1, ()),))
        output, citations = split_cited_output(
            {
                "output": {"name": "Ada", "age": 36},
                "citations": [{"field": "name", "quote": "Ada"}],
            },
            Person,
            cite=True,
            window=window,
        )
        assert output == Person(name="Ada", age=36)
        assert citations[0].page == 4


class TestExtractCite:
    def test_default_extract_does_not_wrap_or_add_instructions(self):
        model = _cited_model(name="Ada", age=36)
        result = extract(Person, model, b"Ada, 36", media_type="text/plain")
        assert result == Person(name="Ada", age=36)

    def test_cite_rebuilds_agent_output_type(self, tmp_path, mocker):
        from tests.test_extract import _make_agent_mock

        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        wrapped = cited_output_schema(Person)
        payload = wrapped(output=Person(name="Ada", age=36), citations=[])
        agent_cls, _ = _make_agent_mock(mocker, output=payload)
        result = extract(Person, "openai:gpt-5", str(local), instructions="pull", cite=True)
        assert result == Person(name="Ada", age=36)
        kwargs = agent_cls.call_args.kwargs
        assert kwargs["output_type"] is wrapped
        assert CITATION_INSTRUCTIONS in kwargs["instructions"]

    def test_cite_true_still_returns_schema_instance(self):
        model = _cited_model(
            output={"name": "Ada", "age": 36},
            citations=[
                {
                    "field": "name",
                    "quote": "Ada Lovelace",
                    "page": 1,
                    "bbox": [0.1, 0.2, 0.3, 0.05],
                }
            ],
        )
        result = extract(Person, model, b"Ada Lovelace, 36", media_type="text/plain", cite=True)
        assert result == Person(name="Ada", age=36)

    def test_with_usage_unwraps_cited_output(self):
        model = _cited_model(output={"name": "Ada", "age": 36}, citations=[])
        output, usage = extract_with_usage(Person, model, b"x", media_type="text/plain", cite=True)
        assert output == Person(name="Ada", age=36)
        assert usage.total_tokens >= 0

    async def test_async_cite_unwraps(self):
        model = _cited_model(output={"name": "Grace", "age": 85}, citations=[])
        result = await extract_async(Person, model, b"x", media_type="text/plain", cite=True)
        assert result == Person(name="Grace", age=85)

    def test_results_api_attaches_sanitized_citations(self):
        model = _cited_model(
            output={"name": "Ada", "age": 36},
            citations=[
                {
                    "field": "name",
                    "quote": "Ada https://user:secret@host/x",
                    "page": 1,
                    "bbox": [0.1, 0.2, 0.3, 0.05],
                },
                {"field": "age", "quote": "36"},
            ],
        )
        results = extract_many_with_results(
            Person,
            model,
            [ExtractionInput(b"doc", media_type="text/plain", name="page.pdf")],
            cite=True,
        )
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, ExtractionResult)
        assert result.output == Person(name="Ada", age=36)
        assert result.source == "page.pdf"
        assert result.citations[0].quote is not None
        assert "secret" not in result.citations[0].quote
        assert result.citations[0].bbox is None
        assert result.citations[0].match == "quote"
        assert result.citations[0].confidence == 0.55
        assert result.citations[1].page is None
        mapped = field_citations_for_extractbench(result.citations)
        assert [item["field_path"] for item in mapped] == ["name"]
        dumped = str(result)
        assert b"doc" not in dumped.encode()
        assert "secret" not in dumped

    def test_default_results_have_empty_citations(self):
        model = _cited_model(name="Ada", age=36)
        results = extract_many_with_results(
            Person, model, [ExtractionInput(b"x", media_type="text/plain")]
        )
        assert results[0].citations == ()

    def test_session_cite_unwraps(self):
        model = _cited_model(output={"name": "Ada", "age": 36}, citations=[])
        with Extractor(Person, model, cite=True) as extractor:
            result = extractor.extract(b"x", media_type="text/plain")
        assert result == Person(name="Ada", age=36)

    def test_batch_and_swarm_cite_keep_schema_instances(self):
        model = _cited_model(
            output={"name": "Ada", "age": 36},
            citations=[{"field": "name", "quote": "Ada", "page": 1}],
        )
        batch = extract_many(
            Person, model, [ExtractionInput(b"x", media_type="text/plain")], cite=True
        )
        assert batch == [Person(name="Ada", age=36)]
        swarm = extract_swarm_with_results(Person, model, b"x", media_type="text/plain", cite=True)
        assert swarm.output == Person(name="Ada", age=36)
        assert swarm.agents[0].citations[0].page == 1
        assert swarm.citations[0].page == 1

    async def test_async_usage_cite(self):
        model = _cited_model(output={"name": "Ada", "age": 36}, citations=[])
        output, usage = await extract_with_usage_async(
            Person, model, b"x", media_type="text/plain", cite=True
        )
        assert output == Person(name="Ada", age=36)
        assert usage.total_tokens >= 0

    def test_large_pdf_is_chunked_across_extract_surfaces(self, monkeypatch):
        from tests.pdf_fixture import synthetic_pdf

        monkeypatch.setattr("openextract._parse.DEFAULT_PARSE_WINDOW_CHARS", 40)
        pdf = synthetic_pdf(pages=["AAAA " * 30, "Ada Lovelace " + "BBBB " * 30])
        model = _cited_model(
            output={"name": "Ada", "age": 36},
            citations=[{"field": "name", "quote": "Ada Lovelace", "page": 2}],
        )
        assert extract(Person, model, pdf, media_type="application/pdf", cite=True) == Person(
            name="Ada", age=36
        )
        output, usage = extract_with_usage(
            Person, model, pdf, media_type="application/pdf", cite=True
        )
        assert output == Person(name="Ada", age=36)
        assert usage.total_tokens >= 0
        results = extract_many_with_results(
            Person, model, [ExtractionInput(pdf, media_type="application/pdf")], cite=True
        )
        assert results[0].output == Person(name="Ada", age=36)
        assert results[0].citations[0].page == 2
        swarm = extract_swarm_with_results(
            Person, model, pdf, media_type="application/pdf", cite=True
        )
        assert swarm.output == Person(name="Ada", age=36)
        with Extractor(Person, model, cite=True) as extractor:
            session_plain = extractor.extract(pdf, media_type="application/pdf")
            session_out, session_usage = extractor.extract_with_usage(
                pdf, media_type="application/pdf"
            )
        assert session_plain == Person(name="Ada", age=36)
        assert session_out == Person(name="Ada", age=36)
        assert session_usage.total_tokens >= 0

    async def test_async_surfaces_chunk_large_pdf(self, monkeypatch):
        from tests.pdf_fixture import synthetic_pdf

        monkeypatch.setattr("openextract._parse.DEFAULT_PARSE_WINDOW_CHARS", 40)
        pdf = synthetic_pdf(pages=["AAAA " * 30, "Ada Lovelace " + "BBBB " * 30])
        model = _cited_model(
            output={"name": "Ada", "age": 36},
            citations=[{"field": "name", "quote": "Ada Lovelace", "page": 2}],
        )
        assert await extract_async(
            Person, model, pdf, media_type="application/pdf", cite=True
        ) == Person(name="Ada", age=36)
        output, usage = await extract_with_usage_async(
            Person, model, pdf, media_type="application/pdf", cite=True
        )
        assert output == Person(name="Ada", age=36)
        assert usage.total_tokens >= 0
        async with AsyncExtractor(Person, model, cite=True) as extractor:
            session_out = await extractor.extract(pdf, media_type="application/pdf")
            session_pair = await extractor.extract_with_usage(pdf, media_type="application/pdf")
        assert session_out == Person(name="Ada", age=36)
        assert session_pair[0] == Person(name="Ada", age=36)


class PartialPerson(BaseModel):
    name: str | None = None
    age: int | None = None


class TestSwarmCiteReduce:
    def test_merge_collects_surviving_field_citations(self):
        left = _cited_model(
            output={"name": "Ada", "age": None},
            citations=[{"field": "name", "quote": "Ada", "page": 1}],
        )
        right = _cited_model(
            output={"name": None, "age": 36},
            citations=[{"field": "age", "quote": "36", "page": 1}],
        )
        swarm = extract_swarm_with_results(
            PartialPerson, [left, right], b"Ada is 36", media_type="text/plain", cite=True
        )
        assert swarm.output == PartialPerson(name="Ada", age=36)
        assert [(item.field, item.page) for item in swarm.citations] == [
            ("name", 1),
            ("age", 1),
        ]
        assert swarm.agents[0].citations[0].field == "name"
        assert swarm.agents[1].citations[0].field == "age"

    def test_vote_keeps_majority_citation_not_the_minority(self):
        minority = _cited_model(
            output={"name": "Ada", "age": None},
            citations=[{"field": "name", "quote": "Ada", "page": 1}],
        )
        majority_a = _cited_model(
            output={"name": "Grace", "age": None},
            citations=[{"field": "name", "quote": "Grace", "page": 2}],
        )
        majority_b = _cited_model(
            output={"name": "Grace", "age": None},
            citations=[{"field": "name", "quote": "Grace Hopper", "page": 3}],
        )
        swarm = extract_swarm_with_results(
            PartialPerson,
            [minority, majority_a, majority_b],
            b"Grace",
            media_type="text/plain",
            reduce="vote",
            cite=True,
        )
        assert swarm.output.name == "Grace"
        assert swarm.citations == (Citation("name", "Grace", 2, confidence=0.55, match="quote"),)
        assert swarm.agents[0].citations[0].page == 1

    def test_first_keeps_the_leading_agent_citations(self):
        first = _cited_model(
            output={"name": "Ada", "age": None},
            citations=[{"field": "name", "quote": "Ada", "page": 1}],
        )
        later = _cited_model(
            output={"name": "Grace", "age": 36},
            citations=[{"field": "name", "quote": "Grace", "page": 2}],
        )
        swarm = extract_swarm_with_results(
            PartialPerson,
            [first, later],
            b"names",
            media_type="text/plain",
            reduce="first",
            cite=True,
        )
        assert swarm.output == PartialPerson(name="Ada")
        assert swarm.citations == swarm.agents[0].citations
        assert swarm.citations[0].page == 1

    def test_bare_extract_swarm_still_returns_the_schema(self):
        model = _cited_model(
            output={"name": "Ada", "age": 36},
            citations=[{"field": "name", "quote": "Ada", "page": 1}],
        )
        result = extract_swarm(Person, model, b"x", media_type="text/plain", cite=True)
        assert result == Person(name="Ada", age=36)

    def test_cite_false_leaves_reduced_citations_empty(self):
        model = TestModel(custom_output_args={"name": "Ada", "age": 36})
        swarm = extract_swarm_with_results(Person, model, b"x", media_type="text/plain")
        assert swarm.citations == ()
        assert swarm.agents[0].citations == ()
