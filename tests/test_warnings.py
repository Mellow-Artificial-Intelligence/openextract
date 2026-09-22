"""Soft-degradation warnings on ExtractionResult."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai.models.test import TestModel

from openextract import (
    AsyncExtractor,
    ExtractionInput,
    ExtractionResult,
    Extractor,
    extract_many_with_results,
    extract_swarm_with_results,
    extract_with_result,
    extract_with_result_async,
)
from openextract._warnings import (
    cite_min_confidence_warning,
    extraction_warnings,
    max_pages_warning,
    merge_warnings,
    page_filter_warnings,
    pages_available_warning,
    pages_requested_warning,
)
from tests.pdf_fixture import synthetic_pdf


class Person(BaseModel):
    name: str
    age: int


def _plain_model() -> TestModel:
    return TestModel(custom_output_args={"name": "Ada", "age": 36})


def _cited_model() -> TestModel:
    return TestModel(
        custom_output_args={
            "output": {"name": "Ada", "age": 36},
            "citations": [
                {"field": "name", "quote": "Ada", "page": 1},
                {"field": "age", "quote": "99", "page": 1},
            ],
        }
    )


def _three_page_pdf() -> bytes:
    return synthetic_pdf(pages=["AAAA " * 30, "Ada Lovelace " + "BBBB " * 30, "CCCC " * 30])


class TestWarningHelpers:
    def test_page_filter_silent_without_filters_or_drops(self):
        assert page_filter_warnings(available=3, kept=3, pages=None, max_pages=None) == ()
        assert page_filter_warnings(available=3, kept=3, pages=(1, 2, 3), max_pages=None) == ()
        assert page_filter_warnings(available=3, kept=3, pages=None, max_pages=5) == ()
        assert page_filter_warnings(available=None, kept=0, pages=(1,), max_pages=2) == ()

    def test_page_filter_requested_and_available_counts(self):
        assert page_filter_warnings(available=3, kept=1, pages=(1, 99), max_pages=None) == (
            pages_requested_warning(1, 2),
            pages_available_warning(2, 3),
        )
        assert page_filter_warnings(available=3, kept=2, pages=None, max_pages=2) == (
            max_pages_warning(2, 1, 3),
        )
        assert page_filter_warnings(available=3, kept=1, pages=(1,), max_pages=None) == (
            pages_available_warning(2, 3),
        )

    def test_cite_and_merge_helpers(self):
        assert cite_min_confidence_warning(0.5, 1) == ("cite_min_confidence=0.5 dropped 1 citation")
        assert cite_min_confidence_warning(0.8, 2) == (
            "cite_min_confidence=0.8 dropped 2 citations"
        )
        assert merge_warnings(("a", "b"), ("b", "c")) == ("a", "b", "c")
        assert extraction_warnings(None, (1,), 2, ("extra",)) == ("extra",)


class TestHappyPathEmpty:
    def test_oneshot_text_has_no_warnings(self):
        result = extract_with_result(
            Person,
            _plain_model(),
            ExtractionInput(b"Ada Lovelace", media_type="text/plain", name="bio.txt"),
        )
        assert result.warnings == ()
        assert result.as_dict()["warnings"] == []

    def test_cite_without_threshold_has_no_warnings(self):
        result = extract_with_result(
            Person,
            _cited_model(),
            b"Ada is 36",
            media_type="text/plain",
            cite=True,
        )
        assert result.warnings == ()
        assert {item.field for item in result.citations} == {"name", "age"}


class TestPageFilterWarnings:
    def test_pages_drops_requested_and_available(self):
        result = extract_with_result(
            Person,
            _plain_model(),
            _three_page_pdf(),
            media_type="application/pdf",
            pages=(1, 99),
        )
        assert result.warnings == (
            pages_requested_warning(1, 2),
            pages_available_warning(2, 3),
        )
        assert all("http" not in item and "/" not in item for item in result.warnings)

    def test_max_pages_drops_available(self):
        result = extract_with_result(
            Person,
            _plain_model(),
            _three_page_pdf(),
            media_type="application/pdf",
            max_pages=2,
        )
        assert result.warnings == (max_pages_warning(2, 1, 3),)

    def test_pages_allowlist_drops_available(self):
        result = extract_with_result(
            Person,
            _plain_model(),
            _three_page_pdf(),
            media_type="application/pdf",
            pages=(1,),
        )
        assert result.warnings == (pages_available_warning(2, 3),)

    def test_full_allowlist_stays_empty(self):
        result = extract_with_result(
            Person,
            _plain_model(),
            _three_page_pdf(),
            media_type="application/pdf",
            pages=(1, 2, 3),
        )
        assert result.warnings == ()

    def test_max_pages_noop_on_plain_text(self):
        result = extract_with_result(
            Person,
            _plain_model(),
            b"Ada is 36",
            media_type="text/plain",
            max_pages=1,
        )
        assert result.warnings == ()


class TestCiteMinConfidenceWarnings:
    def test_oneshot_and_async(self):
        model = _cited_model()
        result = extract_with_result(
            Person,
            model,
            b"Ada is 36",
            media_type="text/plain",
            cite=True,
            cite_min_confidence=0.5,
        )
        assert result.warnings == (cite_min_confidence_warning(0.5, 1),)
        assert [item.field for item in result.citations] == ["name"]

    async def test_async_twin(self):
        result = await extract_with_result_async(
            Person,
            _cited_model(),
            b"Ada is 36",
            media_type="text/plain",
            cite=True,
            cite_min_confidence=0.5,
        )
        assert result.warnings == (cite_min_confidence_warning(0.5, 1),)


class TestSessionBatchSwarm:
    def test_session_page_and_cite_warnings(self):
        with Extractor(Person, _plain_model(), pages=(1,)) as extractor:
            paged = extractor.extract_with_result(_three_page_pdf(), media_type="application/pdf")
        assert paged.warnings == (pages_available_warning(2, 3),)
        with Extractor(Person, _cited_model(), cite=True, cite_min_confidence=0.5) as extractor:
            cited = extractor.extract_with_result(b"Ada is 36", media_type="text/plain")
        assert cited.warnings == (cite_min_confidence_warning(0.5, 1),)
        assert cited.as_dict()["warnings"] == list(cited.warnings)
        with Extractor(Person, _plain_model(), max_pages=2) as extractor:
            many = extractor.extract_many_with_results(
                [ExtractionInput(_three_page_pdf(), media_type="application/pdf")]
            )
        assert many[0].warnings == (max_pages_warning(2, 1, 3),)

    async def test_async_session_max_pages(self):
        async with AsyncExtractor(Person, _plain_model(), max_pages=2) as extractor:
            result = await extractor.extract_with_result(
                _three_page_pdf(), media_type="application/pdf"
            )
            many = await extractor.extract_many_with_results(
                [ExtractionInput(_three_page_pdf(), media_type="application/pdf")]
            )
        assert result.warnings == (max_pages_warning(2, 1, 3),)
        assert many[0].warnings == (max_pages_warning(2, 1, 3),)

    def test_batch_and_swarm(self):
        pdf = _three_page_pdf()
        results = extract_many_with_results(
            Person,
            _plain_model(),
            [ExtractionInput(pdf, media_type="application/pdf")],
            max_pages=2,
        )
        assert results[0].warnings == (max_pages_warning(2, 1, 3),)
        swarm = extract_swarm_with_results(
            Person,
            _plain_model(),
            pdf,
            media_type="application/pdf",
            pages=(1, 99),
        )
        assert isinstance(swarm.agents[0], ExtractionResult)
        assert swarm.agents[0].warnings == (
            pages_requested_warning(1, 2),
            pages_available_warning(2, 3),
        )
        oneshot = extract_with_result(
            Person,
            _plain_model(),
            pdf,
            media_type="application/pdf",
            max_pages=2,
        )
        assert oneshot.warnings == (max_pages_warning(2, 1, 3),)
