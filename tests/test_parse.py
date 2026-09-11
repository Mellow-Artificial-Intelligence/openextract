"""Local PDF parse and citation grounding."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from openextract import Citation
from openextract._parse import (
    DEFAULT_RENDER_MAX_EDGE,
    ParsedDocument,
    ParsedPage,
    ParsedSpan,
    _bbox_for_quote,
    _bbox_on_page,
    _bitmap_to_png,
    _channels_for_mode,
    _char_box,
    _char_text,
    _citation_from_value,
    _encode_png,
    _find_in_text,
    _flush_word,
    _fold_alnum,
    _ground_one,
    _iter_field_values,
    _luma_pixels,
    _needle_covers,
    _normalized_coco,
    _numeric_in_text,
    _page_size,
    _page_text,
    _pages_prompt_len,
    _parse_pdf_page,
    _pdf_box_to_coco,
    _rebind_field,
    _render_page_png,
    _render_scale,
    _split_page,
    _strip_indexes,
    _try_number,
    _union_coco,
    _union_pdf_boxes,
    _value_needles,
    _walk_fields,
    _word_spans,
    align_citations_to_window,
    find_span,
    ground_citations,
    maybe_parsed_inputs,
    parse_windows,
    parsed_image_inputs,
    parsed_run_inputs,
    parsed_window_inputs,
    parsed_window_pairs,
    try_parse_document,
)
from tests.pdf_fixture import synthetic_pdf


def _page(
    text: str,
    *spans: ParsedSpan,
    page: int = 1,
) -> ParsedPage:
    return ParsedPage(page=page, text=text, width=200, height=200, spans=spans)


def test_synthetic_pdf_attaches_parser_box_and_omits_unmatched():
    data = synthetic_pdf("Acme Corp")
    parsed = try_parse_document(data, "application/pdf")
    assert parsed is not None
    assert parsed.has_text()
    assert "Acme" in parsed.as_prompt_text()
    page, bbox = find_span(parsed, "Acme Corp")
    assert page == 1
    assert bbox is not None
    assert all(0 <= value <= 1 for value in bbox)
    assert bbox[2] > 0 and bbox[3] > 0

    cited = ground_citations(
        (
            Citation("vendor", "Acme Corp", page=99, bbox=(0.9, 0.9, 0.05, 0.05)),
            Citation("missing", "no-such-span-in-doc", page=1),
        ),
        parsed,
    )
    assert cited[0].page == 1
    assert cited[0].bbox == bbox
    assert cited[1].page == 1
    assert cited[1].bbox is None


def test_grounding_stamps_page_and_backfills_field_value():
    parsed = ParsedDocument(
        pages=(
            _page(
                "Total 12.50",
                ParsedSpan("Total", 1, (0.1, 0.1, 0.2, 0.05)),
                ParsedSpan("12.50", 1, (0.4, 0.1, 0.2, 0.05)),
            ),
        )
    )
    grounded = ground_citations(
        (),
        parsed,
        {
            "ghost": "zzz-not-in-doc",
            "total": "12.50",
            "label": "Total",
            "empty": None,
            "flag": True,
        },
    )
    fields = {item.field: item for item in grounded}
    assert fields["total"].page == 1
    assert fields["total"].bbox == pytest.approx((0.4, 0.1, 0.2, 0.05))
    assert fields["label"].quote == "Total"
    assert fields["label"].bbox == pytest.approx((0.1, 0.1, 0.2, 0.05))


def test_grounding_uses_field_value_when_quote_and_page_are_wrong():
    parsed = ParsedDocument(
        pages=(
            _page("Total 99", ParsedSpan("99", 1, (0.1, 0.1, 0.1, 0.05)), page=1),
            _page(
                "Acme Corp",
                ParsedSpan("Acme", 2, (0.2, 0.2, 0.2, 0.05)),
                ParsedSpan("Corp", 2, (0.4, 0.2, 0.2, 0.05)),
                page=2,
            ),
        )
    )
    grounded = ground_citations(
        (Citation("vendor", "the supplier name", page=1),),
        parsed,
        {"vendor": "Acme Corp"},
    )
    assert grounded[0].field == "vendor"
    assert grounded[0].page == 2
    assert grounded[0].bbox == pytest.approx((0.2, 0.2, 0.4, 0.05))


def test_grounding_prefers_value_page_over_common_quote_on_hint():
    parsed = ParsedDocument(
        pages=(
            _page("Total 1", ParsedSpan("Total", 1, (0.1, 0.1, 0.2, 0.05)), page=1),
            _page(
                "Total 12.50",
                ParsedSpan("12.50", 2, (0.4, 0.1, 0.2, 0.05)),
                page=2,
            ),
        )
    )
    grounded = ground_citations(
        (Citation("total", "Total", page=1),),
        parsed,
        {"total": "12.50"},
    )
    assert grounded[0].page == 2
    assert grounded[0].bbox == pytest.approx((0.4, 0.1, 0.2, 0.05))


def test_numeric_span_attaches_box_for_plain_extracted_number():
    parsed = ParsedDocument(
        pages=(
            _page(
                "Paid $1,234.00",
                ParsedSpan("$1,234.00", 1, (0.2, 0.1, 0.3, 0.05)),
            ),
        )
    )
    grounded = ground_citations(
        (Citation("amount", "1234", page=1),),
        parsed,
        {"amount": 1234},
    )
    assert grounded[0].page == 1
    assert grounded[0].bbox == pytest.approx((0.2, 0.1, 0.3, 0.05))


def test_rebind_drops_missing_array_index_and_dedupes_boxes():
    parsed = ParsedDocument(
        pages=(
            _page(
                "qty 3",
                ParsedSpan("3", 1, (0.5, 0.1, 0.05, 0.05)),
            ),
        )
    )
    grounded = ground_citations(
        (
            Citation("qty", "3", page=1),
            Citation("lines[0].qty", "3", page=1, bbox=None),
        ),
        parsed,
        {"lines": [{"qty": 3}]},
    )
    assert [item.field for item in grounded] == ["lines[0].qty"]
    assert grounded[0].bbox == pytest.approx((0.5, 0.1, 0.05, 0.05))
    duped = ground_citations(
        (
            Citation("ghost", "nope", page=1),
            Citation("ghost", "3", page=1),
            Citation("ghost", "still-nope", page=1),
        ),
        parsed,
    )
    ghosts = [item for item in duped if item.field == "ghost"]
    assert len(ghosts) == 1
    assert ghosts[0].bbox == pytest.approx((0.5, 0.1, 0.05, 0.05))


def test_align_citations_to_window_stamps_single_page():
    window = ParsedDocument(pages=(_page("Acme Corp", page=5),))
    citations = (
        Citation("vendor", "Acme Corp", page=1),
        Citation("total", "12", page=None),
        Citation("ok", "x", page=5),
    )
    aligned = align_citations_to_window(citations, window)
    assert aligned[0].page == 5
    assert aligned[1].page == 5
    assert aligned[2].page == 5
    assert align_citations_to_window(citations, None) == citations
    empty = ParsedDocument(pages=())
    assert align_citations_to_window(citations, empty) == citations
    multi = ParsedDocument(pages=(_page("a", page=2), _page("b", page=3)))
    assert align_citations_to_window(citations, multi) == citations


def test_parsed_window_pairs_and_mismatch_fallback(monkeypatch):
    parsed = ParsedDocument(pages=(_page("hello", page=1),))
    pairs = parsed_window_pairs(parsed, ["fallback"])
    assert len(pairs) == 1
    assert pairs[0][1] is parsed
    assert parsed_window_pairs(None, ["fallback"]) == [(["fallback"], None)]
    assert parsed_window_pairs(ParsedDocument(pages=()), ["fallback"]) == [(["fallback"], None)]
    calls = {"n": 0}

    def _windows(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return (parsed,)
        return ()

    monkeypatch.setattr("openextract._parse.parse_windows", _windows)
    mismatched = parsed_window_pairs(parsed, ["fallback"])
    assert mismatched == [(["fallback"], parsed)]


def test_maybe_parsed_inputs_swaps_in_page_text():
    data = synthetic_pdf("Acme Corp")
    inputs, parsed = maybe_parsed_inputs(data, "application/pdf", parse=True)
    assert parsed is not None
    assert inputs is not None
    assert inputs == parsed_run_inputs(parsed)
    assert "--- Page 1 ---" in inputs[1]
    none_inputs, none_parsed = maybe_parsed_inputs(data, "application/pdf", parse=False)
    assert none_inputs is None and none_parsed is None
    assert maybe_parsed_inputs(b"hello", "text/plain", parse=True) == (None, None)


def test_try_parse_rejects_invalid_and_non_pdf(monkeypatch):
    assert try_parse_document(b"not-a-pdf", "text/plain") is None
    assert try_parse_document(b"%PDF-not-really", "application/pdf") is None
    assert try_parse_document(b"%PDF-magic", None) is None

    def _boom(name, *args, **kwargs):
        if name == "pypdfium2":
            raise ImportError("missing extra")
        return orig_import(name, *args, **kwargs)

    import builtins

    orig_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", _boom)
    assert try_parse_document(synthetic_pdf(), "application/pdf") is None


def test_page_helpers_cover_api_fallbacks():
    sized = SimpleNamespace(get_size=lambda: (100, 50))
    assert _page_size(sized) == (100.0, 50.0)
    wide = SimpleNamespace(get_width=lambda: 10, get_height=lambda: 20)
    assert _page_size(wide) == (10.0, 20.0)

    assert _page_text(SimpleNamespace()) == ""
    assert _page_text(SimpleNamespace(get_text_bounded=lambda: "  ")) == ""
    bounded = SimpleNamespace(get_text_bounded=lambda: "Hi", get_text_range=lambda: "x")
    assert _page_text(bounded) == "Hi"
    assert _page_text(SimpleNamespace(get_text_range=lambda: "Range")) == "Range"
    assert _page_text(SimpleNamespace(get_text_range=lambda: None)) == ""

    class _Range:
        def get_text_range(self, *args, **kwargs):
            if args == (0, 1):
                raise TypeError("kwargs only")
            if kwargs == {"index": 0, "count": 1}:
                return "A"
            raise TypeError("nope")

    assert _char_text(_Range(), 0) == "A"
    assert _char_text(SimpleNamespace(), 0) == ""

    class _KwargsOnly:
        def get_text_range(self, **kwargs):
            raise TypeError("still no")

    assert _char_text(_KwargsOnly(), 0) == ""
    assert _char_box(SimpleNamespace(), 0) is None
    assert _char_box(SimpleNamespace(get_charbox=lambda _i: None), 0) is None
    assert _char_box(SimpleNamespace(get_charbox=lambda _i: (1, 2)), 0) is None

    def _raise(_i):
        raise RuntimeError("box")

    assert _char_box(SimpleNamespace(get_charbox=_raise), 0) is None
    assert _char_box(SimpleNamespace(get_charbox=lambda _i: (1.0, 2.0, 3.0, 4.0)), 0) == (
        1.0,
        2.0,
        3.0,
        4.0,
    )


def test_box_helpers_drop_invalid_geometry():
    assert _flush_word([], [], 1, 10, 10) is None
    assert _union_pdf_boxes([], 10, 10) is None
    assert _union_pdf_boxes([(0, 0, 1, 1)], 0, 10) is None
    assert _pdf_box_to_coco(0, 0, 10, 10, 10, 10) == (0.0, 0.0, 1.0, 1.0)
    assert _normalized_coco(0.1, 0.1, 0, 0.1) is None
    assert _normalized_coco(-0.1, 0.1, 0.1, 0.1) is None
    assert _normalized_coco(0.8, 0.1, 0.3, 0.1) is None
    assert _union_coco([]) is None
    assert _union_coco([(0.1, 0.2, 0.1, 0.1), (0.2, 0.25, 0.1, 0.1)]) == pytest.approx(
        (0.1, 0.2, 0.2, 0.15)
    )


def test_find_span_exact_fuzzy_and_hint():
    page = _page(
        "Quarterly revenue beat",
        ParsedSpan("Quarterly", 1, (0.1, 0.1, 0.2, 0.05)),
        ParsedSpan("revenue", 1, (0.3, 0.1, 0.2, 0.05)),
        ParsedSpan("beat", 1, (0.6, 0.1, 0.1, 0.05)),
        page=2,
    )
    other = _page("unrelated", page=1)
    parsed = ParsedDocument(pages=(other, page))
    assert find_span(parsed, None) == (None, None)
    assert find_span(parsed, "   ")[0] is None
    loc_page, bbox = find_span(parsed, "revenue beat", hinted_page=2)
    assert loc_page == 2
    assert bbox is not None
    fuzzy_page, _bbox = find_span(parsed, "qarterly revene")
    assert fuzzy_page == 2
    assert _find_in_text("abc", "x") is None
    assert _find_in_text("abc", "z") is None
    assert find_span(parsed, "zzz", hinted_page=2) == (2, None)
    value_page, value_bbox = find_span(parsed, None, field_value="revenue beat")
    assert value_page == 2
    assert value_bbox is not None


def test_ground_citations_without_parse_drops_model_bbox():
    cited = ground_citations((Citation("vendor", "Acme", 1, (0.1, 0.2, 0.3, 0.04)),), None)
    assert cited == (Citation("vendor", "Acme", 1, None),)


def test_parse_pdf_page_error_is_none(monkeypatch):
    pypdfium2 = pytest.importorskip("pypdfium2")

    class _Boom:
        def __len__(self):
            return 1

        def __getitem__(self, _index):
            raise RuntimeError("page")

        def close(self):
            return None

    monkeypatch.setattr(pypdfium2, "PdfDocument", lambda _data: _Boom())
    assert try_parse_document(b"%PDF-1.1", "application/x-pdf") is None

    class _Empty:
        def __len__(self):
            return 0

        def close(self):
            return None

    monkeypatch.setattr(pypdfium2, "PdfDocument", lambda _data: _Empty())
    assert try_parse_document(b"%PDF-1.1", "application/pdf") is None

    class _Chars:
        def count_chars(self):
            return 4

        def get_text_range(self, index, count=1):
            return "A B "[index]

        def get_charbox(self, index):
            return (float(index), 0.0, float(index + 1), 1.0)

    words = list(_word_spans(_Chars(), 1, 10, 10))
    assert [word.text for word in words] == ["A", "B"]
    assert _flush_word(["Z"], [], 1, 10, 10).bbox is None


def test_walk_nested_fields_and_short_quote():
    parsed = ParsedDocument(
        pages=(
            _page(
                "Ada 36 qty 1",
                ParsedSpan("Ada", 1, (0.1, 0.1, 0.1, 0.1)),
                ParsedSpan("1", 1, (0.5, 0.1, 0.05, 0.05)),
            ),
        )
    )
    grounded = ground_citations(
        (Citation("name", "A", page=1),),
        parsed,
        {"lines": [{"qty": 1}], "skip": [], "name": "Ada"},
    )
    assert grounded[0].page == 1
    assert any(item.field == "lines[0].qty" for item in grounded)

    class _Out(BaseModel):
        vendor: str

    assert list(_iter_field_values(_Out(vendor="Ada"))) == [("vendor", "Ada")]
    assert list(_iter_field_values("nope")) == []
    assert list(_walk_fields({1: "x"}, "")) == []
    assert list(_walk_fields("   ", "blank")) == []
    already = ground_citations((Citation("total", "12.50", 1),), parsed, {"total": "12.50"})
    assert len(already) == 1
    assert _citation_from_value("missing", "zzz", parsed) is None
    money = ParsedDocument(
        pages=(
            _page(
                "Paid 1,234.00",
                ParsedSpan("1,234.00", 1, (0.2, 0.1, 0.3, 0.05)),
            ),
        )
    )
    from_int = ground_citations((), money, {"amount": 1234})
    from_float = ground_citations((), money, {"amount": 1234.0})
    assert from_int[0].page == 1
    assert from_int[0].quote in {"1234", "1,234"}
    assert from_float[0].page == 1
    assert "1,234" in _value_needles("1234.0")
    assert _value_needles("Acme") == ("Acme",)
    assert _value_needles("inf") == ("inf",)
    assert _value_needles("nan") == ("nan",)
    assert "$12" in _value_needles("$12") and "12" in _value_needles("$12")
    assert _value_needles("1000000000000000") == ("1000000000000000",)
    assert "12.50" in _value_needles("12.5")
    none_page = _ground_one(Citation("x", "zzz"), parsed)
    assert none_page.page is None and none_page.bbox is None
    assert _find_in_text("abc", "   ") is None
    assert _bbox_for_quote(_page("Ada"), "   ") is None
    spaced = _page(
        "Acme Corp",
        ParsedSpan("  ", 1, None),
        ParsedSpan("Acme", 1, None),
        ParsedSpan("Other", 1, (0.1, 0.1, 0.1, 0.1)),
    )
    assert _bbox_for_quote(spaced, "Acme") is None
    fuzzy_page = _page(
        "zzz Acme",
        ParsedSpan("zzz", 1, None),
        ParsedSpan("Acme", 1, (0.2, 0.2, 0.2, 0.1)),
    )
    assert _bbox_for_quote(fuzzy_page, "Acmee") == pytest.approx((0.2, 0.2, 0.2, 0.1))
    punct = _page(
        "Acme, Inc.",
        ParsedSpan("Acme,", 1, (0.1, 0.1, 0.2, 0.05)),
        ParsedSpan("Inc.", 1, (0.3, 0.1, 0.15, 0.05)),
    )
    assert _bbox_for_quote(punct, "Acme Inc") is not None
    assert list(_walk_fields(object(), "x")) == []
    assert _rebind_field("vendor", None, {"vendor": "Acme"}) == "vendor"
    assert _rebind_field("qty", "3", {"lines[0].qty": "3", "other": "x"}) == "lines[0].qty"
    assert _rebind_field("qty", "3", {"lines[0].qty": "1", "lines[1].qty": "3"}) == "lines[1].qty"
    assert _rebind_field("qty", "zzz", {"lines[0].qty": "1", "lines[1].qty": "2"}) == "qty"
    assert _rebind_field("qty", None, {"lines[0].qty": "1", "lines[1].qty": "2"}) == "qty"
    assert _strip_indexes("lines[0].qty") == "lines[].qty"
    assert _strip_indexes("items[12]") == "items[]"
    assert _strip_indexes("arr[") == "arr["
    assert _strip_indexes("arr[x]") == "arr[x]"
    assert _try_number("(12.5)") == pytest.approx(-12.5)
    assert _try_number("inf") is None
    assert _try_number("not-a-number") is None
    assert _numeric_in_text("Paid $1,234.00", "1234")
    assert not _numeric_in_text("hello", "1234")
    assert not _numeric_in_text("1,234", "abc")
    assert _needle_covers("acme,", "Acme")
    assert not _needle_covers("acme", "")
    assert not _needle_covers("$$$", "!!!")
    assert _fold_alnum("Acme,") == "acme"
    assert _bbox_on_page(ParsedDocument(pages=(_page("x"),)), 9, ("x",)) is None


def test_word_span_space_and_missing_box():
    class _Chars:
        def count_chars(self):
            return 5

        def get_text_range(self, index, count=1):
            return "A  B "[index]

        def get_charbox(self, index):
            return None

    words = list(_word_spans(_Chars(), 1, 10, 10))
    assert [word.text for word in words] == ["A", "B"]

    class _Page:
        def get_size(self):
            return 10, 10

        def get_textpage(self):
            return SimpleNamespace(
                get_text_bounded=lambda: "Hi",
                count_chars=lambda: 0,
            )

    parsed_page = _parse_pdf_page([_Page()], 0)
    assert parsed_page.text == "Hi"


def test_pdf_close_optional(monkeypatch):
    pypdfium2 = pytest.importorskip("pypdfium2")

    class _NoClose:
        def __len__(self):
            return 1

        def __getitem__(self, _index):
            raise RuntimeError("page")

    monkeypatch.setattr(pypdfium2, "PdfDocument", lambda _data: _NoClose())
    assert try_parse_document(b"%PDF-1.1", "application/pdf") is None

    class _TextPage:
        def get_text_bounded(self):
            return "Hi"

        def count_chars(self):
            return 0

    class _Page:
        def get_size(self):
            return 10, 10

        def get_textpage(self):
            return _TextPage()

    class _Doc:
        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return _Page()

    monkeypatch.setattr(pypdfium2, "PdfDocument", lambda _data: _Doc())
    parsed = try_parse_document(b"%PDF-1.1", "application/pdf")
    assert parsed is not None
    assert parsed.has_text()


def test_parse_windows_splits_and_caps_large_multipage():
    pages = tuple(_page("x" * 50, page=index) for index in (1, 2, 3, 4))
    parsed = ParsedDocument(pages=pages)
    assert parse_windows(parsed, max_chars=10_000, max_pages=10) == (parsed,)
    deck = ParsedDocument(pages=tuple(_page("slide", page=index) for index in range(1, 6)))
    deck_windows = parse_windows(deck, max_chars=10_000)
    assert len(deck_windows) == 5
    assert [page.page for window in deck_windows for page in window.pages] == [1, 2, 3, 4, 5]
    veralto = ParsedDocument(
        pages=tuple(_page(f"Q4 slide {index} " * 20, page=index) for index in range(1, 21))
    )
    assert _pages_prompt_len(veralto.pages) < 80_000
    veralto_windows = parse_windows(veralto)
    assert len(veralto_windows) == 20
    assert all(len(window.pages) == 1 for window in veralto_windows)
    empty = ParsedDocument(pages=())
    assert parse_windows(empty) == (empty,)
    assert _pages_prompt_len(()) == 0
    windows = parse_windows(parsed, max_chars=80)
    assert len(windows) >= 2
    assert all(_pages_prompt_len(window.pages) <= 80 for window in windows)
    assert [page.page for window in windows for page in window.pages] == [1, 2, 3, 4]
    oversized = ParsedDocument(pages=(_page("y" * 200, page=1),))
    sliced = parse_windows(oversized, max_chars=10)
    assert len(sliced) >= 2
    assert all(_pages_prompt_len(window.pages) <= 10 or len(window.pages) == 1 for window in sliced)
    assert all(page.page == 1 for window in sliced for page in window.pages)
    empty = _page("", page=1)
    assert _split_page(empty, 1) == (empty,)
    lined = _page("aaaa\nbbbb\ncccc\ndddd\neeee", page=3)
    lined_slices = _split_page(lined, 20)
    assert len(lined_slices) >= 2
    assert all(slice_page.page == 3 for slice_page in lined_slices)
    assert "".join(slice_page.text for slice_page in lined_slices) == lined.text
    fallback = ["keep-me"]
    assert parsed_window_inputs(None, fallback) == [fallback]
    assert parsed_window_inputs(None, fallback)[0] is fallback
    assert parsed_window_inputs(parsed, fallback, max_chars=10_000, max_pages=10)[0] is fallback
    chunked = parsed_window_inputs(parsed, fallback, max_chars=80)
    assert len(chunked) >= 2
    assert all(len(window[1]) <= 80 or "--- Page " in window[1] for window in chunked)


def test_chunk_span_still_gets_parser_box():
    parsed = ParsedDocument(
        pages=(
            _page("noise " * 40, page=1),
            _page(
                "Acme Corp",
                ParsedSpan("Acme", 2, (0.1, 0.2, 0.2, 0.05)),
                ParsedSpan("Corp", 2, (0.3, 0.2, 0.2, 0.05)),
                page=2,
            ),
        )
    )
    windows = parse_windows(parsed, max_chars=40)
    assert len(windows) >= 2
    match = next(window for window in windows if "Acme Corp" in window.as_prompt_text())
    page, bbox = find_span(match, "Acme Corp")
    assert page == 2
    assert bbox is not None
    grounded = ground_citations((Citation("vendor", "Acme Corp", page=2),), parsed)
    assert grounded[0].bbox is not None


def test_oversized_page_split_keeps_parser_box():
    parsed = ParsedDocument(
        pages=(
            _page(
                "noise " * 40 + "Acme Corp " + "tail " * 40,
                ParsedSpan("Acme", 1, (0.1, 0.2, 0.2, 0.05)),
                ParsedSpan("Corp", 1, (0.3, 0.2, 0.2, 0.05)),
                page=1,
            ),
        )
    )
    windows = parse_windows(parsed, max_chars=40)
    assert len(windows) >= 2
    assert any("Acme Corp" in window.as_prompt_text() for window in windows)
    page, bbox = find_span(parsed, "Acme Corp")
    assert page == 1
    assert bbox is not None


def test_concurrent_pdf_parse_does_not_crash():
    data = synthetic_pdf("Hello concurrent parse")
    errors: list[BaseException] = []
    results: list[ParsedDocument | None] = []

    def _worker() -> None:
        try:
            results.append(try_parse_document(data, "application/pdf"))
        except BaseException as exc:  # noqa: BLE001 - crash is the failure mode
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert all(parsed is not None and parsed.has_text() for parsed in results)


def test_extract_chunks_large_parse_via_extract_once(monkeypatch, mocker):
    from pydantic_ai.models.test import TestModel

    import openextract._extract as extract_mod
    from openextract import extract

    monkeypatch.setattr("openextract._parse.DEFAULT_PARSE_WINDOW_CHARS", 40)

    class _Vendor(BaseModel):
        vendor: str

    pdf = synthetic_pdf(pages=["AAAA " * 30, "Acme Corp " + "BBBB " * 30])
    spy = mocker.spy(extract_mod, "_extract_once")
    model = TestModel(
        custom_output_args={
            "output": {"vendor": "Acme Corp"},
            "citations": [{"field": "vendor", "quote": "Acme Corp", "page": 2}],
        }
    )
    result = extract(_Vendor, model, pdf, media_type="application/pdf", cite=True)
    assert result.vendor == "Acme Corp"
    assert spy.call_count >= 2
    for call in spy.call_args_list:
        prompt = call.args[1][1]
        assert isinstance(prompt, str)
        assert len(prompt) <= 200
        assert "--- Page " in prompt


def test_empty_text_pdf_renders_page_images_not_file_upload():
    from pydantic_ai import BinaryContent

    data = synthetic_pdf(pages=["", ""])
    parsed = try_parse_document(data, "application/pdf")
    assert parsed is not None
    assert not parsed.has_text()
    assert parsed.has_images()
    assert all(page.image and page.image.startswith(b"\x89PNG") for page in parsed.pages)
    assert all(page.image and len(page.image) < 20_000 for page in parsed.pages)
    inputs, got = maybe_parsed_inputs(data, "application/pdf", parse=True)
    assert got is parsed or (got is not None and got.has_images())
    assert inputs is not None
    pngs = [
        part
        for part in inputs
        if isinstance(part, BinaryContent) and part.media_type == "image/png"
    ]
    assert pngs
    assert all(
        not (isinstance(part, BinaryContent) and part.media_type == "application/pdf")
        for part in inputs
    )
    fallback = ["pdf-upload"]
    windows = parsed_window_inputs(got, fallback)
    assert windows[0] is not fallback
    assert all(
        any(isinstance(part, BinaryContent) and part.media_type == "image/png" for part in window)
        for window in windows
    )
    headers_only = ParsedDocument(pages=(_page("", page=1), _page("", page=2)))
    assert not headers_only.has_images()
    header_windows = parsed_window_inputs(headers_only, fallback)
    assert header_windows[0] is not fallback
    assert all("--- Page " in window[1] for window in header_windows)
    assert parsed_image_inputs(headers_only)[0].startswith("Extract")


def test_render_and_png_helpers(monkeypatch):
    assert _channels_for_mode("L") == 1
    assert _channels_for_mode("RGB") == 3
    assert _channels_for_mode("BGR") == 3
    assert _channels_for_mode("BGRA") == 4
    assert _channels_for_mode("nope") == 0
    assert _render_scale(3300, 2550) == pytest.approx(DEFAULT_RENDER_MAX_EDGE / 3300)
    assert _render_scale(100, 80) == 1.0
    assert _render_scale(0, 0) == 1.0
    assert _luma_pixels(b"\x80", 1, 1, 1) == (b"\x80", 1)
    assert _luma_pixels(b"\xff\xff\xff", 1, 1, 3)[1] == 1
    assert _luma_pixels(b"\x10\x20\x30\x40", 1, 1, 4)[1] == 1
    png = _encode_png(1, 1, b"\xff\x00\x00", 3)
    assert png.startswith(b"\x89PNG")
    assert _render_page_png(SimpleNamespace()) is None

    class _BadRender:
        def render(self, **_kwargs):
            raise RuntimeError("render")

    assert _render_page_png(_BadRender()) is None

    class _TypeThenFail:
        def render(self, **kwargs):
            if "rev_byteorder" in kwargs:
                raise TypeError("no rev")
            raise RuntimeError("still no")

    assert _render_page_png(_TypeThenFail()) is None

    class _TypeThenOk:
        def render(self, **kwargs):
            if "rev_byteorder" in kwargs:
                raise TypeError("no rev")
            return SimpleNamespace(
                width=1, height=1, mode="BGR", n_channels=3, stride=3, buffer=b"\x00\x11\x22"
            )

    assert _render_page_png(_TypeThenOk()).startswith(b"\x89PNG")
    assert _bitmap_to_png(SimpleNamespace(width=0, height=1, buffer=b"x")) is None
    tight = SimpleNamespace(width=1, height=1, mode="RGB", n_channels=3, stride=1, buffer=b"abc")
    assert _bitmap_to_png(tight) is None
    rgba = SimpleNamespace(
        width=1, height=1, mode="BGRA", n_channels=4, stride=4, buffer=b"\x01\x02\x03\x04"
    )
    assert _bitmap_to_png(rgba).startswith(b"\x89PNG")
    gray = SimpleNamespace(width=1, height=1, mode="L", n_channels=0, stride=1, buffer=b"\x80")
    assert _bitmap_to_png(gray).startswith(b"\x89PNG")

    class _Closeable:
        closed = False

        def close(self):
            self.closed = True

    bitmap = _Closeable()
    bitmap.width = 1
    bitmap.height = 1
    bitmap.mode = "RGB"
    bitmap.n_channels = 3
    bitmap.stride = 3
    bitmap.buffer = b"\x10\x20\x30"

    class _Page:
        def render(self, **_kwargs):
            return bitmap

    assert _render_page_png(_Page()).startswith(b"\x89PNG")
    assert bitmap.closed is True

    class _SizedPage:
        def get_size(self):
            return 2550.0, 3300.0

        def render(self, scale=1, **_kwargs):
            assert scale == pytest.approx(DEFAULT_RENDER_MAX_EDGE / 3300)
            return SimpleNamespace(
                width=1, height=1, mode="RGB", n_channels=3, stride=3, buffer=b"\x10\x20\x30"
            )

    assert _render_page_png(_SizedPage()).startswith(b"\x89PNG")


def test_maybe_parsed_inputs_headers_when_scan_has_no_image(monkeypatch):
    empty = ParsedDocument(pages=(_page("", page=1),))
    monkeypatch.setattr("openextract._parse.try_parse_document", lambda *_a, **_k: empty)
    inputs, parsed = maybe_parsed_inputs(b"%PDF", "application/pdf", parse=True)
    assert parsed is empty
    assert inputs == parsed_run_inputs(empty)
    none = ParsedDocument(pages=())
    monkeypatch.setattr("openextract._parse.try_parse_document", lambda *_a, **_k: none)
    assert maybe_parsed_inputs(b"%PDF", "application/pdf", parse=True) == (None, none)


def test_split_page_keeps_image_bytes():
    page = ParsedPage(
        page=1, text="aaaa bbbb cccc dddd", width=1, height=1, spans=(), image=b"\x89PNG"
    )
    slices = _split_page(page, 20)
    assert len(slices) >= 2
    assert all(slice_page.image == b"\x89PNG" for slice_page in slices)
