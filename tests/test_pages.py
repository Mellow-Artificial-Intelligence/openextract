"""Optional PDF page-range filtering on parse, extract, and CLI."""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from pydantic_ai.models.test import TestModel

from openextract import (
    AsyncExtractor,
    ExtractionInput,
    Extractor,
    ExtractProgress,
    extract,
    extract_async,
    extract_many,
    extract_many_async,
    extract_many_with_results,
    extract_many_with_results_async,
    extract_swarm,
    extract_with_usage,
    extract_with_usage_async,
)
from openextract._cli import main
from openextract._config import (
    _select_pages,
    _validate_max_pages,
    _validate_pages,
    parse_page_range,
)
from openextract._parse import maybe_parsed_inputs, parsed_window_inputs, try_parse_document
from tests.pdf_fixture import synthetic_pdf


class Person(BaseModel):
    name: str
    age: int


_BASE_ARGS = ["--schema", "tests.test_cli:_FixtureSchema", "--model", "xai:grok-4.3"]


def _plain_model() -> TestModel:
    return TestModel(custom_output_args={"name": "Ada", "age": 36})


def _cited_model() -> TestModel:
    return TestModel(
        custom_output_args={
            "output": {"name": "Ada", "age": 36},
            "citations": [{"field": "name", "quote": "Ada", "page": 1}],
        }
    )


def _three_page_pdf() -> bytes:
    return synthetic_pdf(pages=["AAAA " * 30, "Ada Lovelace " + "BBBB " * 30, "CCCC " * 30])


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("1-3,5,8", (1, 2, 3, 5, 8)),
        (" 1 - 3 , 5 ", (1, 2, 3, 5)),
        ("5,1-2", (5, 1, 2)),
        ("1-1", (1,)),
        ("8", (8,)),
        ("1-3,2", (1, 2, 3)),
        ("01,2", (1, 2)),
    ],
)
def test_parse_page_range_accepts_compact_syntax(spec, expected):
    assert parse_page_range(spec) == expected


@pytest.mark.parametrize(
    "spec",
    [
        "",
        "   ",
        None,
        1,
        ",",
        "1,",
        ",1",
        "1-2-3",
        "3-1",
        "0",
        "0-2",
        "1-",
        "-1",
        "abc",
        "1.5",
        "+2",
        "1-a",
    ],
)
def test_parse_page_range_rejects_invalid_tokens(spec):
    with pytest.raises(ValueError, match="pages|invalid page"):
        parse_page_range(spec)  # type: ignore[arg-type]


def test_validate_pages_none_and_unique_order():
    assert _validate_pages(None) is None
    assert _validate_pages([3, 1, 3, 2]) == (3, 1, 2)
    assert _validate_pages(range(1, 3)) == (1, 2)


@pytest.mark.parametrize("value", ["1,2", b"1", 1, True, [0], [-1], [1.5], [True], []])
def test_validate_pages_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="pages must"):
        _validate_pages(value)


def test_maybe_parsed_inputs_keeps_only_requested_pages():
    data = synthetic_pdf(pages=["page-one vendor", "page-two secret", "page-three total"])
    inputs, parsed = maybe_parsed_inputs(data, "application/pdf", parse=True, pages=(1, 3))
    assert parsed is not None
    assert [page.page for page in parsed.pages] == [1, 3]
    prompt = parsed.as_prompt_text()
    assert "page-one vendor" in prompt
    assert "page-three total" in prompt
    assert "page-two secret" not in prompt
    assert "--- Page 2 ---" not in prompt
    assert inputs is not None
    assert "page-two secret" not in inputs[1]
    windows = parsed_window_inputs(parsed, inputs)
    assert len(windows) == 2
    joined = "\n".join(str(part) for window in windows for part in window)
    assert "page-two secret" not in joined
    assert "page-one vendor" in joined
    assert "page-three total" in joined


def test_try_parse_skips_unlisted_pages():
    data = _three_page_pdf()
    parsed = try_parse_document(data, "application/pdf", pages=(2, 99))
    assert parsed is not None
    assert [page.page for page in parsed.pages] == [2]
    full = try_parse_document(data, "application/pdf")
    assert full is not None
    assert len(full.pages) == 3


def test_out_of_range_only_raises():
    data = synthetic_pdf("only page")
    with pytest.raises(ValueError, match="does not match any page"):
        maybe_parsed_inputs(data, "application/pdf", parse=True, pages=(9, 10))


def test_pages_ignored_for_non_pdf():
    assert maybe_parsed_inputs(b"hello", "text/plain", parse=True, pages=(1,)) == (None, None)


def test_extract_pages_shrinks_parse_windows():
    events: list[ExtractProgress] = []
    result = extract(
        Person,
        _cited_model(),
        _three_page_pdf(),
        media_type="application/pdf",
        cite=True,
        pages=[2],
        on_progress=events.append,
    )
    assert result == Person(name="Ada", age=36)
    assert [event.page for event in events] == [2]
    assert {event.total for event in events} == {1}


def test_pages_forces_direct_style_parse():
    events: list[ExtractProgress] = []
    result = extract(
        Person,
        TestModel(custom_output_args={"name": "Ada", "age": 36}),
        _three_page_pdf(),
        media_type="application/pdf",
        pages=(1, 3),
        on_progress=events.append,
    )
    assert result == Person(name="Ada", age=36)
    assert [event.page for event in events] == [1, 3]


def test_default_none_keeps_every_page():
    events: list[ExtractProgress] = []
    extract(
        Person,
        _cited_model(),
        _three_page_pdf(),
        media_type="application/pdf",
        cite=True,
        on_progress=events.append,
    )
    assert [event.page for event in events] == [1, 2, 3]


def test_invalid_pages_raise_at_call_time():
    model = TestModel(custom_output_args={"name": "Ada", "age": 36})
    with pytest.raises(ValueError, match="pages must"):
        extract(Person, model, b"x", media_type="text/plain", pages=[])
    with pytest.raises(ValueError, match="pages must"):
        Extractor(Person, model, pages=[0])
    with pytest.raises(ValueError, match="pages must"):
        extract_many(
            Person,
            model,
            [ExtractionInput(b"x", media_type="text/plain")],
            pages="1-2",
        )
    with pytest.raises(ValueError, match="pages must"):
        extract_swarm(Person, model, b"x", media_type="text/plain", pages=[True])


def test_session_constructor_and_per_call_override():
    model = _cited_model()
    pdf = _three_page_pdf()
    session_events: list[ExtractProgress] = []
    with Extractor(Person, model, cite=True, pages=(1,)) as extractor:
        extractor.extract(pdf, media_type="application/pdf", on_progress=session_events.append)
        override: list[ExtractProgress] = []
        extractor.extract(
            pdf, media_type="application/pdf", pages=(3,), on_progress=override.append
        )
        output, _usage = extractor.extract_with_usage(pdf, media_type="application/pdf", pages=(2,))
    assert [event.page for event in session_events] == [1]
    assert [event.page for event in override] == [3]
    assert output == Person(name="Ada", age=36)


async def test_async_extract_and_session_pages():
    model = _plain_model()
    pdf = _three_page_pdf()
    events: list[ExtractProgress] = []
    result = await extract_async(
        Person,
        model,
        pdf,
        media_type="application/pdf",
        pages=(2,),
        on_progress=events.append,
    )
    assert result == Person(name="Ada", age=36)
    assert [event.page for event in events] == [2]
    output, _usage = await extract_with_usage_async(
        Person, model, pdf, media_type="application/pdf", pages=(3,)
    )
    assert output == Person(name="Ada", age=36)
    async with AsyncExtractor(Person, model, pages=(1, 3)) as extractor:
        session_events: list[ExtractProgress] = []
        await extractor.extract(
            pdf, media_type="application/pdf", on_progress=session_events.append
        )
        assert [event.page for event in session_events] == [1, 3]
        override: list[ExtractProgress] = []
        await extractor.extract_with_usage(
            pdf, media_type="application/pdf", pages=(2,), on_progress=override.append
        )
        assert [event.page for event in override] == [2]


def test_oneshot_extraction_input_pages_override_kwargs():
    events: list[ExtractProgress] = []
    result = extract(
        Person,
        _plain_model(),
        ExtractionInput(_three_page_pdf(), media_type="application/pdf", pages=(3,)),
        pages=(1,),
        on_progress=events.append,
    )
    assert result == Person(name="Ada", age=36)
    assert [event.page for event in events] == [3]


def test_batch_item_pages_override_vs_default():
    model = _plain_model()
    pdf = _three_page_pdf()
    events: list[ExtractProgress] = []
    results = extract_many(
        Person,
        model,
        [
            ExtractionInput(pdf, media_type="application/pdf", pages=(2,)),
            ExtractionInput(pdf, media_type="application/pdf"),
        ],
        pages=(1,),
        max_concurrency=1,
        on_progress=events.append,
    )
    assert results == [Person(name="Ada", age=36), Person(name="Ada", age=36)]
    assert [event.page for event in events] == [2, 1]


def test_batch_item_max_pages_empty_set_and_invalid_pages():
    model = _plain_model()
    with pytest.raises(ValueError, match="pages must include at least one"):
        extract_many(
            Person,
            model,
            [ExtractionInput(b"x", media_type="text/plain", pages=(5, 6), max_pages=3)],
        )
    with pytest.raises(ValueError, match="pages must"):
        extract_many(
            Person,
            model,
            [ExtractionInput(b"x", media_type="text/plain", pages=[])],
        )
    with pytest.raises(ValueError, match="max_pages must"):
        extract_many(
            Person,
            model,
            [ExtractionInput(b"x", media_type="text/plain", max_pages=0)],
        )


async def test_batch_item_pages_async_and_with_results():
    model = _plain_model()
    pdf = _three_page_pdf()
    events: list[ExtractProgress] = []
    results = await extract_many_async(
        Person,
        model,
        [ExtractionInput(pdf, media_type="application/pdf", pages=(3,))],
        pages=(1,),
        on_progress=events.append,
    )
    assert results == [Person(name="Ada", age=36)]
    assert [event.page for event in events] == [3]
    rich_events: list[ExtractProgress] = []
    rich = extract_many_with_results(
        Person,
        model,
        [ExtractionInput(pdf, media_type="application/pdf", max_pages=1)],
        pages=(1, 2, 3),
        on_progress=rich_events.append,
    )
    assert [item.output for item in rich] == [Person(name="Ada", age=36)]
    assert [event.page for event in rich_events] == [1]
    async_rich_events: list[ExtractProgress] = []
    async_rich = await extract_many_with_results_async(
        Person,
        model,
        [ExtractionInput(pdf, media_type="application/pdf", pages=(2,))],
        max_pages=3,
        on_progress=async_rich_events.append,
    )
    assert [item.output for item in async_rich] == [Person(name="Ada", age=36)]
    assert [event.page for event in async_rich_events] == [2]


def test_batch_and_swarm_forward_pages():
    model = _plain_model()
    pdf = _three_page_pdf()
    batch_events: list[ExtractProgress] = []
    results = extract_many(
        Person,
        model,
        [ExtractionInput(pdf, media_type="application/pdf")],
        pages=(2,),
        on_progress=batch_events.append,
    )
    assert results == [Person(name="Ada", age=36)]
    assert [event.page for event in batch_events] == [2]
    swarm_events: list[ExtractProgress] = []
    swarm = extract_swarm(
        Person,
        model,
        pdf,
        media_type="application/pdf",
        pages=(3,),
        on_progress=swarm_events.append,
    )
    assert swarm == Person(name="Ada", age=36)
    assert [event.page for event in swarm_events] == [3]


def test_extract_with_usage_pages():
    output, usage = extract_with_usage(
        Person,
        _plain_model(),
        _three_page_pdf(),
        media_type="application/pdf",
        pages=(2,),
    )
    assert output == Person(name="Ada", age=36)
    assert usage.total_tokens >= 0


def test_cli_pages_is_forwarded(mocker, capsys):
    mock_fn = mocker.patch(
        "openextract._cli.extract",
        return_value=Person(name="Ada", age=36),
    )
    assert main(["input.pdf", *_BASE_ARGS, "--pages", "1-3,5,8"]) == 0
    assert mock_fn.call_args.kwargs["pages"] == (1, 2, 3, 5, 8)
    capsys.readouterr()


def test_cli_batch_pages_is_forwarded(mocker, capsys):
    async def _stream(*_args, **_kwargs):
        yield (0, Person(name="Ada", age=36))
        yield (1, Person(name="Ada", age=36))

    mock_stream = mocker.patch("openextract._cli._iter_extractions", side_effect=_stream)
    assert main(["a.pdf", "b.pdf", *_BASE_ARGS, "--pages", "2"]) == 0
    assert mock_stream.call_args.args[3].pages == (2,)
    capsys.readouterr()


def test_cli_swarm_pages_is_forwarded(mocker, capsys):
    plain = mocker.patch("openextract._cli.extract_swarm", return_value=Person(name="Ada", age=36))
    mocker.patch("openextract._cli.extract_swarm_with_results")
    assert main(["input.pdf", *_BASE_ARGS, "--swarm", "2", "--pages", "1,3"]) == 0
    assert plain.call_args.kwargs["pages"] == (1, 3)
    capsys.readouterr()


@pytest.mark.parametrize("spec", ["", "1-", "1,,2", "abc", "3-1"])
def test_cli_invalid_pages_returns_1(capsys, spec):
    assert main(["input.pdf", *_BASE_ARGS, "--pages", spec]) == 1
    assert "page" in capsys.readouterr().err.lower()


def test_cli_pages_and_max_pages_empty_filter_returns_1(capsys):
    assert main(["input.pdf", *_BASE_ARGS, "--pages", "5,6", "--max-pages", "3"]) == 1
    assert "pages must" in capsys.readouterr().err.lower()


def test_validate_max_pages_none_and_positive():
    assert _validate_max_pages(None) is None
    assert _validate_max_pages(1) == 1
    assert _validate_max_pages(8) == 8


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "2", [2]])
def test_validate_max_pages_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="max_pages must"):
        _validate_max_pages(value)


def test_select_pages_caps_allowlist_and_preserves_order():
    assert _select_pages(None, None) == (None, None)
    assert _select_pages(None, 3) == (None, 3)
    assert _select_pages([3, 1, 3, 5], 3) == ((3, 1), 3)
    with pytest.raises(ValueError, match="pages must include at least one"):
        _select_pages([4, 5], 3)


def test_maybe_parsed_inputs_keeps_first_n_pages():
    data = synthetic_pdf(pages=["page-one vendor", "page-two secret", "page-three total"])
    inputs, parsed = maybe_parsed_inputs(data, "application/pdf", parse=True, max_pages=2)
    assert parsed is not None
    assert [page.page for page in parsed.pages] == [1, 2]
    prompt = parsed.as_prompt_text()
    assert "page-one vendor" in prompt
    assert "page-two secret" in prompt
    assert "page-three total" not in prompt
    assert "--- Page 3 ---" not in prompt
    assert inputs is not None
    assert "page-three total" not in inputs[1]


def test_max_pages_applies_after_pages_allowlist():
    data = synthetic_pdf(pages=["page-one vendor", "page-two secret", "page-three total"])
    _, parsed = maybe_parsed_inputs(data, "application/pdf", parse=True, pages=(1, 3), max_pages=2)
    assert parsed is not None
    assert [page.page for page in parsed.pages] == [1]


def test_try_parse_max_pages_skips_later_pages():
    data = _three_page_pdf()
    parsed = try_parse_document(data, "application/pdf", max_pages=2)
    assert parsed is not None
    assert [page.page for page in parsed.pages] == [1, 2]
    with_pages = try_parse_document(data, "application/pdf", pages=(2, 3), max_pages=2)
    assert with_pages is not None
    assert [page.page for page in with_pages.pages] == [2]


def test_max_pages_ignored_for_non_pdf():
    assert maybe_parsed_inputs(b"hello", "text/plain", parse=True, max_pages=1) == (None, None)


def test_extract_max_pages_caps_parse_windows():
    events: list[ExtractProgress] = []
    result = extract(
        Person,
        _cited_model(),
        _three_page_pdf(),
        media_type="application/pdf",
        cite=True,
        max_pages=2,
        on_progress=events.append,
    )
    assert result == Person(name="Ada", age=36)
    assert [event.page for event in events] == [1, 2]
    assert {event.total for event in events} == {2}


def test_max_pages_forces_direct_style_parse():
    events: list[ExtractProgress] = []
    result = extract(
        Person,
        TestModel(custom_output_args={"name": "Ada", "age": 36}),
        _three_page_pdf(),
        media_type="application/pdf",
        max_pages=1,
        on_progress=events.append,
    )
    assert result == Person(name="Ada", age=36)
    assert [event.page for event in events] == [1]


def test_max_pages_noop_on_plain_text():
    result = extract(
        Person,
        TestModel(custom_output_args={"name": "Ada", "age": 36}),
        b"Ada is 36",
        media_type="text/plain",
        max_pages=1,
    )
    assert result == Person(name="Ada", age=36)


def test_extract_pages_then_max_pages():
    events: list[ExtractProgress] = []
    result = extract(
        Person,
        TestModel(custom_output_args={"name": "Ada", "age": 36}),
        _three_page_pdf(),
        media_type="application/pdf",
        pages=(1, 3),
        max_pages=2,
        on_progress=events.append,
    )
    assert result == Person(name="Ada", age=36)
    assert [event.page for event in events] == [1]


def test_constructor_empty_after_max_pages_raises():
    model = TestModel(custom_output_args={"name": "Ada", "age": 36})
    with pytest.raises(ValueError, match="pages must include at least one"):
        Extractor(Person, model, pages=(5, 6), max_pages=3)
    model = TestModel(custom_output_args={"name": "Ada", "age": 36})
    with pytest.raises(ValueError, match="max_pages must"):
        extract(Person, model, b"x", media_type="text/plain", max_pages=0)
    with pytest.raises(ValueError, match="max_pages must"):
        Extractor(Person, model, max_pages=True)
    with pytest.raises(ValueError, match="pages must include at least one"):
        extract_many(
            Person,
            model,
            [ExtractionInput(b"x", media_type="text/plain")],
            pages=(5, 6),
            max_pages=3,
        )
    with pytest.raises(ValueError, match="max_pages must"):
        extract_swarm(Person, model, b"x", media_type="text/plain", max_pages=-1)


def test_session_constructor_and_per_call_max_pages_override():
    model = _cited_model()
    pdf = _three_page_pdf()
    session_events: list[ExtractProgress] = []
    with Extractor(Person, model, cite=True, max_pages=1) as extractor:
        extractor.extract(pdf, media_type="application/pdf", on_progress=session_events.append)
        override: list[ExtractProgress] = []
        extractor.extract(
            pdf, media_type="application/pdf", max_pages=2, on_progress=override.append
        )
        output, _usage = extractor.extract_with_usage(
            pdf, media_type="application/pdf", pages=(2, 3), max_pages=2
        )
    assert [event.page for event in session_events] == [1]
    assert [event.page for event in override] == [1, 2]
    assert output == Person(name="Ada", age=36)


async def test_async_extract_and_session_max_pages():
    model = _plain_model()
    pdf = _three_page_pdf()
    events: list[ExtractProgress] = []
    result = await extract_async(
        Person,
        model,
        pdf,
        media_type="application/pdf",
        max_pages=2,
        on_progress=events.append,
    )
    assert result == Person(name="Ada", age=36)
    assert [event.page for event in events] == [1, 2]
    async with AsyncExtractor(Person, model, max_pages=3) as extractor:
        override: list[ExtractProgress] = []
        await extractor.extract(
            pdf, media_type="application/pdf", max_pages=1, on_progress=override.append
        )
        assert [event.page for event in override] == [1]


def test_batch_and_swarm_forward_max_pages():
    model = _plain_model()
    pdf = _three_page_pdf()
    batch_events: list[ExtractProgress] = []
    results = extract_many(
        Person,
        model,
        [ExtractionInput(pdf, media_type="application/pdf")],
        max_pages=1,
        on_progress=batch_events.append,
    )
    assert results == [Person(name="Ada", age=36)]
    assert [event.page for event in batch_events] == [1]
    swarm_events: list[ExtractProgress] = []
    swarm = extract_swarm(
        Person,
        model,
        pdf,
        media_type="application/pdf",
        max_pages=2,
        on_progress=swarm_events.append,
    )
    assert swarm == Person(name="Ada", age=36)
    assert [event.page for event in swarm_events] == [1, 2]


def test_cli_max_pages_is_forwarded(mocker, capsys):
    mock_fn = mocker.patch(
        "openextract._cli.extract",
        return_value=Person(name="Ada", age=36),
    )
    assert main(["input.pdf", *_BASE_ARGS, "--max-pages", "2"]) == 0
    assert mock_fn.call_args.kwargs["max_pages"] == 2
    capsys.readouterr()


def test_cli_batch_max_pages_is_forwarded(mocker, capsys):
    async def _stream(*_args, **_kwargs):
        yield (0, Person(name="Ada", age=36))
        yield (1, Person(name="Ada", age=36))

    mock_stream = mocker.patch("openextract._cli._iter_extractions", side_effect=_stream)
    assert main(["a.pdf", "b.pdf", *_BASE_ARGS, "--max-pages", "3"]) == 0
    assert mock_stream.call_args.args[3].max_pages == 3
    capsys.readouterr()


def test_cli_swarm_max_pages_is_forwarded(mocker, capsys):
    plain = mocker.patch("openextract._cli.extract_swarm", return_value=Person(name="Ada", age=36))
    mocker.patch("openextract._cli.extract_swarm_with_results")
    assert main(["input.pdf", *_BASE_ARGS, "--swarm", "2", "--max-pages", "4"]) == 0
    assert plain.call_args.kwargs["max_pages"] == 4
    capsys.readouterr()


@pytest.mark.parametrize("value", ["0", "-1"])
def test_cli_invalid_max_pages_returns_1(capsys, value):
    assert main(["input.pdf", *_BASE_ARGS, "--max-pages", value]) == 1
    assert "max_pages" in capsys.readouterr().err.lower()
