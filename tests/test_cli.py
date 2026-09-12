"""Tests for openextract._cli."""

import io
import json
import sys
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from openextract import (
    Citation,
    ExtractionError,
    ExtractionInput,
    ExtractionResult,
    ExtractionStyle,
    InputTooLargeError,
    ModelError,
    ProviderNotInstalledError,
    SchemaValidationError,
    UrlFetchError,
    Usage,
)
from openextract._cli import (
    _citations_payload,
    _discard_stdout,
    _resolve_schema,
    _result_citations,
    main,
)


class _FixtureSchema(BaseModel):
    name: str
    age: int


class _NotASchema:
    """Plain class that is not a pydantic BaseModel subclass."""


def _patch_extract(mocker, return_value=None, side_effect=None):
    mock_extract = mocker.patch("openextract._cli.extract")
    if side_effect is not None:
        mock_extract.side_effect = side_effect
    else:
        mock_extract.return_value = return_value
    return mock_extract


def _patch_extract_with_usage(mocker, return_value=None, side_effect=None):
    mock_fn = mocker.patch("openextract._cli.extract_with_usage")
    if side_effect is not None:
        mock_fn.side_effect = side_effect
    else:
        mock_fn.return_value = return_value
    return mock_fn


def _patch_iter_extractions(mocker, events=(), error=None):
    """Patch the CLI's batch stream with a fake indexed async iterator."""

    async def _stream(*_args, **_kwargs):
        for event in events:
            yield event
        if error is not None:
            raise error

    return mocker.patch("openextract._cli._iter_extractions", side_effect=_stream)


def _rich_result(output, input_tokens=10, output_tokens=5, citations=()):
    return ExtractionResult(
        output=output,
        usage=Usage(input_tokens, output_tokens, input_tokens + output_tokens),
        attempts=1,
        duration=0.1,
        model="xai:grok-4.3",
        media_type=None,
        source="fixture",
        citations=citations,
    )


def _patch_extract_many_with_results(mocker, return_value=None, side_effect=None):
    mock_fn = mocker.patch("openextract._cli.extract_many_with_results")
    if side_effect is not None:
        mock_fn.side_effect = side_effect
    else:
        mock_fn.return_value = return_value
    return mock_fn


_BASE_ARGS = ["--schema", "tests.test_cli:_FixtureSchema", "--model", "xai:grok-4.3"]


# ---------------------------------------------------------------------------
# _resolve_schema
# ---------------------------------------------------------------------------


class TestResolveSchema:
    def test_resolves_valid_schema(self):
        cls = _resolve_schema("tests.test_cli:_FixtureSchema")
        assert cls is _FixtureSchema

    def test_missing_colon_raises(self):
        with pytest.raises(ValueError, match="Expected format"):
            _resolve_schema("tests.test_cli._FixtureSchema")

    def test_empty_module_raises(self):
        with pytest.raises(ValueError, match="Expected format"):
            _resolve_schema(":_FixtureSchema")

    def test_empty_class_raises(self):
        with pytest.raises(ValueError, match="Expected format"):
            _resolve_schema("tests.test_cli:")

    def test_missing_class_raises(self):
        with pytest.raises(ValueError, match="not found in module"):
            _resolve_schema("tests.test_cli:DoesNotExist")

    def test_non_basemodel_raises(self):
        with pytest.raises(ValueError, match="does not refer to a Pydantic BaseModel"):
            _resolve_schema("tests.test_cli:_NotASchema")

    def test_bad_module_raises_import_error(self):
        with pytest.raises(ImportError):
            _resolve_schema("definitely_not_a_real_module_xyz:Thing")


# ---------------------------------------------------------------------------
# argparse behavior
# ---------------------------------------------------------------------------


class TestArgparse:
    def test_missing_required_args_returns_1(self, capsys):
        assert main([]) == 1
        captured = capsys.readouterr()
        assert "schema" in captured.err.lower()

    def test_missing_schema_returns_1(self, capsys):
        assert main(["input.txt", "--model", "xai:grok-4.3"]) == 1
        captured = capsys.readouterr()
        assert "schema" in captured.err.lower()

    def test_missing_model_returns_1(self, capsys):
        assert main(["input.txt", "--schema", "tests.test_cli:_FixtureSchema"]) == 1
        captured = capsys.readouterr()
        assert "model" in captured.err.lower()

    def test_invalid_output_choice_exits_nonzero(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["input.txt", *_BASE_ARGS, "--output", "yaml"])
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "output" in captured.err.lower()

    def test_no_inputs_and_no_manifest_returns_1(self, capsys):
        exit_code = main(_BASE_ARGS)
        assert exit_code == 1
        assert "input files" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# main success paths (single input)
# ---------------------------------------------------------------------------


class TestMainSuccess:
    def test_max_input_bytes_is_forwarded(self, mocker, capsys):
        fake = _FixtureSchema(name="Ada", age=36)
        mock_extract = _patch_extract(mocker, return_value=fake)

        assert main(["input.txt", *_BASE_ARGS, "--max-input-bytes", "1024"]) == 0

        assert mock_extract.call_args.kwargs["max_input_bytes"] == 1024
        capsys.readouterr()

    def test_loads_dotenv_at_application_boundary(self, mocker, capsys):
        load_dotenv = mocker.patch("openextract._cli.load_dotenv")
        _patch_extract(mocker, return_value=_FixtureSchema(name="Ada", age=36))

        assert main(["input.txt", *_BASE_ARGS]) == 0

        load_dotenv.assert_called_once_with()
        capsys.readouterr()

    def test_json_output_default(self, mocker, capsys):
        fake = _FixtureSchema(name="Ada", age=36)
        mock_extract = _patch_extract(mocker, return_value=fake)

        exit_code = main(["input.txt", *_BASE_ARGS, "--instructions", "find the person"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert '"name": "Ada"' in captured.out
        mock_extract.assert_called_once_with(
            schema=_FixtureSchema,
            model="xai:grok-4.3",
            input_file="input.txt",
            instructions="find the person",
            style="direct",
            media_type=None,
            max_input_bytes=None,
            max_retries=0,
            retry_backoff=1.0,
            retry_max_backoff=60.0,
            cite=False,
        )

    def test_repr_output(self, mocker, capsys):
        fake = MagicMock()
        fake.__repr__ = lambda self: "_FixtureSchema(name='Linus', age=54)"
        _patch_extract(mocker, return_value=fake)

        exit_code = main(["input.txt", *_BASE_ARGS, "--output", "repr"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "_FixtureSchema(name='Linus', age=54)" in captured.out

    def test_instructions_default_to_none(self, mocker):
        fake = MagicMock()
        fake.model_dump_json.return_value = "{}"
        mock_extract = _patch_extract(mocker, return_value=fake)

        exit_code = main(["input.txt", *_BASE_ARGS])

        assert exit_code == 0
        assert mock_extract.call_args.kwargs["instructions"] is None

    def test_retry_defaults(self, mocker):
        fake = MagicMock()
        fake.model_dump_json.return_value = "{}"
        mock_extract = _patch_extract(mocker, return_value=fake)

        main(["input.txt", *_BASE_ARGS])

        assert mock_extract.call_args.kwargs["max_retries"] == 0
        assert mock_extract.call_args.kwargs["retry_backoff"] == 1.0
        assert mock_extract.call_args.kwargs["retry_max_backoff"] == 60.0

    def test_retry_options_custom(self, mocker):
        fake = MagicMock()
        fake.model_dump_json.return_value = "{}"
        mock_extract = _patch_extract(mocker, return_value=fake)

        main(
            [
                "input.txt",
                *_BASE_ARGS,
                "--max-retries",
                "3",
                "--retry-backoff",
                "2.5",
                "--retry-max-backoff",
                "12.0",
            ]
        )

        assert mock_extract.call_args.kwargs["max_retries"] == 3
        assert mock_extract.call_args.kwargs["retry_backoff"] == 2.5
        assert mock_extract.call_args.kwargs["retry_max_backoff"] == 12.0

    def test_invalid_max_retries_returns_1(self, capsys):
        exit_code = main(["input.txt", *_BASE_ARGS, "--max-retries", "-1"])

        assert exit_code == 1
        assert "max_retries" in capsys.readouterr().err

    def test_invalid_retry_backoff_returns_1(self, capsys):
        exit_code = main(["input.txt", *_BASE_ARGS, "--retry-backoff", "-0.1"])

        assert exit_code == 1
        assert "retry_backoff" in capsys.readouterr().err

    def test_invalid_retry_max_backoff_returns_1(self, capsys):
        exit_code = main(["input.txt", *_BASE_ARGS, "--retry-max-backoff", "-1"])

        assert exit_code == 1
        assert "retry_max_backoff" in capsys.readouterr().err

    def test_invalid_max_concurrency_returns_1_before_extraction(self, mocker, capsys):
        mock_extract = _patch_extract(mocker)

        exit_code = main(["input.txt", *_BASE_ARGS, "--max-concurrency", "0"])

        assert exit_code == 1
        assert "max_concurrency" in capsys.readouterr().err
        mock_extract.assert_not_called()


# ---------------------------------------------------------------------------
# main error paths / exit codes
# ---------------------------------------------------------------------------


class TestMainErrorCodes:
    def test_input_too_large_returns_5(self, mocker, capsys):
        _patch_extract(mocker, side_effect=InputTooLargeError("too large"))

        exit_code = main(["input.txt", *_BASE_ARGS])

        assert exit_code == 5
        assert "too large" in capsys.readouterr().err

    def _invoke(self, mocker, exc):
        _patch_extract(mocker, side_effect=exc)
        return main(["input.txt", *_BASE_ARGS])

    def test_url_fetch_error_returns_2(self, mocker, capsys):
        assert self._invoke(mocker, UrlFetchError("404")) == 2
        assert "404" in capsys.readouterr().err

    def test_schema_validation_error_returns_3(self, mocker, capsys):
        assert self._invoke(mocker, SchemaValidationError("bad shape")) == 3
        assert "bad shape" in capsys.readouterr().err

    def test_model_error_returns_4(self, mocker, capsys):
        assert self._invoke(mocker, ModelError("upstream")) == 4
        assert "upstream" in capsys.readouterr().err

    def test_extraction_error_returns_5(self, mocker, capsys):
        assert self._invoke(mocker, ExtractionError("misc")) == 5
        assert "misc" in capsys.readouterr().err

    def test_provider_not_installed_error_returns_6(self, mocker, capsys):
        exc = ProviderNotInstalledError("Install it with: pip install 'openextract[openai]'")
        assert self._invoke(mocker, exc) == 6
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "pip install 'openextract[openai]'" in captured.err

    def test_bad_schema_module_returns_1(self, capsys):
        exit_code = main(
            ["input.txt", "--schema", "definitely_not_a_real_module_xyz:Thing", "--model", "m"]
        )
        assert exit_code == 1
        assert "error" in capsys.readouterr().err.lower()

    def test_schema_not_basemodel_returns_1(self, capsys):
        exit_code = main(
            ["input.txt", "--schema", "tests.test_cli:_NotASchema", "--model", "xai:grok-4.3"]
        )
        assert exit_code == 1
        assert "BaseModel" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# batch runs
# ---------------------------------------------------------------------------


class TestMainBatch:
    def test_multiple_files_stream_the_batch(self, mocker, capsys):
        ada = _FixtureSchema(name="Ada", age=36)
        mock_stream = _patch_iter_extractions(mocker, events=[(0, ada), (1, ada)])

        exit_code = main(["a.pdf", "b.pdf", *_BASE_ARGS])

        assert exit_code == 0
        mock_stream.assert_called_once()
        options = mock_stream.call_args.args[3]
        assert options.max_concurrency == 5
        assert options.return_exceptions is False
        assert options.max_input_bytes == 50 * 1024 * 1024
        assert options.rich is False
        assert options.style is ExtractionStyle.DIRECT
        payload = json.loads(capsys.readouterr().out)
        assert payload == [{"name": "Ada", "age": 36}, {"name": "Ada", "age": 36}]

    def test_batch_array_output_is_input_ordered(self, mocker, capsys):
        ada = _FixtureSchema(name="Ada", age=36)
        linus = _FixtureSchema(name="Linus", age=54)
        _patch_iter_extractions(mocker, events=[(1, linus), (0, ada)])

        exit_code = main(["a.pdf", "b.pdf", *_BASE_ARGS])

        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out)
        assert [entry["name"] for entry in payload] == ["Ada", "Linus"]

    def test_max_concurrency_is_forwarded(self, mocker, capsys):
        ada = _FixtureSchema(name="Ada", age=36)
        mock_stream = _patch_iter_extractions(mocker, events=[(0, ada), (1, ada)])

        main(["a.pdf", "b.pdf", *_BASE_ARGS, "--max-concurrency", "2"])

        assert mock_stream.call_args.args[3].max_concurrency == 2
        capsys.readouterr()

    def test_continue_on_error_passes_return_exceptions(self, mocker, capsys):
        ada = _FixtureSchema(name="Ada", age=36)
        mock_stream = _patch_iter_extractions(mocker, events=[(0, ada), (1, ada)])

        main(["a.pdf", "b.pdf", *_BASE_ARGS, "--continue-on-error"])

        assert mock_stream.call_args.args[3].return_exceptions is True
        capsys.readouterr()

    def test_continue_on_error_reports_failures_and_exits_7(self, mocker, capsys):
        ada = _FixtureSchema(name="Ada", age=36)
        _patch_iter_extractions(mocker, events=[(0, ada), (1, ModelError("boom"))])

        exit_code = main(["a.pdf", "b.pdf", *_BASE_ARGS, "--continue-on-error"])

        assert exit_code == 7
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload[0] == {"name": "Ada", "age": 36}
        assert payload[1] == {"input": "b.pdf", "error": "boom", "error_type": "ModelError"}
        assert "1 of 2 input(s) failed" in captured.err

    def test_continue_on_error_all_success_exits_0(self, mocker, capsys):
        ada = _FixtureSchema(name="Ada", age=36)
        _patch_iter_extractions(mocker, events=[(0, ada), (1, ada)])

        exit_code = main(["a.pdf", "b.pdf", *_BASE_ARGS, "--continue-on-error"])

        assert exit_code == 0
        assert capsys.readouterr().err == ""

    def test_fail_fast_batch_maps_exit_code_and_keeps_stdout_empty(self, mocker, capsys):
        ada = _FixtureSchema(name="Ada", age=36)
        _patch_iter_extractions(mocker, events=[(0, ada)], error=ModelError("upstream"))

        exit_code = main(["a.pdf", "b.pdf", *_BASE_ARGS])

        assert exit_code == 4
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "upstream" in captured.err

    def test_batch_repr_output(self, mocker, capsys):
        ada = _FixtureSchema(name="Ada", age=36)
        _patch_iter_extractions(mocker, events=[(0, ada), (1, ada)])

        exit_code = main(["a.pdf", "b.pdf", *_BASE_ARGS, "--output", "repr"])

        assert exit_code == 0
        assert capsys.readouterr().out.startswith("[{'name': 'Ada'")

    def test_batch_invalid_retry_values_return_1_before_any_model_call(self, mocker, capsys):
        mock_stream = _patch_iter_extractions(mocker)

        exit_code = main(["a.pdf", "b.pdf", *_BASE_ARGS, "--max-retries", "-1"])

        assert exit_code == 1
        assert "max_retries" in capsys.readouterr().err
        mock_stream.assert_not_called()

    def test_batch_invalid_max_concurrency_returns_1_before_any_model_call(self, mocker, capsys):
        mock_stream = _patch_iter_extractions(mocker)

        exit_code = main(["a.pdf", "b.pdf", *_BASE_ARGS, "--max-concurrency", "-2"])

        assert exit_code == 1
        assert "max_concurrency" in capsys.readouterr().err
        mock_stream.assert_not_called()

    def test_batch_invalid_max_input_bytes_returns_1(self, mocker, capsys):
        mock_stream = _patch_iter_extractions(mocker)

        exit_code = main(["a.pdf", "b.pdf", *_BASE_ARGS, "--max-input-bytes", "0"])

        assert exit_code == 1
        assert "max_input_bytes" in capsys.readouterr().err
        mock_stream.assert_not_called()


# ---------------------------------------------------------------------------
# usage output
# ---------------------------------------------------------------------------


class TestUsage:
    def test_usage_flag_calls_extract_with_usage(self, mocker, capsys):
        fake = MagicMock()
        fake.model_dump.return_value = {"name": "Ada", "age": 36}
        usage = Usage(input_tokens=10, output_tokens=5, total_tokens=15)
        mock_usage = _patch_extract_with_usage(mocker, return_value=(fake, usage))

        exit_code = main(["input.txt", *_BASE_ARGS, "--usage"])

        assert exit_code == 0
        mock_usage.assert_called_once()
        captured = capsys.readouterr()
        assert '"usage"' in captured.out
        assert captured.out.count("input_tokens") >= 1

    def test_batch_usage_selects_rich_results(self, mocker, capsys):
        ada = _rich_result(_FixtureSchema(name="Ada", age=36))
        mock_stream = _patch_iter_extractions(mocker, events=[(0, ada), (1, ada)])

        exit_code = main(["a.pdf", "b.pdf", *_BASE_ARGS, "--usage"])

        assert exit_code == 0
        assert mock_stream.call_args.args[3].rich is True
        capsys.readouterr()

    def test_batch_usage_reports_per_item_and_aggregate(self, mocker, capsys):
        ada = _rich_result(_FixtureSchema(name="Ada", age=36), input_tokens=10, output_tokens=5)
        linus = _rich_result(_FixtureSchema(name="Linus", age=54), input_tokens=2, output_tokens=1)
        _patch_iter_extractions(mocker, events=[(0, ada), (1, linus)])

        exit_code = main(["a.pdf", "b.pdf", *_BASE_ARGS, "--usage"])

        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["results"][0] == {
            "input": "a.pdf",
            "result": {"name": "Ada", "age": 36},
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
        assert payload["usage"] == {"input_tokens": 12, "output_tokens": 6, "total_tokens": 18}

    def test_batch_usage_aggregates_successes_only_on_partial_failure(self, mocker, capsys):
        ada = _rich_result(_FixtureSchema(name="Ada", age=36), input_tokens=10, output_tokens=5)
        _patch_iter_extractions(mocker, events=[(0, ada), (1, ModelError("boom"))])

        exit_code = main(["a.pdf", "b.pdf", *_BASE_ARGS, "--usage", "--continue-on-error"])

        assert exit_code == 7
        payload = json.loads(capsys.readouterr().out)
        assert payload["results"][1] == {
            "input": "b.pdf",
            "error": "boom",
            "error_type": "ModelError",
        }
        assert payload["usage"] == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


# ---------------------------------------------------------------------------
# JSONL output
# ---------------------------------------------------------------------------


class TestJsonl:
    def test_jsonl_streams_records_in_completion_order(self, mocker, capsys):
        ada = _FixtureSchema(name="Ada", age=36)
        linus = _FixtureSchema(name="Linus", age=54)
        _patch_iter_extractions(mocker, events=[(1, linus), (0, ada)])

        exit_code = main(["a.pdf", "b.pdf", *_BASE_ARGS, "--output", "jsonl"])

        assert exit_code == 0
        lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert lines == [
            {"index": 1, "input": "b.pdf", "result": {"name": "Linus", "age": 54}},
            {"index": 0, "input": "a.pdf", "result": {"name": "Ada", "age": 36}},
        ]

    def test_jsonl_single_input_uses_batch_semantics(self, mocker, capsys):
        ada = _FixtureSchema(name="Ada", age=36)
        _patch_iter_extractions(mocker, events=[(0, ada)])
        mock_extract = _patch_extract(mocker)

        exit_code = main(["a.pdf", *_BASE_ARGS, "--output", "jsonl"])

        assert exit_code == 0
        mock_extract.assert_not_called()
        lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert lines == [{"index": 0, "input": "a.pdf", "result": {"name": "Ada", "age": 36}}]

    def test_jsonl_failure_records_preserve_input_identity(self, mocker, capsys):
        ada = _FixtureSchema(name="Ada", age=36)
        _patch_iter_extractions(mocker, events=[(0, ada), (1, UrlFetchError("404"))])

        exit_code = main(
            ["a.pdf", "b.pdf", *_BASE_ARGS, "--output", "jsonl", "--continue-on-error"]
        )

        assert exit_code == 7
        captured = capsys.readouterr()
        lines = [json.loads(line) for line in captured.out.splitlines()]
        assert lines[1] == {
            "index": 1,
            "input": "b.pdf",
            "error": "404",
            "error_type": "UrlFetchError",
        }
        assert "1 of 2 input(s) failed" in captured.err

    def test_jsonl_usage_emits_per_item_usage_and_summary(self, mocker, capsys):
        ada = _rich_result(_FixtureSchema(name="Ada", age=36), input_tokens=10, output_tokens=5)
        _patch_iter_extractions(mocker, events=[(0, ada)])

        exit_code = main(["a.pdf", *_BASE_ARGS, "--output", "jsonl", "--usage"])

        assert exit_code == 0
        lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert lines[0]["usage"] == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        assert lines[1] == {
            "summary": {
                "inputs": 1,
                "failed": 0,
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            }
        }

    def test_jsonl_without_usage_has_no_summary_line(self, mocker, capsys):
        ada = _FixtureSchema(name="Ada", age=36)
        _patch_iter_extractions(mocker, events=[(0, ada)])

        exit_code = main(["a.pdf", *_BASE_ARGS, "--output", "jsonl"])

        assert exit_code == 0
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 1
        assert "summary" not in lines[0]

    def test_jsonl_fail_fast_keeps_already_emitted_records(self, mocker, capsys):
        ada = _FixtureSchema(name="Ada", age=36)
        _patch_iter_extractions(mocker, events=[(0, ada)], error=UrlFetchError("404"))

        exit_code = main(["a.pdf", "b.pdf", *_BASE_ARGS, "--output", "jsonl"])

        assert exit_code == 2
        captured = capsys.readouterr()
        assert json.loads(captured.out)["input"] == "a.pdf"
        assert "404" in captured.err


# ---------------------------------------------------------------------------
# progress reporting
# ---------------------------------------------------------------------------


class TestProgress:
    def test_progress_reports_each_completion_on_stderr(self, mocker, capsys):
        ada = _FixtureSchema(name="Ada", age=36)
        _patch_iter_extractions(mocker, events=[(0, ada), (1, ada)])

        exit_code = main(["a.pdf", "b.pdf", *_BASE_ARGS, "--progress"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "progress: 1/2 completed (0 failed): a.pdf" in captured.err
        assert "progress: 2/2 completed (0 failed): b.pdf" in captured.err
        assert "progress" not in captured.out

    def test_progress_counts_failures(self, mocker, capsys):
        ada = _FixtureSchema(name="Ada", age=36)
        _patch_iter_extractions(mocker, events=[(0, ModelError("boom")), (1, ada)])

        exit_code = main(["a.pdf", "b.pdf", *_BASE_ARGS, "--progress", "--continue-on-error"])

        assert exit_code == 7
        assert "progress: 1/2 completed (1 failed): a.pdf" in capsys.readouterr().err

    def test_no_progress_lines_by_default(self, mocker, capsys):
        ada = _FixtureSchema(name="Ada", age=36)
        _patch_iter_extractions(mocker, events=[(0, ada), (1, ada)])

        main(["a.pdf", "b.pdf", *_BASE_ARGS])

        assert "progress" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# manifest input
# ---------------------------------------------------------------------------


class TestManifest:
    def _write_manifest(self, tmp_path, lines):
        path = tmp_path / "manifest.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(path)

    def test_manifest_builds_extraction_inputs_and_labels(self, mocker, capsys, tmp_path):
        manifest = self._write_manifest(
            tmp_path,
            [
                '{"source": "a.pdf", "media_type": "application/pdf", "name": "invoice-a"}',
                "",
                '{"source": "b.txt"}',
            ],
        )
        ada = _FixtureSchema(name="Ada", age=36)
        mock_stream = _patch_iter_extractions(mocker, events=[(0, ada), (1, ada)])

        exit_code = main([*_BASE_ARGS, "--manifest", manifest, "--output", "jsonl"])

        assert exit_code == 0
        items = mock_stream.call_args.args[2]
        assert items == [
            ExtractionInput(source="a.pdf", media_type="application/pdf", name="invoice-a"),
            ExtractionInput(source="b.txt", media_type=None, name=None),
        ]
        lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert [line["input"] for line in lines] == ["invoice-a", "b.txt"]

    def test_manifest_single_entry_still_uses_batch_output(self, mocker, capsys, tmp_path):
        manifest = self._write_manifest(tmp_path, ['{"source": "a.pdf"}'])
        ada = _FixtureSchema(name="Ada", age=36)
        _patch_iter_extractions(mocker, events=[(0, ada)])

        exit_code = main([*_BASE_ARGS, "--manifest", manifest])

        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload == [{"name": "Ada", "age": 36}]

    def test_manifest_with_positional_inputs_returns_1(self, capsys, tmp_path):
        manifest = self._write_manifest(tmp_path, ['{"source": "a.pdf"}'])

        exit_code = main(["other.pdf", *_BASE_ARGS, "--manifest", manifest])

        assert exit_code == 1
        assert "cannot be combined" in capsys.readouterr().err

    def test_manifest_missing_file_returns_1(self, capsys, tmp_path):
        exit_code = main([*_BASE_ARGS, "--manifest", str(tmp_path / "missing.jsonl")])

        assert exit_code == 1
        assert "cannot read manifest" in capsys.readouterr().err

    def test_manifest_without_entries_returns_1(self, capsys, tmp_path):
        manifest = self._write_manifest(tmp_path, ["", "   "])

        exit_code = main([*_BASE_ARGS, "--manifest", manifest])

        assert exit_code == 1
        assert "contains no entries" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("line", "message"),
        [
            ("not json", "invalid JSON"),
            ('["a.pdf"]', "expected a JSON object"),
            ('{"source": "a.pdf", "mediatype": "text/plain"}', "unknown keys: mediatype"),
            ('{"media_type": "text/plain"}', "'source' must be a non-empty string"),
            ('{"source": ""}', "'source' must be a non-empty string"),
            ('{"source": "-"}', "stdin (-) is not supported"),
            ('{"source": "a.pdf", "media_type": 7}', "'media_type' must be a string"),
            ('{"source": "a.pdf", "name": 7}', "'name' must be a string"),
        ],
    )
    def test_manifest_invalid_entries_return_1(self, capsys, tmp_path, line, message):
        manifest = self._write_manifest(tmp_path, [line])

        exit_code = main([*_BASE_ARGS, "--manifest", manifest])

        assert exit_code == 1
        error = capsys.readouterr().err
        assert "manifest line 1" in error
        assert message in error


# ---------------------------------------------------------------------------
# stdin
# ---------------------------------------------------------------------------


class TestStdin:
    def test_stdin_without_media_type_returns_1(self, capsys, mocker):
        mocker.patch("sys.stdin")
        exit_code = main(["-", *_BASE_ARGS])
        assert exit_code == 1
        assert "media-type" in capsys.readouterr().err.lower()

    def test_stdin_with_other_paths_returns_1(self, capsys):
        exit_code = main(["-", "other.pdf", *_BASE_ARGS, "--media-type", "application/pdf"])
        assert exit_code == 1
        assert "stdin" in capsys.readouterr().err.lower()

    def test_stdin_passes_buffer_for_bounded_reading(self, mocker, capsys):
        fake = MagicMock()
        fake.model_dump_json.return_value = "{}"
        mock_extract = _patch_extract(mocker, return_value=fake)
        stdin = mocker.patch("openextract._cli.sys.stdin")

        exit_code = main(["-", *_BASE_ARGS, "--media-type", "application/pdf"])

        assert exit_code == 0
        assert mock_extract.call_args.kwargs["input_file"] is stdin.buffer
        assert mock_extract.call_args.kwargs["media_type"] == "application/pdf"
        stdin.buffer.read.assert_not_called()


# ---------------------------------------------------------------------------
# cancellation and broken pipes
# ---------------------------------------------------------------------------


class TestInterruptAndBrokenPipe:
    def test_single_keyboard_interrupt_returns_130(self, mocker, capsys):
        _patch_extract(mocker, side_effect=KeyboardInterrupt)

        exit_code = main(["input.txt", *_BASE_ARGS])

        assert exit_code == 130
        assert "interrupted" in capsys.readouterr().err

    def test_batch_keyboard_interrupt_returns_130(self, mocker, capsys):
        _patch_iter_extractions(mocker, error=KeyboardInterrupt())

        exit_code = main(["a.pdf", "b.pdf", *_BASE_ARGS])

        assert exit_code == 130
        assert "interrupted" in capsys.readouterr().err

    def test_single_broken_pipe_returns_141(self, mocker, capsys):
        _patch_extract(mocker, return_value=_FixtureSchema(name="Ada", age=36))
        mocker.patch("openextract._cli._print_json", side_effect=BrokenPipeError)
        discard = mocker.patch("openextract._cli._discard_stdout")

        exit_code = main(["input.txt", *_BASE_ARGS, "--output", "repr"])

        assert exit_code == 141
        discard.assert_called_once_with()
        capsys.readouterr()

    def test_jsonl_broken_pipe_returns_141(self, mocker, capsys):
        ada = _FixtureSchema(name="Ada", age=36)
        _patch_iter_extractions(mocker, events=[(0, ada), (1, ada)])
        mocker.patch("openextract._cli._emit_json_line", side_effect=BrokenPipeError)
        discard = mocker.patch("openextract._cli._discard_stdout")

        exit_code = main(["a.pdf", "b.pdf", *_BASE_ARGS, "--output", "jsonl"])

        assert exit_code == 141
        discard.assert_called_once_with()
        capsys.readouterr()

    def test_discard_stdout_redirects_to_devnull(self, monkeypatch, tmp_path):
        target = tmp_path / "captured.txt"
        with open(target, "w", encoding="utf-8") as handle:
            monkeypatch.setattr(sys, "stdout", handle)
            _discard_stdout()
            print("lost after redirect")
        assert target.read_text(encoding="utf-8") == ""

    def test_discard_stdout_tolerates_stdout_without_fileno(self, monkeypatch):
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        _discard_stdout()  # must not raise


# ---------------------------------------------------------------------------
# Swarm and agent flags
# ---------------------------------------------------------------------------

_AGENT_SOURCE = """
from openextract import define_agent
from tests.test_cli import _FixtureSchema

agent = define_agent({description!r}, model={model!r}, output_schema=_FixtureSchema)
"""

_SCHEMALESS_AGENT_SOURCE = """
from openextract import define_agent

agent = define_agent({description!r}, model={model!r})
"""


def _write_agent(path, description="Fixture", model="test:a", source=_AGENT_SOURCE):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source.format(description=description, model=model), encoding="utf-8")
    return str(path)


def _patch_swarm(mocker, output=None, usage_result=None):
    plain = mocker.patch("openextract._cli.extract_swarm")
    plain.return_value = output if output is not None else _FixtureSchema(name="Ada", age=36)
    rich = mocker.patch("openextract._cli.extract_swarm_with_results")
    rich.return_value = usage_result
    return plain, rich


class _SwarmResultStub:
    def __init__(self):
        self.output = _FixtureSchema(name="Ada", age=36)
        self.usage = MagicMock(input_tokens=1, output_tokens=2, total_tokens=3)
        self.agents = (1, 2)
        self.reduce = MagicMock(value="vote")


class TestSwarmFlags:
    def test_swarm_size_fans_out_one_model(self, mocker, capsys):
        plain, _ = _patch_swarm(mocker)

        assert main(["input.txt", *_BASE_ARGS, "--model", "test:a", "--swarm", "3"]) == 0

        assert plain.call_args.args[1] == ["test:a"]
        assert plain.call_args.kwargs["size"] == 3
        assert "Ada" in capsys.readouterr().out

    def test_models_list_becomes_the_agent_list(self, mocker, capsys):
        plain, _ = _patch_swarm(mocker)

        argv = [
            "input.txt",
            "--schema",
            "tests.test_cli:_FixtureSchema",
            "--models",
            "test:a, test:b",
            "--reduce",
            "vote",
        ]
        assert main(argv) == 0

        assert plain.call_args.args[1] == ["test:a", "test:b"]
        assert plain.call_args.kwargs["size"] is None
        assert plain.call_args.kwargs["reduce"] == "vote"
        capsys.readouterr()

    def test_usage_reports_agent_count_and_reduce(self, mocker, capsys):
        _patch_swarm(mocker, usage_result=_SwarmResultStub())

        argv = [
            "input.txt",
            "--schema",
            "tests.test_cli:_FixtureSchema",
            "--models",
            "test:a,test:b",
            "--usage",
        ]
        assert main(argv) == 0

        payload = capsys.readouterr().out
        assert '"agents": 2' in payload
        assert '"reduce": "vote"' in payload
        assert '"total_tokens": 3' in payload

    def test_a_single_model_stays_a_one_shot_call(self, mocker):
        swarm = mocker.patch("openextract._cli.extract_swarm")
        _patch_extract(mocker, return_value=_FixtureSchema(name="Ada", age=36))

        assert main(["input.txt", *_BASE_ARGS]) == 0

        swarm.assert_not_called()

    @pytest.mark.parametrize(
        ("extra", "message"),
        [
            (["--model", "test:a", "--swarm", "0"], "--swarm must be a positive integer"),
            (["--models", "test:a,test:b", "--swarm", "3"], "does not match"),
            (["--models", "test:a,test:b", "--model", "test:c"], "not both"),
        ],
    )
    def test_invalid_swarm_combinations_exit_one(self, capsys, extra, message):
        argv = ["input.txt", "--schema", "tests.test_cli:_FixtureSchema", *extra]
        assert main(argv) == 1
        assert message in capsys.readouterr().err

    def test_swarm_flags_require_a_single_input(self, capsys):
        argv = ["a.txt", "b.txt", *_BASE_ARGS, "--model", "test:a", "--swarm", "2"]
        assert main(argv) == 1
        assert "single input" in capsys.readouterr().err

    def test_swarm_flags_reject_jsonl_output(self, capsys):
        argv = ["input.txt", *_BASE_ARGS, "--swarm", "2", "--output", "jsonl"]
        assert main(argv) == 1
        assert "single result" in capsys.readouterr().err

    def test_swarm_flags_reject_manifests(self, capsys, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text('{"source": "a.pdf"}\n', encoding="utf-8")
        argv = [*_BASE_ARGS, "--manifest", str(manifest), "--swarm", "2"]
        assert main(argv) == 1
        assert "single input" in capsys.readouterr().err


class TestAgentFlags:
    def test_an_agent_supplies_the_model_and_schema(self, mocker, tmp_path):
        path = _write_agent(tmp_path / "invoices.py")
        run = _patch_extract(mocker, return_value=_FixtureSchema(name="Ada", age=36))

        assert main(["input.txt", "--agent", path]) == 0

        assert run.call_args.kwargs["schema"] is _FixtureSchema

    def test_an_explicit_schema_still_wins(self, mocker, tmp_path):
        path = _write_agent(tmp_path / "invoices.py", source=_SCHEMALESS_AGENT_SOURCE)
        run = _patch_extract(mocker, return_value=_FixtureSchema(name="Ada", age=36))

        argv = ["input.txt", "--agent", path, "--schema", "tests.test_cli:_FixtureSchema"]
        assert main(argv) == 0

        assert run.call_args.kwargs["schema"] is _FixtureSchema

    def test_an_agent_without_a_schema_needs_one(self, capsys, tmp_path):
        path = _write_agent(tmp_path / "invoices.py", source=_SCHEMALESS_AGENT_SOURCE)
        assert main(["input.txt", "--agent", path]) == 1
        assert "--schema is required" in capsys.readouterr().err

    def test_several_agents_form_a_swarm(self, mocker, tmp_path):
        first = _write_agent(tmp_path / "a.py", description="First")
        second = _write_agent(tmp_path / "b.py", description="Second")
        plain, _ = _patch_swarm(mocker)

        assert main(["input.txt", "--agents", f"{first},{second}"]) == 0

        assert [agent.description for agent in plain.call_args.args[1]] == ["First", "Second"]

    def test_agent_and_agents_are_mutually_exclusive(self, capsys, tmp_path):
        path = _write_agent(tmp_path / "a.py")
        assert main(["input.txt", "--agent", path, "--agents", path]) == 1
        assert "not both" in capsys.readouterr().err

    def test_an_agent_may_be_fanned_out_with_swarm(self, mocker, tmp_path):
        path = _write_agent(tmp_path / "a.py")
        plain, _ = _patch_swarm(mocker)

        assert main(["input.txt", "--agent", path, "--swarm", "2"]) == 0

        assert plain.call_args.kwargs["size"] == 2

    def test_agents_may_not_be_combined_with_a_batch(self, capsys, tmp_path):
        path = _write_agent(tmp_path / "a.py")
        assert main(["a.txt", "b.txt", "--agent", path]) == 1
        assert "single input" in capsys.readouterr().err

    def test_a_missing_agent_path_exits_one(self, capsys):
        assert main(["input.txt", "--agent", "no-such-agent"]) == 1
        assert "Expected a directory" in capsys.readouterr().err


class TestRemoteAgentExit:
    def test_remote_agent_failures_exit_eight(self, mocker, capsys):
        from openextract import RemoteAgentError

        _patch_extract(mocker, side_effect=RemoteAgentError("agent unreachable"))

        assert main(["input.txt", *_BASE_ARGS, "--model", "test:a"]) == 8
        assert "agent unreachable" in capsys.readouterr().err


_CITE_NAME = Citation(field="name", quote="Ada", page=1)
_CITE_AGE = Citation(field="age", quote="36", page=1, bbox=(0.1, 0.2, 0.3, 0.05))


class TestCite:
    def test_citations_payload_omits_missing_bbox(self):
        assert _citations_payload((_CITE_NAME, _CITE_AGE)) == [
            {"field": "name", "quote": "Ada", "page": 1},
            {"field": "age", "quote": "36", "page": 1, "bbox": [0.1, 0.2, 0.3, 0.05]},
        ]
        assert _result_citations(_FixtureSchema(name="Ada", age=36)) == []

    def test_cite_flag_is_forwarded_to_results_api(self, mocker, capsys):
        mock_fn = _patch_extract_many_with_results(
            mocker,
            return_value=[
                _rich_result(_FixtureSchema(name="Ada", age=36), citations=(_CITE_NAME,))
            ],
        )

        assert main(["input.txt", *_BASE_ARGS, "--cite"]) == 0

        assert mock_fn.call_args.kwargs["cite"] is True
        payload = json.loads(capsys.readouterr().out)
        assert payload == {
            "result": {"name": "Ada", "age": 36},
            "citations": [{"field": "name", "quote": "Ada", "page": 1}],
        }

    def test_cite_json_includes_optional_bbox(self, mocker, capsys):
        _patch_extract_many_with_results(
            mocker,
            return_value=[
                _rich_result(_FixtureSchema(name="Ada", age=36), citations=(_CITE_NAME, _CITE_AGE))
            ],
        )

        assert main(["input.txt", *_BASE_ARGS, "--cite"]) == 0

        payload = json.loads(capsys.readouterr().out)
        assert payload["citations"][1]["bbox"] == [0.1, 0.2, 0.3, 0.05]
        assert "bbox" not in payload["citations"][0]

    def test_cite_with_usage_keeps_usage_shape(self, mocker, capsys):
        _patch_extract_many_with_results(
            mocker,
            return_value=[
                _rich_result(
                    _FixtureSchema(name="Ada", age=36),
                    input_tokens=10,
                    output_tokens=5,
                    citations=(_CITE_NAME,),
                )
            ],
        )

        assert main(["input.txt", *_BASE_ARGS, "--cite", "--usage"]) == 0

        payload = json.loads(capsys.readouterr().out)
        assert payload["usage"] == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        assert payload["citations"] == [{"field": "name", "quote": "Ada", "page": 1}]

    def test_cite_json_payload_with_test_model(self, mocker, tmp_path, capsys):
        from pydantic_ai.models.test import TestModel

        from openextract._batch import extract_many_with_results

        source = tmp_path / "doc.txt"
        source.write_text("Ada Lovelace, 36", encoding="utf-8")

        def _run(schema, model, input_files, **kwargs):
            return extract_many_with_results(
                schema,
                TestModel(
                    custom_output_args={
                        "output": {"name": "Ada", "age": 36},
                        "citations": [{"field": "name", "quote": "Ada Lovelace", "page": 1}],
                    }
                ),
                input_files,
                **kwargs,
            )

        mocker.patch("openextract._cli.extract_many_with_results", side_effect=_run)

        assert main([str(source), *_BASE_ARGS, "--cite"]) == 0

        payload = json.loads(capsys.readouterr().out)
        assert payload["result"] == {"name": "Ada", "age": 36}
        assert payload["citations"] == [
            {"field": "name", "quote": "Ada Lovelace", "page": 1},
        ]

    def test_cite_without_flag_does_not_change_output(self, mocker, capsys):
        _patch_extract(mocker, return_value=_FixtureSchema(name="Ada", age=36))

        assert main(["input.txt", *_BASE_ARGS]) == 0

        assert json.loads(capsys.readouterr().out) == {"name": "Ada", "age": 36}

    def test_injected_agent_error_exits_one(self, mocker, capsys):
        _patch_extract_many_with_results(
            mocker,
            side_effect=ValueError("cite cannot be used with an injected agent."),
        )

        assert main(["input.txt", *_BASE_ARGS, "--cite"]) == 1
        assert "cite cannot be used with an injected agent" in capsys.readouterr().err

    def test_batch_cite_wraps_results_and_forwards_flag(self, mocker, capsys):
        ada = _rich_result(_FixtureSchema(name="Ada", age=36), citations=(_CITE_NAME,))
        mock_stream = _patch_iter_extractions(mocker, events=[(0, ada), (1, ada)])

        assert main(["a.pdf", "b.pdf", *_BASE_ARGS, "--cite"]) == 0

        options = mock_stream.call_args.args[3]
        assert options.cite is True
        assert options.rich is True
        payload = json.loads(capsys.readouterr().out)
        assert payload == [
            {
                "result": {"name": "Ada", "age": 36},
                "citations": [{"field": "name", "quote": "Ada", "page": 1}],
            },
            {
                "result": {"name": "Ada", "age": 36},
                "citations": [{"field": "name", "quote": "Ada", "page": 1}],
            },
        ]

    def test_jsonl_cite_adds_citations_to_records(self, mocker, capsys):
        ada = _rich_result(_FixtureSchema(name="Ada", age=36), citations=(_CITE_NAME,))
        _patch_iter_extractions(mocker, events=[(0, ada)])

        assert main(["a.pdf", *_BASE_ARGS, "--output", "jsonl", "--cite"]) == 0

        lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert lines == [
            {
                "index": 0,
                "input": "a.pdf",
                "result": {"name": "Ada", "age": 36},
                "citations": [{"field": "name", "quote": "Ada", "page": 1}],
            }
        ]

    def test_batch_cite_with_usage_keeps_usage_envelope(self, mocker, capsys):
        ada = _rich_result(
            _FixtureSchema(name="Ada", age=36),
            input_tokens=10,
            output_tokens=5,
            citations=(_CITE_NAME,),
        )
        _patch_iter_extractions(mocker, events=[(0, ada)])

        assert main(["a.pdf", "b.pdf", *_BASE_ARGS, "--cite", "--usage"]) == 0

        payload = json.loads(capsys.readouterr().out)
        assert payload["results"][0]["citations"] == [{"field": "name", "quote": "Ada", "page": 1}]
        assert payload["results"][0]["usage"]["total_tokens"] == 15

    def test_swarm_cite_merges_agent_citations(self, mocker, capsys):
        cited = _rich_result(_FixtureSchema(name="Ada", age=36), citations=(_CITE_NAME,))
        stub = _SwarmResultStub()
        stub.agents = (cited, ModelError("boom"))
        _patch_swarm(mocker, usage_result=stub)

        assert main(["input.txt", *_BASE_ARGS, "--swarm", "2", "--cite"]) == 0

        payload = json.loads(capsys.readouterr().out)
        assert payload["result"] == {"name": "Ada", "age": 36}
        assert payload["citations"] == [{"field": "name", "quote": "Ada", "page": 1}]
        assert "usage" not in payload

    def test_fanning_agent_with_cite_uses_swarm(self, mocker, tmp_path, capsys):
        first = _write_agent(tmp_path / "a.py", description="First")
        second = _write_agent(tmp_path / "b.py", description="Second")
        parent = tmp_path / "parent.py"
        parent.write_text(
            "from openextract import define_agent, load_agent\n"
            "from tests.test_cli import _FixtureSchema\n"
            f"agent = define_agent('Parent', output_schema=_FixtureSchema, "
            f"subagents=[load_agent({first!r}), load_agent({second!r})])\n",
            encoding="utf-8",
        )
        stub = _SwarmResultStub()
        stub.agents = (_rich_result(_FixtureSchema(name="Ada", age=36), citations=(_CITE_AGE,)),)
        _, rich = _patch_swarm(mocker, usage_result=stub)

        assert main(["input.txt", "--agent", str(parent), "--cite"]) == 0

        rich.assert_called_once()
        assert rich.call_args.kwargs["cite"] is True
        assert json.loads(capsys.readouterr().out)["citations"][0]["bbox"] == [
            0.1,
            0.2,
            0.3,
            0.05,
        ]

    def test_single_agent_with_cite_stays_oneshot(self, mocker, tmp_path, capsys):
        path = _write_agent(tmp_path / "invoices.py")
        mock_fn = _patch_extract_many_with_results(
            mocker,
            return_value=[_rich_result(_FixtureSchema(name="Ada", age=36))],
        )
        swarm = mocker.patch("openextract._cli.extract_swarm_with_results")

        assert main(["input.txt", "--agent", path, "--cite"]) == 0

        swarm.assert_not_called()
        assert mock_fn.call_args.kwargs["cite"] is True
        capsys.readouterr()
