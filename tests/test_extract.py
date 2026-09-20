"""Tests for openextract._extract."""

import asyncio
import io
import ipaddress
import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import BaseModel, ValidationError
from pydantic_ai import BinaryContent
from pydantic_ai.models.test import TestModel

import openextract._agent as agent_module
from openextract import (
    ExtractionError,
    ExtractionInput,
    ExtractionResult,
    InputFileError,
    InputTooLargeError,
    ModelError,
    ProviderNotInstalledError,
    SchemaValidationError,
    UrlFetchError,
    Usage,
    extract,
    extract_async,
    extract_many,
    extract_many_async,
    extract_many_with_results,
    extract_many_with_results_async,
    extract_with_result,
    extract_with_result_async,
    extract_with_usage,
    extract_with_usage_async,
    iter_extract_many_async,
    total_usage,
)
from openextract._agent import _install_hint, _model_identifier
from openextract._batch import _run_with_shared_agent, _run_with_shared_agent_result
from openextract._config import (
    _max_redirects,
    _resolve_max_input_bytes,
    _url_fetch_timeout,
)
from openextract._errors import (
    _is_output_retry_error,
    _is_transient_model_exception,
    _map_exception,
    _model_retry_after,
    _model_status_code,
    _parse_retry_after,
)
from openextract._media import (
    _get_media,
    _get_media_async,
    _get_media_type,
    _is_public_ip,
    _is_safe_host,
    _item_source_label,
    _read_from_path,
    _read_url_with_client_async,
    _safe_source_context,
)
from openextract._retry import _retry_delay
from openextract._types import _extraction_result, _resolve_item


def _build_response(
    *,
    content: bytes = b"",
    content_type: str = "application/octet-stream",
    is_redirect: bool = False,
    status_code: int = 200,
    location: str | None = None,
    content_length: int | str | None = None,
) -> MagicMock:
    """Build a MagicMock that behaves like an httpx.Response."""
    response = MagicMock()
    response.content = content
    headers: dict[str, str] = {}
    if content_type:
        headers["content-type"] = content_type
    if location is not None:
        headers["location"] = location
    if content_length is not None:
        headers["content-length"] = str(content_length)
    response.headers = headers
    response.is_redirect = is_redirect
    response.status_code = status_code
    response.raise_for_status.return_value = None
    response.iter_bytes.return_value = iter([content])

    async def aiter_bytes(*, chunk_size):
        yield content

    response.aiter_bytes = MagicMock(side_effect=aiter_bytes)
    response.aclose = AsyncMock()
    return response


def _mock_sync_http_client(mocker, *, response=None, side_effect=None):
    client_cls = mocker.patch("openextract._media.httpx.Client")
    client = client_cls.return_value.__enter__.return_value
    client.build_request.return_value = MagicMock()
    if side_effect is not None:
        client.get.side_effect = side_effect
        client.send.side_effect = side_effect
    else:
        client.get.return_value = response
        client.send.return_value = response
    return client_cls, client


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_star_import_exposes_only_existing_names():
    """`from openextract import *` must not reference names that aren't defined."""
    namespace: dict = {}
    exec("from openextract import *", namespace)
    exported = {name for name in namespace if not name.startswith("_")}
    assert exported == {
        "Extractor",
        "AsyncExtractor",
        "RetryPolicy",
        "ExtractionInput",
        "ExtractionResult",
        "ExtractProgress",
        "Citation",
        "ExtractionStyle",
        "DefinedAgent",
        "RemoteAgent",
        "SwarmMember",
        "SwarmReduce",
        "SwarmResult",
        "define_agent",
        "define_remote_agent",
        "extract",
        "extract_async",
        "extract_many",
        "extract_many_async",
        "iter_extract_many_async",
        "extract_many_with_results",
        "extract_many_with_results_async",
        "extract_with_usage",
        "extract_with_result",
        "extract_swarm",
        "extract_swarm_async",
        "extract_swarm_with_results",
        "extract_swarm_with_results_async",
        "extract_with_usage_async",
        "extract_with_result_async",
        "flatten_agent",
        "load_agent",
        "load_agent_directory",
        "load_agents",
        "normalize_reduce",
        "reduce_outputs",
        "resolve_output_schema",
        "resolve_swarm_members",
        "total_usage",
        "Usage",
        "ExtractionError",
        "InputFileError",
        "InputTooLargeError",
        "ModelError",
        "ProviderNotInstalledError",
        "RemoteAgentError",
        "SchemaValidationError",
        "UrlFetchError",
    }


# ---------------------------------------------------------------------------
# _get_media_type
# ---------------------------------------------------------------------------


class TestGetMediaType:
    def test_returns_png_mime_type(self):
        assert _get_media_type("image.png") == "image/png"

    def test_returns_jpeg_mime_type(self):
        assert _get_media_type("photo.jpg") == "image/jpeg"

    def test_returns_text_mime_type(self):
        assert _get_media_type("notes.txt") == "text/plain"

    def test_returns_octet_stream_for_unknown_extension(self):
        assert _get_media_type("mystery.xyz123") == "application/octet-stream"

    def test_returns_octet_stream_for_no_extension(self):
        assert _get_media_type("README") == "application/octet-stream"

    def test_handles_full_path(self):
        assert _get_media_type("/tmp/nested/dir/file.png") == "image/png"


# ---------------------------------------------------------------------------
# _get_media
# ---------------------------------------------------------------------------


class TestGetMedia:
    def test_reads_local_file_with_known_extension(self, tmp_path):
        local = tmp_path / "hello.txt"
        local.write_bytes(b"hello world")

        media_bytes, media_type = _get_media(str(local))

        assert media_bytes == b"hello world"
        assert media_type == "text/plain"

    def test_reads_local_file_with_unknown_extension(self, tmp_path):
        odd_file = tmp_path / "data.weirdext"
        odd_file.write_bytes(b"opaque-bytes")

        media_bytes, media_type = _get_media(str(odd_file))

        assert media_bytes == b"opaque-bytes"
        assert media_type == "application/octet-stream"

    def test_reads_project_pdf_fixture(self):
        media_bytes, media_type = _get_media("tests/test.pdf")

        assert media_bytes.startswith(b"%PDF")
        assert media_type == "application/pdf"

    def test_missing_local_file_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist.txt"
        with pytest.raises(InputFileError, match="does_not_exist.txt") as exc_info:
            _get_media(str(missing))
        assert str(tmp_path) not in str(exc_info.value)

    def test_directory_raises_input_file_error(self, tmp_path):
        with pytest.raises(InputFileError, match="is a directory|Is a directory") as exc_info:
            _get_media(str(tmp_path))
        assert tmp_path.name in str(exc_info.value)
        assert str(tmp_path.parent) not in str(exc_info.value)

    def test_permission_denied_raises_input_file_error(self, tmp_path, mocker):
        locked = tmp_path / "locked.txt"
        locked.write_bytes(b"secret")
        mocker.patch.object(Path, "open", side_effect=PermissionError(13, "Permission denied"))
        with pytest.raises(InputFileError, match="Permission denied") as exc_info:
            _get_media(str(locked))
        assert "locked.txt" in str(exc_info.value)
        assert str(tmp_path) not in str(exc_info.value)

    def test_other_open_oserror_raises_input_file_error(self, tmp_path, mocker):
        local = tmp_path / "data.bin"
        local.write_bytes(b"ok")
        mocker.patch.object(Path, "stat", side_effect=OSError(5, "Input/output error"))
        with pytest.raises(InputFileError, match="Input/output error") as exc_info:
            _get_media(str(local))
        assert "data.bin" in str(exc_info.value)

    def test_oserror_without_strerror_uses_type_name(self, tmp_path, mocker):
        local = tmp_path / "data.bin"
        local.write_bytes(b"ok")
        mocker.patch.object(Path, "stat", side_effect=OSError("boom"))
        with pytest.raises(InputFileError, match="OSError"):
            _get_media(str(local))

    def test_filelike_read_oserror_raises_input_file_error(self):
        class _Broken:
            def read(self, _size=-1):
                raise OSError(5, "Input/output error")

        with pytest.raises(InputFileError, match="file-like input") as exc_info:
            _get_media(_Broken(), media_type="text/plain")
        assert "Input/output error" in str(exc_info.value)

    def test_filelike_missing_media_type_stays_type_error(self):
        class _Broken:
            def read(self, _size=-1):
                raise OSError(5, "Input/output error")

        with pytest.raises(TypeError, match="media_type is required"):
            _get_media(_Broken())

    def test_fetches_https_url(self, mocker):
        fake_response = _build_response(content=b"<html>remote</html>")
        client_cls, client = _mock_sync_http_client(mocker, response=fake_response)

        media_bytes, media_type = _get_media("https://example.com/page.html")

        client_cls.assert_called_once_with(follow_redirects=False, timeout=30.0)
        client.build_request.assert_called_once_with("GET", "https://example.com/page.html")
        client.send.assert_called_once_with(client.build_request.return_value, stream=True)
        # follow_redirects is disabled at the httpx layer; redirects are followed
        # manually by the URL reader so the SSRF host check runs at every hop.
        assert media_bytes == b"<html>remote</html>"
        assert media_type == "text/html"

    def test_fetches_http_url(self, mocker):
        """http:// URLs are fetched, not treated as local paths."""
        fake_response = _build_response(content=b"plain")
        _mock_sync_http_client(mocker, response=fake_response)

        media_bytes, media_type = _get_media("http://example.com/page.html")

        assert media_bytes == b"plain"
        assert media_type == "text/html"

    def test_url_without_useful_extension_falls_back_to_response_header(self, mocker):
        fake_response = _build_response(
            content=b"raw-bytes",
            content_type="application/pdf; charset=binary",
        )
        _mock_sync_http_client(mocker, response=fake_response)

        media_bytes, media_type = _get_media("https://example.com/download?id=42")

        assert media_bytes == b"raw-bytes"
        assert media_type == "application/pdf"

    def test_url_with_no_extension_and_no_header_stays_octet_stream(self, mocker):
        fake_response = _build_response(content=b"raw-bytes", content_type="")
        _mock_sync_http_client(mocker, response=fake_response)

        _, media_type = _get_media("https://example.com/blob")

        assert media_type == "application/octet-stream"

    def test_known_url_extension_ignores_response_header(self, mocker):
        """URL extension wins when it's specific; protects against misconfigured servers."""
        fake_response = _build_response(content=b"%PDF", content_type="text/html")
        _mock_sync_http_client(mocker, response=fake_response)

        _, media_type = _get_media("https://example.com/doc.pdf")

        assert media_type == "application/pdf"

    def test_http_error_status_raises(self, mocker):
        fake_response = _build_response(content=b"<html>404</html>")
        fake_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found", request=MagicMock(), response=fake_response
        )
        _mock_sync_http_client(mocker, response=fake_response)

        with pytest.raises(httpx.HTTPStatusError):
            _get_media("https://example.com/missing.pdf")

    def test_bytes_at_limit_are_allowed(self):
        assert _get_media(
            b"12345",
            media_type="text/plain",
            max_input_bytes=5,
        ) == (b"12345", "text/plain")

    def test_oversized_bytes_are_rejected(self):
        with pytest.raises(InputTooLargeError, match=r"5 bytes.*at least 6 bytes"):
            _get_media(
                b"123456",
                media_type="text/plain",
                max_input_bytes=5,
            )

    def test_local_path_is_rejected_before_read(self, tmp_path, mocker):
        local = tmp_path / "large.bin"
        local.write_bytes(b"123456")
        open_mock = mocker.patch.object(Path, "open")

        with pytest.raises(InputTooLargeError, match="large.bin"):
            _get_media(str(local), max_input_bytes=5)

        open_mock.assert_not_called()

    def test_non_seekable_stream_is_read_in_bounded_chunks(self):
        class NonSeekable:
            def __init__(self, data: bytes):
                self.data = data
                self.offset = 0
                self.read_sizes: list[int] = []

            def read(self, size: int) -> bytes:
                self.read_sizes.append(size)
                chunk = self.data[self.offset : self.offset + size]
                self.offset += len(chunk)
                return chunk

        stream = NonSeekable(b"123456")

        with pytest.raises(InputTooLargeError, match=r"5 bytes.*at least 6 bytes"):
            _get_media(
                stream,  # type: ignore[arg-type]
                media_type="text/plain",
                max_input_bytes=5,
            )

        assert stream.read_sizes == [6]

    def test_url_content_length_fast_fails_before_iteration(self, mocker):
        response = _build_response(content=b"", content_length=6)
        _mock_sync_http_client(mocker, response=response)

        with pytest.raises(InputTooLargeError, match=r"5 bytes.*at least 6 bytes"):
            _get_media("https://example.com/file.bin", max_input_bytes=5)

        response.iter_bytes.assert_not_called()

    @pytest.mark.parametrize("content_length", [None, -1, 2, "invalid"])
    def test_url_stream_cap_catches_missing_or_incorrect_length(
        self,
        mocker,
        content_length,
    ):
        response = _build_response(content=b"123456", content_length=content_length)
        _mock_sync_http_client(mocker, response=response)

        with pytest.raises(InputTooLargeError, match=r"5 bytes.*at least 6 bytes"):
            _get_media("https://example.com/file.bin", max_input_bytes=5)

    def test_url_error_context_does_not_leak_credentials_or_query(self, mocker):
        response = _build_response(content=b"123456")
        _mock_sync_http_client(mocker, response=response)

        with pytest.raises(InputTooLargeError) as exc_info:
            _get_media(
                "https://user:secret@example.com/file.bin?token=sensitive#fragment",
                max_input_bytes=5,
            )

        message = str(exc_info.value)
        assert "example.com/file.bin" in message
        assert "secret" not in message
        assert "sensitive" not in message

    def test_streaming_fetch_follows_redirect_and_closes_each_response(self, mocker):
        redirect = _build_response(
            is_redirect=True,
            status_code=302,
            location="https://2.2.2.2/final.bin",
        )
        final = _build_response(content=b"ok")
        _, client = _mock_sync_http_client(mocker, side_effect=[redirect, final])

        assert _get_media("https://1.1.1.1/start", max_input_bytes=5)[0] == b"ok"

        assert client.send.call_count == 2
        redirect.close.assert_called_once_with()
        final.close.assert_called_once_with()

    def test_streaming_redirect_without_location_is_rejected(self, mocker):
        response = _build_response(is_redirect=True, status_code=302)
        _mock_sync_http_client(mocker, response=response)

        with pytest.raises(UrlFetchError, match="missing Location"):
            _get_media("https://1.1.1.1/start", max_input_bytes=5)

    def test_streaming_redirect_limit_is_enforced(self, mocker, monkeypatch):
        monkeypatch.setenv("OPENEXTRACT_MAX_REDIRECTS", "2")
        redirect = _build_response(
            is_redirect=True,
            status_code=302,
            location="https://1.1.1.1/loop",
        )
        _mock_sync_http_client(mocker, response=redirect)

        with pytest.raises(UrlFetchError, match="Too many redirects"):
            _get_media("https://1.1.1.1/loop", max_input_bytes=5)

        assert redirect.close.call_count == 2


class TestMaxInputBytesConfiguration:
    def test_default_is_50_mib(self, monkeypatch):
        monkeypatch.delenv("OPENEXTRACT_MAX_INPUT_BYTES", raising=False)
        assert _resolve_max_input_bytes(None) == 52_428_800

    def test_environment_override(self, monkeypatch):
        monkeypatch.setenv("OPENEXTRACT_MAX_INPUT_BYTES", "123")
        assert _resolve_max_input_bytes(None) == 123

    def test_kwarg_wins_over_environment(self, monkeypatch):
        monkeypatch.setenv("OPENEXTRACT_MAX_INPUT_BYTES", "123")
        assert _resolve_max_input_bytes(456) == 456

    @pytest.mark.parametrize("value", [0, -1, True, 1.5, "100"])
    def test_invalid_explicit_value_is_rejected(self, value):
        with pytest.raises(ValueError, match="max_input_bytes"):
            _resolve_max_input_bytes(value)

    @pytest.mark.parametrize("value", ["0", "-1", "abc", "1.5"])
    def test_invalid_environment_value_is_rejected(self, monkeypatch, value):
        monkeypatch.setenv("OPENEXTRACT_MAX_INPUT_BYTES", value)
        with pytest.raises(ValueError, match="OPENEXTRACT_MAX_INPUT_BYTES"):
            _resolve_max_input_bytes(None)

    def test_safe_url_context_tolerates_invalid_port(self):
        context = _safe_source_context("https://example.com:not-a-port/file?secret=yes")

        assert context == "URL https://example.com/file"
        assert "secret" not in context


# ---------------------------------------------------------------------------
# SSRF host validation
# ---------------------------------------------------------------------------


def _addrinfo(ip: str):
    """Build a getaddrinfo-shaped tuple for ``ip``."""
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    return [(family, socket.SOCK_STREAM, 0, "", (ip, 0))]


class TestIsPublicIp:
    def test_public_ipv4_is_public(self):
        assert _is_public_ip(ipaddress.ip_address("1.1.1.1")) is True

    def test_private_ipv4_is_not_public(self):
        assert _is_public_ip(ipaddress.ip_address("10.0.0.1")) is False

    def test_loopback_ipv4_is_not_public(self):
        assert _is_public_ip(ipaddress.ip_address("127.0.0.1")) is False

    def test_aws_metadata_link_local_is_not_public(self):
        assert _is_public_ip(ipaddress.ip_address("169.254.169.254")) is False

    def test_ipv6_loopback_is_not_public(self):
        assert _is_public_ip(ipaddress.ip_address("::1")) is False

    def test_ipv4_mapped_ipv6_loopback_is_not_public(self):
        # ::ffff:127.0.0.1 wraps an IPv4 loopback; must be unwrapped and rejected.
        assert _is_public_ip(ipaddress.ip_address("::ffff:127.0.0.1")) is False


class TestIsSafeHost:
    def test_opt_out_env_var_bypasses_validation(self, monkeypatch):
        monkeypatch.setenv("OPENEXTRACT_ALLOW_PRIVATE_URLS", "1")
        # Even a clearly private address is permitted when the opt-out is set.
        assert _is_safe_host("127.0.0.1") is True

    def test_opt_out_env_var_unset_does_not_bypass(self, monkeypatch):
        monkeypatch.delenv("OPENEXTRACT_ALLOW_PRIVATE_URLS", raising=False)
        assert _is_safe_host("127.0.0.1") is False

    def test_empty_host_is_unsafe(self):
        assert _is_safe_host("") is False
        assert _is_safe_host(None) is False

    def test_private_ip_literal_is_unsafe(self):
        assert _is_safe_host("192.168.1.1") is False
        assert _is_safe_host("169.254.169.254") is False

    def test_public_ip_literal_is_safe(self):
        assert _is_safe_host("1.1.1.1") is True

    def test_ipv6_literal_in_brackets_is_handled(self):
        # urlparse strips brackets, but defend in depth: _is_safe_host must too.
        assert _is_safe_host("[::1]") is False

    def test_hostname_resolving_to_private_ip_is_unsafe(self, monkeypatch):
        monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: _addrinfo("10.0.0.5"))
        assert _is_safe_host("internal.example.com") is False

    def test_hostname_with_dns_failure_is_unsafe(self, monkeypatch):
        def boom(*args, **kwargs):
            raise socket.gaierror("nope")

        monkeypatch.setattr(socket, "getaddrinfo", boom)
        assert _is_safe_host("nonexistent.invalid") is False

    def test_hostname_with_empty_resolution_is_unsafe(self, monkeypatch):
        monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [])
        assert _is_safe_host("ghost.example.com") is False

    def test_hostname_with_unparseable_resolved_address_is_unsafe(self, monkeypatch):
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("not-an-ip", 0))],
        )
        assert _is_safe_host("weird.example.com") is False


class TestUrlRedirectHandling:
    """The URL reader follows redirects manually so SSRF checks run at every hop."""

    def test_refuses_private_host(self):
        with pytest.raises(UrlFetchError, match="non-public host"):
            _get_media("http://127.0.0.1/secret")

    def test_refuses_aws_metadata_url(self):
        with pytest.raises(UrlFetchError, match="non-public host"):
            _get_media("http://169.254.169.254/latest/meta-data/")

    def test_follows_safe_redirect(self, mocker):
        redirect = _build_response(
            content=b"", status_code=302, is_redirect=True, location="https://2.2.2.2/final.html"
        )
        final = _build_response(content=b"ok", content_type="text/html")
        _, client = _mock_sync_http_client(mocker, side_effect=[redirect, final])

        media_bytes, media_type = _get_media("https://1.1.1.1/start")

        assert media_bytes == b"ok"
        assert media_type == "text/html"
        assert client.send.call_count == 2

    def test_blocks_redirect_to_private_host(self, mocker):
        redirect = _build_response(
            content=b"",
            status_code=302,
            is_redirect=True,
            location="http://169.254.169.254/latest/meta-data/",
        )
        _mock_sync_http_client(mocker, response=redirect)

        with pytest.raises(UrlFetchError, match="non-public host"):
            _get_media("https://1.1.1.1/start")

    def test_redirect_without_location_raises_url_fetch_error(self, mocker):
        no_location = _build_response(content=b"", status_code=302, is_redirect=True, location=None)
        _mock_sync_http_client(mocker, response=no_location)

        with pytest.raises(UrlFetchError, match="missing Location"):
            _get_media("https://1.1.1.1/start")

    def test_too_many_redirects_raises(self, mocker):
        redirect = _build_response(
            content=b"", status_code=302, is_redirect=True, location="https://1.1.1.1/loop"
        )
        _mock_sync_http_client(mocker, response=redirect)

        with pytest.raises(UrlFetchError, match="Too many redirects"):
            _get_media("https://1.1.1.1/loop")


def _mock_async_http_client(*, response=None, side_effect=None):
    """Build an AsyncClient double for the streaming URL reader."""
    client = MagicMock()
    client.build_request.return_value = MagicMock()
    client.send = (
        AsyncMock(side_effect=side_effect) if side_effect else AsyncMock(return_value=response)
    )
    return client


class TestUrlRedirectHandlingAsync:
    async def test_refuses_private_host_before_request(self):
        client = _mock_async_http_client(response=_build_response())

        with pytest.raises(UrlFetchError, match="non-public host"):
            await _read_url_with_client_async("http://127.0.0.1/secret", client, limit=1024)

        client.send.assert_not_awaited()

    async def test_follows_safe_redirect(self):
        redirect = _build_response(
            content=b"", status_code=302, is_redirect=True, location="https://2.2.2.2/final"
        )
        final = _build_response(content=b"ok")
        client = _mock_async_http_client(side_effect=[redirect, final])

        content, headers = await _read_url_with_client_async(
            "https://1.1.1.1/start", client, limit=1024
        )

        assert content == b"ok"
        assert headers["content-type"] == "application/octet-stream"
        assert client.send.await_count == 2

    async def test_blocks_redirect_to_private_host(self):
        redirect = _build_response(
            content=b"",
            status_code=302,
            is_redirect=True,
            location="http://169.254.169.254/latest/meta-data/",
        )
        client = _mock_async_http_client(response=redirect)

        with pytest.raises(UrlFetchError, match="non-public host"):
            await _read_url_with_client_async("https://1.1.1.1/start", client, limit=1024)

        client.send.assert_awaited_once()

    async def test_redirect_without_location_raises(self):
        response = _build_response(is_redirect=True, status_code=302, location=None)
        client = _mock_async_http_client(response=response)

        with pytest.raises(UrlFetchError, match="missing Location"):
            await _read_url_with_client_async("https://1.1.1.1/start", client, limit=1024)

    async def test_too_many_redirects_raises(self):
        redirect = _build_response(
            is_redirect=True, status_code=302, location="https://1.1.1.1/loop"
        )
        client = _mock_async_http_client(response=redirect)

        with pytest.raises(UrlFetchError, match="Too many redirects"):
            await _read_url_with_client_async("https://1.1.1.1/loop", client, limit=1024)


class TestGetMediaAsync:
    async def test_url_without_client_owns_client_lifecycle(self, mocker):
        response = _build_response(content=b"pdf", content_type="application/pdf")
        client = MagicMock()
        client.build_request.return_value = MagicMock()
        client.send = AsyncMock(return_value=response)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client_cls = mocker.patch("openextract._media.httpx.AsyncClient", return_value=client)

        result = await _get_media_async("https://1.1.1.1/download")

        assert result == (b"pdf", "application/pdf")
        client_cls.assert_called_once_with(follow_redirects=False, timeout=30.0)
        client.__aexit__.assert_awaited_once()

    async def test_fetches_url_and_uses_response_media_type(self):
        response = _build_response(content=b"pdf", content_type="application/pdf")
        client = MagicMock()
        client.build_request.return_value = MagicMock()
        client.send = AsyncMock(return_value=response)

        media_bytes, media_type = await _get_media_async("https://1.1.1.1/download", client)

        assert media_bytes == b"pdf"
        assert media_type == "application/pdf"

    async def test_reads_local_path_with_override(self, tmp_path):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")

        result = await _get_media_async(str(local), MagicMock(), media_type="application/custom")

        assert result == (b"hello", "application/custom")

    async def test_missing_local_file_raises_input_file_error(self, tmp_path):
        missing = tmp_path / "gone.txt"
        with pytest.raises(InputFileError, match="gone.txt") as exc_info:
            await _get_media_async(str(missing), MagicMock())
        assert str(tmp_path) not in str(exc_info.value)

    async def test_bytes_input_uses_existing_validation(self):
        result = await _get_media_async(b"hello", MagicMock(), media_type="text/plain")

        assert result == (b"hello", "text/plain")

    async def test_filelike_read_is_offloaded(self):
        result = await _get_media_async(io.BytesIO(b"hello"), MagicMock(), media_type="text/plain")

        assert result == (b"hello", "text/plain")

    async def test_oversized_url_is_rejected_while_streaming(self):
        response = _build_response(content=b"123456")
        client = MagicMock()
        client.build_request.return_value = MagicMock()
        client.send = AsyncMock(return_value=response)

        with pytest.raises(InputTooLargeError, match=r"5 bytes.*at least 6 bytes"):
            await _get_media_async(
                "https://1.1.1.1/file.bin",
                client,
                max_input_bytes=5,
            )

        response.aclose.assert_awaited_once()

    async def test_content_length_fast_fails_before_async_iteration(self):
        response = _build_response(content=b"", content_length=6)
        client = MagicMock()
        client.build_request.return_value = MagicMock()
        client.send = AsyncMock(return_value=response)

        with pytest.raises(InputTooLargeError, match=r"5 bytes.*at least 6 bytes"):
            await _get_media_async(
                "https://1.1.1.1/file.bin",
                client,
                max_input_bytes=5,
            )

        response.aiter_bytes.assert_not_called()

    async def test_streaming_fetch_refuses_private_host_before_request(self):
        client = MagicMock()
        client.send = AsyncMock()

        with pytest.raises(UrlFetchError, match="non-public host"):
            await _get_media_async(
                "http://127.0.0.1/secret",
                client,
                max_input_bytes=5,
            )

        client.send.assert_not_awaited()

    async def test_streaming_fetch_follows_redirect(self):
        redirect = _build_response(
            is_redirect=True,
            status_code=302,
            location="https://2.2.2.2/final.bin",
        )
        final = _build_response(content=b"ok")
        client = MagicMock()
        client.build_request.return_value = MagicMock()
        client.send = AsyncMock(side_effect=[redirect, final])

        result = await _get_media_async(
            "https://1.1.1.1/start",
            client,
            max_input_bytes=5,
        )

        assert result[0] == b"ok"
        assert client.send.await_count == 2
        redirect.aclose.assert_awaited_once()
        final.aclose.assert_awaited_once()

    async def test_streaming_redirect_without_location_is_rejected(self):
        response = _build_response(is_redirect=True, status_code=302)
        client = MagicMock()
        client.build_request.return_value = MagicMock()
        client.send = AsyncMock(return_value=response)

        with pytest.raises(UrlFetchError, match="missing Location"):
            await _get_media_async(
                "https://1.1.1.1/start",
                client,
                max_input_bytes=5,
            )

    async def test_streaming_redirect_limit_is_enforced(self, monkeypatch):
        monkeypatch.setenv("OPENEXTRACT_MAX_REDIRECTS", "2")
        response = _build_response(
            is_redirect=True,
            status_code=302,
            location="https://1.1.1.1/loop",
        )
        client = MagicMock()
        client.build_request.return_value = MagicMock()
        client.send = AsyncMock(return_value=response)

        with pytest.raises(UrlFetchError, match="Too many redirects"):
            await _get_media_async(
                "https://1.1.1.1/loop",
                client,
                max_input_bytes=5,
            )

        assert response.aclose.await_count == 2

    async def test_unsupported_input_preserves_type_error(self):
        with pytest.raises(TypeError, match="input_file must be"):
            await _get_media_async(123, MagicMock())  # type: ignore[arg-type]


class TestUrlFetchConfiguration:
    def test_default_timeout_and_redirects(self, monkeypatch):
        monkeypatch.delenv("OPENEXTRACT_URL_TIMEOUT", raising=False)
        monkeypatch.delenv("OPENEXTRACT_MAX_REDIRECTS", raising=False)
        assert _url_fetch_timeout() == 30.0
        assert _max_redirects() == 10

    def test_custom_timeout_from_env(self, monkeypatch, mocker):
        monkeypatch.setenv("OPENEXTRACT_URL_TIMEOUT", "45")
        fake_response = _build_response(content=b"ok")
        client_cls, _ = _mock_sync_http_client(mocker, response=fake_response)

        _get_media("https://1.1.1.1/doc.pdf")

        assert client_cls.call_args.kwargs["timeout"] == 45.0

    def test_invalid_timeout_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("OPENEXTRACT_URL_TIMEOUT", "not-a-number")
        assert _url_fetch_timeout() == 30.0

    def test_non_positive_timeout_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("OPENEXTRACT_URL_TIMEOUT", "0")
        assert _url_fetch_timeout() == 30.0

    def test_custom_max_redirects_from_env(self, monkeypatch, mocker):
        monkeypatch.setenv("OPENEXTRACT_MAX_REDIRECTS", "2")
        redirect = _build_response(
            content=b"", status_code=302, is_redirect=True, location="https://1.1.1.1/loop"
        )
        _mock_sync_http_client(mocker, response=redirect)

        with pytest.raises(UrlFetchError, match="Too many redirects \\(>2\\)"):
            _get_media("https://1.1.1.1/loop")

    def test_invalid_max_redirects_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("OPENEXTRACT_MAX_REDIRECTS", "-1")
        assert _max_redirects() == 10

    def test_invalid_redirect_count_string_falls_back(self, monkeypatch):
        monkeypatch.setenv("OPENEXTRACT_MAX_REDIRECTS", "not-a-number")
        assert _max_redirects() == 10


class TestSsrfIntegration:
    def test_extract_with_private_url_raises_url_fetch_error(self, mocker):
        # Agent should never be constructed when the URL is rejected.
        agent_cls = mocker.patch("openextract._agent.Agent")
        with pytest.raises(UrlFetchError):
            extract(
                schema=_Person,
                model="openai:gpt-5",
                input_file="http://169.254.169.254/latest/meta-data/",
            )
        agent_cls.assert_not_called()


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


class _Person(BaseModel):
    name: str
    age: int


def _make_bare_provider_error(module_name: str, attr: str, message: str) -> Exception:
    """Build a subclass of a provider's API-error type that bypasses its __init__.

    Most provider error classes require a constructed request/response object;
    we only care about ``isinstance`` matching, so we skip straight to
    ``Exception.__init__`` with the message.
    """
    import importlib

    base = getattr(importlib.import_module(module_name), attr)

    class _BareProviderError(base):
        def __init__(self, msg: str):
            Exception.__init__(self, msg)

    return _BareProviderError(message)


def _make_pydantic_ai_error(message: str) -> Exception:
    from pydantic_ai.exceptions import ModelHTTPError

    return ModelHTTPError(status_code=503, model_name="gpt-5", body=message)


def _make_bedrock_error(message: str) -> Exception:
    from botocore.exceptions import ClientError

    return ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": message}},
        "InvokeModel",
    )


def _make_cohere_error(message: str) -> Exception:
    from cohere.core.api_error import ApiError

    class _BareCohereError(ApiError):
        def __init__(self, msg: str):
            Exception.__init__(self, msg)
            # ApiError.__str__ touches these attributes.
            self.headers = None
            self.status_code = 401
            self.body = msg

    return _BareCohereError(message)


def _make_mistral_error(message: str) -> Exception:
    from mistralai.client.errors.sdkerror import SDKError

    class _BareMistralError(SDKError):
        def __init__(self, msg: str):
            Exception.__init__(self, msg)
            # SDKError uses __slots__; bypass via object.__setattr__.
            object.__setattr__(self, "message", msg)

    return _BareMistralError(message)


def _make_grpc_error(message: str) -> Exception:
    import grpc
    from grpc import StatusCode

    class _BareGrpcError(grpc.RpcError):
        def code(self) -> grpc.StatusCode:
            return StatusCode.RESOURCE_EXHAUSTED

        def details(self) -> str:
            return message

    return _BareGrpcError()


def _make_agent_mock(mocker, output=None, run_sync_side_effect=None, usage=None):
    """Patch openextract._agent.Agent and return (AgentClass, instance, run_sync)."""
    agent_instance = MagicMock()
    if run_sync_side_effect is not None:
        agent_instance.run_sync.side_effect = run_sync_side_effect
    else:
        run_result = MagicMock()
        run_result.output = output
        if usage is not None:
            run_result.usage.return_value = usage
        agent_instance.run_sync.return_value = run_result
    agent_cls = mocker.patch("openextract._agent.Agent", return_value=agent_instance)
    return agent_cls, agent_instance


def test_model_error_classification_does_not_import_provider_sdks():
    from pydantic_ai.exceptions import ModelAPIError

    provider_prefixes = (
        "openai",
        "anthropic",
        "google.genai",
        "botocore",
        "cohere",
        "huggingface_hub",
        "groq",
        "mistralai",
        "grpc",
    )
    modules_before = set(sys.modules)

    mapped = _map_exception(ModelAPIError(model_name="openai:gpt-5", message="failed"))

    newly_loaded_provider_modules = {
        module_name
        for module_name in set(sys.modules) - modules_before
        if any(
            module_name == prefix or module_name.startswith(f"{prefix}.")
            for prefix in provider_prefixes
        )
    }
    assert isinstance(mapped, ModelError)
    assert newly_loaded_provider_modules == set()


def test_package_import_defers_pydantic_ai_runtime():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import openextract; assert 'pydantic_ai' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_lazy_agent_proxy_constructs_pydantic_agent(mocker):
    expected = MagicMock()
    pydantic_agent = mocker.patch("pydantic_ai.Agent", return_value=expected)

    result = agent_module.Agent("openai:gpt-5", output_type=_Person)

    assert result is expected
    pydantic_agent.assert_called_once_with("openai:gpt-5", output_type=_Person)


class TestExtract:
    def test_oversized_input_fails_before_agent_build(self, mocker):
        agent = mocker.patch("openextract._agent.Agent")
        sleep = mocker.patch("openextract._retry.time.sleep")

        with pytest.raises(InputTooLargeError):
            extract(
                schema=_Person,
                model="openai:gpt-5",
                input_file=b"123456",
                media_type="text/plain",
                max_input_bytes=5,
                max_retries=3,
            )

        agent.assert_not_called()
        sleep.assert_not_called()

    def test_usage_helper_enforces_input_limit(self, mocker):
        agent = mocker.patch("openextract._agent.Agent")

        with pytest.raises(InputTooLargeError):
            extract_with_usage(
                schema=_Person,
                model="openai:gpt-5",
                input_file=b"123456",
                media_type="text/plain",
                max_input_bytes=5,
            )

        agent.assert_not_called()

    def test_library_does_not_load_dotenv(self, tmp_path, monkeypatch, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        (tmp_path / ".env").write_text("OPENEXTRACT_TEST_DOTENV=loaded\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("OPENEXTRACT_TEST_DOTENV", raising=False)
        _make_agent_mock(mocker, output=_Person(name="Ada", age=36))

        extract(schema=_Person, model="openai:gpt-5", input_file=str(local))

        assert "OPENEXTRACT_TEST_DOTENV" not in os.environ

    def test_returns_schema_instance_from_local_file(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        expected = _Person(name="Ada", age=36)
        agent_cls, agent_instance = _make_agent_mock(mocker, output=expected)

        result = extract(
            schema=_Person,
            model="openai:gpt-5",
            input_file=str(local),
            instructions="pull the person",
        )

        assert result is expected
        agent_cls.assert_called_once()
        assert agent_cls.call_args.args == ("openai-responses:gpt-5",)
        # Non-ollama models pass the schema directly as output_type.
        kwargs = agent_cls.call_args.kwargs
        assert kwargs["instructions"] == "pull the person"
        assert kwargs["output_type"] is _Person
        agent_instance.run_sync.assert_called_once()

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("openai:gpt-5.6-luna", "openai-responses:gpt-5.6-luna"),
            ("openai-responses:gpt-5.6-luna", "openai-responses:gpt-5.6-luna"),
            ("openai-chat:gpt-5.6-luna", "openai-chat:gpt-5.6-luna"),
            ("anthropic:claude-sonnet-4", "anthropic:claude-sonnet-4"),
        ],
    )
    def test_model_routing(self, tmp_path, mocker, model, expected):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        agent_cls, _ = _make_agent_mock(mocker, output=_Person(name="Ada", age=36))

        extract(schema=_Person, model=model, input_file=str(local))

        assert agent_cls.call_args.args == (expected,)

    def test_ollama_model_wraps_output_in_native_output(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        expected = _Person(name="Linus", age=54)
        agent_cls, _ = _make_agent_mock(mocker, output=expected)

        result = extract(
            schema=_Person,
            model="ollama:llama3",
            input_file=str(local),
        )

        assert result is expected
        output_type = agent_cls.call_args.kwargs["output_type"]
        # NativeOutput wraps the schema; assert it's not the bare schema class.
        assert output_type is not _Person
        assert type(output_type).__name__ == "NativeOutput"

    def test_http_status_error_is_wrapped(self, mocker):
        response = MagicMock()
        response.status_code = 503
        err = httpx.HTTPStatusError("boom", request=MagicMock(), response=response)
        mocker.patch("openextract._extract._get_media", side_effect=err)

        with pytest.raises(UrlFetchError, match="503"):
            extract(schema=_Person, model="openai:gpt-5", input_file="https://x/y")

    def test_request_error_is_wrapped(self, mocker):
        err = httpx.ConnectError("dns failure")
        mocker.patch("openextract._extract._get_media", side_effect=err)

        with pytest.raises(UrlFetchError, match="dns failure"):
            extract(schema=_Person, model="openai:gpt-5", input_file="https://x/y")

    def test_http_404_from_url_becomes_url_fetch_error(self, mocker):
        """End-to-end: a 404 response from the URL fetch surfaces as UrlFetchError, not garbage."""
        fake_response = _build_response(content=b"<html>not found</html>")
        fake_response.status_code = 404
        fake_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=fake_response
        )
        _mock_sync_http_client(mocker, response=fake_response)

        with pytest.raises(UrlFetchError, match="404"):
            extract(
                schema=_Person,
                model="openai:gpt-5",
                input_file="https://example.com/missing.pdf",
            )

    def test_validation_error_is_wrapped(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        try:
            _Person(name="x", age="not-an-int")  # type: ignore[arg-type]
        except ValidationError as exc:
            validation_error = exc
        _make_agent_mock(mocker, run_sync_side_effect=validation_error)

        with pytest.raises(SchemaValidationError, match=r"age: expected int, got str"):
            extract(schema=_Person, model="openai:gpt-5", input_file=str(local))

    @pytest.mark.parametrize(
        "factory,model_id",
        [
            (lambda msg: _make_pydantic_ai_error(msg), "openai:gpt-5"),
            (lambda msg: _make_bare_provider_error("openai", "APIError", msg), "openai:gpt-5"),
            (
                lambda msg: _make_bare_provider_error("anthropic", "APIError", msg),
                "anthropic:claude-sonnet-4",
            ),
            (
                lambda msg: _make_bare_provider_error("google.genai.errors", "APIError", msg),
                "google-gla:gemini-2.5-pro",
            ),
            (
                lambda msg: _make_bedrock_error(msg),
                "bedrock:anthropic.claude-sonnet-4-20250514-v1:0",
            ),
            (lambda msg: _make_cohere_error(msg), "cohere:command-r-plus"),
            (
                lambda msg: _make_bare_provider_error(
                    "huggingface_hub.errors", "HfHubHTTPError", msg
                ),
                "huggingface:meta-llama/Llama-3.3-70B-Instruct",
            ),
            (
                lambda msg: _make_bare_provider_error("groq", "APIError", msg),
                "groq:llama-3.3-70b-versatile",
            ),
            (lambda msg: _make_mistral_error(msg), "mistral:mistral-large-latest"),
            (lambda msg: _make_grpc_error(msg), "xai:grok-4.3"),
        ],
        ids=[
            "pydantic_ai_ModelHTTPError",
            "openai_APIError",
            "anthropic_APIError",
            "google_APIError",
            "bedrock_ClientError",
            "cohere_ApiError",
            "huggingface_HfHubHTTPError",
            "groq_APIError",
            "mistral_SDKError",
            "xai_grpc_RpcError",
        ],
    )
    def test_provider_api_error_is_wrapped_as_model_error(
        self, tmp_path, mocker, factory, model_id
    ):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        _make_agent_mock(mocker, run_sync_side_effect=factory("rate limited"))

        with pytest.raises(ModelError, match="Model API error"):
            extract(schema=_Person, model=model_id, input_file=str(local))

    def test_message_mentioning_model_is_wrapped_as_extraction_error(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        _make_agent_mock(
            mocker,
            run_sync_side_effect=RuntimeError("unknown model identifier"),
        )

        with pytest.raises(ExtractionError, match="Extraction failed: unknown model identifier"):
            extract(schema=_Person, model="openai:gpt-5", input_file=str(local))

    def test_generic_exception_is_wrapped_as_extraction_error(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        _make_agent_mock(mocker, run_sync_side_effect=RuntimeError("kaboom"))

        with pytest.raises(ExtractionError, match="Extraction failed: kaboom"):
            extract(schema=_Person, model="openai:gpt-5", input_file=str(local))

    def test_message_mentioning_model_is_not_subclass_of_model_error(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        _make_agent_mock(
            mocker,
            run_sync_side_effect=RuntimeError("the model said no"),
        )

        with pytest.raises(ExtractionError) as exc_info:
            extract(schema=_Person, model="openai:gpt-5", input_file=str(local))
        # The new classifier should NOT promote this to ModelError just because
        # the message mentions "model".
        assert not isinstance(exc_info.value, ModelError)

    def test_passes_through_existing_extraction_error(self, tmp_path, mocker):
        """If the wrapped code already raises an ExtractionError, it is not re-wrapped."""
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        original = ModelError("already mapped")
        _make_agent_mock(mocker, run_sync_side_effect=original)

        with pytest.raises(ModelError) as exc_info:
            extract(schema=_Person, model="openai:gpt-5", input_file=str(local))
        assert exc_info.value is original

    @pytest.mark.parametrize(
        "status_code,retryable",
        [(400, False), (401, False), (403, False), (429, True), (500, True), (503, True)],
    )
    def test_model_http_error_preserves_retry_metadata(self, status_code, retryable):
        from pydantic_ai.exceptions import ModelHTTPError

        provider_error = ModelHTTPError(
            status_code=status_code,
            model_name="openai:gpt-5",
            body="provider failed",
        )
        provider_error.headers = {"Retry-After": "12.5"}

        mapped = _map_exception(provider_error)

        assert isinstance(mapped, ModelError)
        assert mapped.provider == "openai"
        assert mapped.status_code == status_code
        assert mapped.retryable is retryable
        assert mapped.retry_after == 12.5

    def test_transient_transport_model_error_is_retryable(self):
        from pydantic_ai.exceptions import ModelAPIError

        class APITimeoutError(ModelAPIError):
            pass

        mapped = _map_exception(APITimeoutError(model_name="anthropic:claude", message="timed out"))

        assert isinstance(mapped, ModelError)
        assert mapped.provider == "anthropic"
        assert mapped.status_code is None
        assert mapped.retryable is True
        assert mapped.retry_after is None

    def test_bedrock_transport_error_is_retryable(self):
        from botocore.exceptions import EndpointConnectionError

        mapped = _map_exception(EndpointConnectionError(endpoint_url="https://bedrock.example.com"))

        assert isinstance(mapped, ModelError)
        assert mapped.provider == "bedrock"
        assert mapped.status_code is None
        assert mapped.retryable is True

    def test_unknown_model_error_is_fail_fast(self):
        from pydantic_ai.exceptions import ModelAPIError

        mapped = _map_exception(ModelAPIError(model_name="custom-model", message="failed"))

        assert isinstance(mapped, ModelError)
        assert mapped.provider is None
        assert mapped.retryable is False

    def test_token_limit_is_permanent_even_without_model_signature(self):
        mapped = _map_exception(
            RuntimeError(
                "Model token limit (provider default) exceeded before any response was generated."
            )
        )
        assert isinstance(mapped, ModelError)
        assert mapped.retryable is False

    def test_output_retry_exhausted_is_retryable_model_error(self):
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        mapped = _map_exception(UnexpectedModelBehavior("Exceeded maximum output retries (1)"))
        assert isinstance(mapped, ModelError)
        assert mapped.retryable is True
        assert "output retry" in str(mapped).lower()

    def test_empty_tool_call_message_is_output_retry_error(self):
        exc = RuntimeError("Please return text or include your response in a tool call.")
        assert _is_output_retry_error(exc)
        assert isinstance(_map_exception(exc), ModelError)
        wrapped = ExtractionError("Extraction failed: please retry")
        wrapped.__cause__ = RuntimeError("Exceeded maximum output retries (1)")
        assert _is_output_retry_error(wrapped)

        class ToolRetryError(Exception):
            pass

        assert _is_output_retry_error(ToolRetryError("empty call"))
        assert not _is_output_retry_error(RuntimeError("kaboom"))

    def test_model_error_constructor_remains_backwards_compatible(self):
        error = ModelError("flaky")

        assert str(error) == "flaky"
        assert error.provider is None
        assert error.status_code is None
        assert error.retryable is True
        assert error.retry_after is None

    @pytest.mark.parametrize(
        "status_code,retryable",
        [(400, False), (401, False), (403, False), (429, True), (503, True)],
    )
    def test_model_error_constructor_infers_retryability(self, status_code, retryable):
        error = ModelError("provider failed", status_code=status_code)

        assert error.retryable is retryable


class TestModelErrorMetadataHelpers:
    def test_status_from_response_object(self):
        error = Exception("failed")
        error.response = MagicMock(status_code=502)  # type: ignore[attr-defined]

        assert _model_status_code(error) == 502

    def test_status_and_retry_after_from_raw_response(self):
        error = Exception("failed")
        error.raw_response = MagicMock(  # type: ignore[attr-defined]
            status_code=503,
            headers={"Retry-After": "8"},
        )

        assert _model_status_code(error) == 503
        assert _model_retry_after(error) == 8

    def test_status_from_bedrock_metadata(self):
        error = Exception("failed")
        error.response = {  # type: ignore[attr-defined]
            "ResponseMetadata": {"HTTPStatusCode": 429}
        }

        assert _model_status_code(error) == 429

    def test_status_from_integer_code(self):
        error = Exception("failed")
        error.code = 503  # type: ignore[attr-defined]

        assert _model_status_code(error) == 503

    def test_boolean_status_values_are_ignored(self):
        error = Exception("failed")
        error.status_code = True  # type: ignore[attr-defined]
        error.response = {  # type: ignore[attr-defined]
            "ResponseMetadata": {"HTTPStatusCode": False}
        }
        error.code = True  # type: ignore[attr-defined]

        assert _model_status_code(error) is None

    def test_non_mapping_bedrock_metadata_is_ignored(self):
        error = Exception("failed")
        error.response = {"ResponseMetadata": None}  # type: ignore[attr-defined]

        assert _model_status_code(error) is None

    def test_retry_after_from_response_headers(self):
        error = Exception("failed")
        error.response = MagicMock(  # type: ignore[attr-defined]
            headers={"x-request-id": "123", "Retry-After": "7"}
        )

        assert _model_retry_after(error) == 7

    def test_retry_after_from_direct_attribute(self):
        error = Exception("failed")
        error.retry_after = 6  # type: ignore[attr-defined]

        assert _model_retry_after(error) == 6

    def test_invalid_direct_retry_after_falls_back_to_headers(self):
        error = Exception("failed")
        error.retry_after = "invalid"  # type: ignore[attr-defined]
        error.headers = {"Retry-After": "5"}  # type: ignore[attr-defined]

        assert _model_retry_after(error) == 5

    def test_retry_after_scan_continues_past_header_sets_without_the_key(self):
        error = Exception("failed")
        error.headers = {"x-request-id": "123"}  # type: ignore[attr-defined]
        error.response = MagicMock(headers={"Retry-After": "4"})  # type: ignore[attr-defined]

        assert _model_retry_after(error) == 4

    def test_retry_after_from_bedrock_metadata_headers(self):
        error = Exception("failed")
        error.response = {  # type: ignore[attr-defined]
            "ResponseMetadata": {"HTTPHeaders": {"retry-after": "9"}}
        }

        assert _model_retry_after(error) == 9

    @pytest.mark.parametrize(
        "response",
        [
            {"ResponseMetadata": None},
            {"ResponseMetadata": {"HTTPHeaders": None}},
            {"Error": "not-a-mapping"},
        ],
    )
    def test_malformed_response_metadata_is_ignored(self, response):
        error = Exception("failed")
        error.response = response  # type: ignore[attr-defined]

        assert _model_retry_after(error) is None
        assert _is_transient_model_exception(error, None) is False

    def test_permanent_error_name_is_not_transient(self):
        class AuthenticationError(Exception):
            pass

        assert _is_transient_model_exception(AuthenticationError(), None) is False

    def test_permanent_grpc_status_is_not_transient(self):
        class _Code:
            name = "PERMISSION_DENIED"

        class _GrpcError(Exception):
            def code(self):
                return _Code()

        assert _is_transient_model_exception(_GrpcError(), None) is False

    def test_empty_grpc_status_falls_back_to_class_name(self):
        class ConnectionError(Exception):
            def code(self):
                return object()

        assert _is_transient_model_exception(ConnectionError(), None) is True

    def test_extract_returns_only_model_instance(self, tmp_path, mocker):
        """Regression: extract() still returns just the schema instance, not a tuple."""
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        expected = _Person(name="Grace", age=42)
        usage_obj = MagicMock(input_tokens=1, output_tokens=2, total_tokens=3)
        _make_agent_mock(mocker, output=expected, usage=usage_obj)

        result = extract(schema=_Person, model="openai:gpt-5", input_file=str(local))

        assert result is expected
        assert not isinstance(result, tuple)


# ---------------------------------------------------------------------------
# Provider-not-installed guardrail
# ---------------------------------------------------------------------------


class TestProviderNotInstalled:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("openai:gpt-4o", "openextract[openai]"),
            ("openai-chat:gpt-4o", "openextract[openai]"),
            ("openai-responses:gpt-5.6-luna", "openextract[openai]"),
            ("anthropic:claude-sonnet-4-20250514", "openextract[anthropic]"),
            ("google-gla:gemini-2.5-flash", "openextract[google]"),
            ("google-vertex:gemini-2.5-pro", "openextract[google]"),
            ("bedrock:anthropic.claude-sonnet-4-20250514-v1:0", "openextract[bedrock]"),
            ("cohere:command-r-plus", "openextract[cohere]"),
            ("groq:llama-3.3-70b-versatile", "openextract[groq]"),
            ("huggingface:meta-llama/Llama-3.3-70B-Instruct", "openextract[huggingface]"),
            ("mistral:mistral-large-latest", "openextract[mistral]"),
            ("openrouter:anthropic/claude-sonnet-4", "openextract[openrouter]"),
            ("xai:grok-4.3", "openextract[xai]"),
            ("cerebras:llama-3.3-70b", "openextract[openai]"),
            ("ollama:llama3.2", "openextract[openai]"),
            ("madeup:model", "openextract[all]"),
        ],
    )
    def test_install_hint_maps_prefix_to_extra(self, model, expected):
        assert _install_hint(model) == f"pip install '{expected}'"

    def test_missing_provider_raises_provider_not_installed_error(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        mocker.patch(
            "openextract._agent.Agent",
            side_effect=ImportError("No module named 'openai'"),
        )

        with pytest.raises(ProviderNotInstalledError) as exc_info:
            extract(schema=_Person, model="openai:gpt-4o", input_file=str(local))

        message = str(exc_info.value)
        assert "pip install 'openextract[openai]'" in message
        assert "No module named 'openai'" in message

    def test_provider_not_installed_is_extraction_error(self):
        assert issubclass(ProviderNotInstalledError, ExtractionError)

    def test_input_too_large_is_extraction_error(self):
        assert issubclass(InputTooLargeError, ExtractionError)

    def test_input_file_error_is_extraction_error(self):
        assert issubclass(InputFileError, ExtractionError)


# ---------------------------------------------------------------------------
# extract: input_file polymorphism
# ---------------------------------------------------------------------------


def _binary_content_arg(agent_instance) -> BinaryContent:
    """Pull the BinaryContent that was passed to agent.run_sync."""
    (call_args,) = agent_instance.run_sync.call_args.args
    binary = next(part for part in call_args if isinstance(part, BinaryContent))
    return binary


class TestExtractInputs:
    def test_bytes_input_with_media_type(self, mocker):
        expected = _Person(name="Grace", age=85)
        _, agent_instance = _make_agent_mock(mocker, output=expected)

        result = extract(
            schema=_Person,
            model="openai:gpt-5",
            input_file=b"raw-payload",
            media_type="application/pdf",
        )

        assert result is expected
        binary = _binary_content_arg(agent_instance)
        assert binary.data == b"raw-payload"
        assert binary.media_type == "application/pdf"

    def test_filelike_input_with_media_type(self, mocker):
        expected = _Person(name="Ada", age=36)
        _, agent_instance = _make_agent_mock(mocker, output=expected)
        buffer = io.BytesIO(b"buffered-bytes")

        result = extract(
            schema=_Person,
            model="openai:gpt-5",
            input_file=buffer,
            media_type="image/png",
        )

        assert result is expected
        binary = _binary_content_arg(agent_instance)
        assert binary.data == b"buffered-bytes"
        assert binary.media_type == "image/png"
        # Caller owns the handle; we must not close it.
        assert not buffer.closed

    def test_bytes_input_without_media_type_raises(self, mocker):
        _make_agent_mock(mocker, output=_Person(name="x", age=1))

        with pytest.raises(TypeError, match="media_type is required"):
            extract(schema=_Person, model="openai:gpt-5", input_file=b"abc")

    def test_filelike_input_without_media_type_raises(self, mocker):
        _make_agent_mock(mocker, output=_Person(name="x", age=1))

        with pytest.raises(TypeError, match="media_type is required"):
            extract(
                schema=_Person,
                model="openai:gpt-5",
                input_file=io.BytesIO(b"abc"),
            )

    def test_str_input_still_works(self, tmp_path, mocker):
        local = tmp_path / "doc.txt"
        local.write_bytes(b"file-bytes")
        expected = _Person(name="Linus", age=54)
        _, agent_instance = _make_agent_mock(mocker, output=expected)

        result = extract(
            schema=_Person,
            model="openai:gpt-5",
            input_file=str(local),
        )

        assert result is expected
        binary = _binary_content_arg(agent_instance)
        assert binary.data == b"file-bytes"
        assert binary.media_type == "text/plain"

    def test_str_input_media_type_override(self, tmp_path, mocker):
        local = tmp_path / "doc.txt"
        local.write_bytes(b"file-bytes")
        _, agent_instance = _make_agent_mock(mocker, output=_Person(name="x", age=1))

        extract(
            schema=_Person,
            model="openai:gpt-5",
            input_file=str(local),
            media_type="text/markdown",
        )

        binary = _binary_content_arg(agent_instance)
        assert binary.media_type == "text/markdown"

    def test_unsupported_input_type_raises_type_error(self, mocker):
        _make_agent_mock(mocker, output=_Person(name="x", age=1))

        with pytest.raises(TypeError, match="input_file must be"):
            extract(schema=_Person, model="openai:gpt-5", input_file=12345)  # type: ignore[arg-type]

    def test_missing_local_file_raises_input_file_error(self, tmp_path, mocker):
        _make_agent_mock(mocker, output=_Person(name="x", age=1))
        missing = tmp_path / "gone.txt"

        with pytest.raises(InputFileError, match="gone.txt") as exc_info:
            extract(schema=_Person, model="openai:gpt-5", input_file=str(missing))

        assert str(tmp_path) not in str(exc_info.value)


# ---------------------------------------------------------------------------
# extract_with_usage
# ---------------------------------------------------------------------------


class TestExtractWithUsage:
    def test_returns_output_and_usage_tuple(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        expected = _Person(name="Ada", age=36)
        usage_obj = MagicMock(input_tokens=10, output_tokens=20, total_tokens=30)
        _make_agent_mock(mocker, output=expected, usage=usage_obj)

        output, usage = extract_with_usage(
            schema=_Person,
            model="openai:gpt-5",
            input_file=str(local),
            instructions="pull the person",
        )

        assert output is expected
        assert usage == Usage(input_tokens=10, output_tokens=20, total_tokens=30)

    def test_missing_usage_fields_default_to_zero(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        expected = _Person(name="Linus", age=54)

        class _BareUsage:
            pass

        _make_agent_mock(mocker, output=expected, usage=_BareUsage())

        _, usage = extract_with_usage(
            schema=_Person,
            model="openai:gpt-5",
            input_file=str(local),
        )

        assert usage == Usage(input_tokens=0, output_tokens=0, total_tokens=0)

    def test_zero_input_tokens_do_not_hide_provider_aliases(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        expected = _Person(name="Ada", age=36)
        usage_obj = SimpleNamespace(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            request_tokens=1280,
            prompt_tokens=1280,
            response_tokens=64,
            completion_tokens=64,
        )
        _make_agent_mock(mocker, output=expected, usage=usage_obj)

        output, usage = extract_with_usage(
            schema=_Person,
            model="openai:gpt-5",
            input_file=str(local),
        )

        assert output is expected
        assert usage == Usage(input_tokens=1280, output_tokens=64, total_tokens=1344)

    def test_none_usage_fields_default_to_zero(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        expected = _Person(name="Hedy", age=29)
        usage_obj = MagicMock(input_tokens=None, output_tokens=None, total_tokens=None)
        _make_agent_mock(mocker, output=expected, usage=usage_obj)

        _, usage = extract_with_usage(
            schema=_Person,
            model="openai:gpt-5",
            input_file=str(local),
        )

        assert usage == Usage(input_tokens=0, output_tokens=0, total_tokens=0)

    def test_provider_usage_aliases_and_nested_response(self):
        from openextract._agent import (
            _extract_usage_object,
            _usage_from_raw,
            _usage_from_result,
            _usage_model_settings,
        )

        class _OpenRouterUsage:
            request_tokens = 11
            response_tokens = 7
            total_tokens = 0

        assert _usage_from_raw(_OpenRouterUsage()) == Usage(11, 7, 18)
        assert _usage_from_raw(Usage(1, 2, 3)) == Usage(1, 2, 3)
        assert _usage_from_raw(None) == Usage(0, 0, 0)
        assert _usage_from_raw(
            {"details": {"prompt_tokens": 4, "completion_tokens": 5, "totalTokens": 9}}
        ) == Usage(4, 5, 9)
        assert _usage_from_raw(
            SimpleNamespace(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                request_tokens=1280,
                response_tokens=64,
            )
        ) == Usage(1280, 64, 1344)

        class _Result:
            @property
            def usage(self):
                return SimpleNamespace(input_tokens=0, output_tokens=0, total_tokens=0)

            @property
            def response(self):
                return SimpleNamespace(usage=SimpleNamespace(input_tokens=21, output_tokens=3))

        assert _usage_from_result(_Result()) == Usage(21, 3, 24)
        assert _usage_from_raw(
            SimpleNamespace(details=SimpleNamespace(prompt_tokens=8, completion_tokens=1))
        ) == Usage(8, 1, 9)

        class _NeedsArg:
            def usage(self, extra):
                return extra

        assert _usage_from_result(_NeedsArg()) == Usage(0, 0, 0)
        held = SimpleNamespace(request_tokens=2, response_tokens=3, total_tokens=5)
        assert _extract_usage_object(SimpleNamespace(usage=held)) == held
        assert _extract_usage_object(None) is None
        assert _usage_from_result(SimpleNamespace()) == Usage(0, 0, 0)
        assert _usage_model_settings("openai:gpt-5", None) is None
        settings = _usage_model_settings("openrouter:anthropic/claude-sonnet-4", {"temperature": 0})
        assert settings["openrouter_usage"] == {"include": True}
        assert settings["extra_body"]["usage"] == {"include": True}
        assert settings["temperature"] == 0
        merged = _usage_model_settings(
            "openrouter:z-ai/glm-5.3-flash", {"extra_body": {"provider": {"sort": "price"}}}
        )
        assert merged["extra_body"]["usage"] == {"include": True}
        assert merged["extra_body"]["provider"] == {"sort": "price"}
        replaced = _usage_model_settings("openrouter:x", {"extra_body": "nope"})
        assert replaced["extra_body"]["usage"] == {"include": True}

        class _ZeroRunUsage:
            input_tokens = 0
            output_tokens = 0

            @property
            def request_tokens(self):
                return self.input_tokens

            @property
            def total_tokens(self):
                return 0

            details = {}

        class _LiveOpenRouterResult:
            @property
            def usage(self):
                return _ZeroRunUsage()

            def all_messages(self):
                return [
                    SimpleNamespace(
                        usage=None,
                        provider_details={
                            "native_tokens_prompt": 640,
                            "native_tokens_completion": 16,
                        },
                    )
                ]

        assert _usage_from_result(_LiveOpenRouterResult()) == Usage(640, 16, 656)
        nested_usage = {"usage": {"prompt_tokens": "99", "completion_tokens": 2.0}}
        assert _usage_from_raw(nested_usage) == Usage(99, 2, 101)
        assert _usage_from_raw({"prompt_tokens": True}) == Usage(0, 0, 0)
        assert _usage_from_raw({"completion_tokens": "nope"}) == Usage(0, 0, 0)
        too_deep = {"details": {"details": {"details": {"details": {"details": {}}}}}}
        too_deep["details"]["details"]["details"]["details"]["details"]["prompt_tokens"] = 5
        assert _usage_from_raw(too_deep) == Usage(0, 0, 0)
        assert _usage_from_raw({"tokens_prompt": 40, "tokens_completion": 2}) == Usage(40, 2, 42)

        class _Dumped:
            input_tokens = 0
            output_tokens = 0

            def model_dump(self, exclude_none=True):
                return {"prompt_tokens": 77, "completion_tokens": 3}

        assert _usage_from_raw(_Dumped()) == Usage(77, 3, 80)

        class _DumpNeedsNoKwargs:
            def model_dump(self):
                return {"prompt_tokens": 5, "completion_tokens": 1}

        assert _usage_from_raw(_DumpNeedsNoKwargs()) == Usage(5, 1, 6)

        class _DumpBroken:
            def model_dump(self, extra):
                return extra

        assert _usage_from_raw(_DumpBroken()) == Usage(0, 0, 0)

        class _DumpList:
            def model_dump(self, exclude_none=True):
                return [1]

        assert _usage_from_raw(_DumpList()) == Usage(0, 0, 0)

        class _DumpNoTokenAliases:
            def model_dump(self, exclude_none=True):
                return {"details": {}}

        assert _usage_from_raw(_DumpNoTokenAliases()) == Usage(0, 0, 0)

        class _NestedDump:
            input_tokens = 0
            output_tokens = 0

            def model_dump(self, exclude_none=True):
                return {"details": {"prompt_tokens": 8, "completion_tokens": 1}}

        assert _usage_from_raw(_NestedDump()) == Usage(8, 1, 9)

        class _LiveProviderResponse:
            @property
            def usage(self):
                return SimpleNamespace(input_tokens=0, output_tokens=0, total_tokens=0)

            def all_messages(self):
                return [
                    SimpleNamespace(
                        usage=SimpleNamespace(input_tokens=0, output_tokens=0),
                        provider_details={},
                        provider_response=SimpleNamespace(
                            usage=SimpleNamespace(prompt_tokens=321, completion_tokens=7)
                        ),
                    )
                ]

        assert _usage_from_result(_LiveProviderResponse()) == Usage(321, 7, 328)

        class _EmptyResponse:
            @property
            def response(self):
                raise ValueError("No response found")

        assert _usage_from_result(_EmptyResponse()) == Usage(0, 0, 0)

    def test_openai_usage_fallback_reads_provider_prompt_tokens(self, monkeypatch):
        from pydantic_ai.models import openai as openai_mod

        import openextract._agent as agent_mod

        class _Mapped:
            def __init__(self, inp=0, out=0):
                self.input_tokens = inp
                self.output_tokens = out

        monkeypatch.setattr(agent_mod, "_OPENAI_USAGE_PATCHED", False)
        monkeypatch.setattr(openai_mod, "_map_usage", lambda *_a, **_k: _Mapped())
        agent_mod._ensure_openai_usage_fallback()
        agent_mod._ensure_openai_usage_fallback()
        filled = openai_mod._map_usage(
            SimpleNamespace(usage=SimpleNamespace(prompt_tokens=99, completion_tokens=4)),
            "openrouter",
            "https://openrouter.ai/api/v1",
            "z-ai/glm-5.3-flash",
        )
        assert (filled.input_tokens, filled.output_tokens) == (99, 4)
        empty = openai_mod._map_usage(SimpleNamespace(usage=None), "openrouter", "https://x", "m")
        assert (empty.input_tokens, empty.output_tokens) == (0, 0)

        monkeypatch.setattr(agent_mod, "_OPENAI_USAGE_PATCHED", False)
        monkeypatch.setattr(openai_mod, "_map_usage", lambda *_a, **_k: _Mapped(5, 1))
        agent_mod._ensure_openai_usage_fallback()
        kept = openai_mod._map_usage(SimpleNamespace(usage=None), "openrouter", "https://x", "m")
        assert (kept.input_tokens, kept.output_tokens) == (5, 1)

        monkeypatch.setattr(agent_mod, "_OPENAI_USAGE_PATCHED", False)
        real_import = __import__

        def _boom(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "pydantic_ai.models" and fromlist and "openai" in fromlist:
                raise ImportError("no openai extra")
            return real_import(name, globals, locals, fromlist, level)

        import builtins

        monkeypatch.setattr(builtins, "__import__", _boom)
        agent_mod._ensure_openai_usage_fallback()
        assert agent_mod._OPENAI_USAGE_PATCHED is False

    def test_propagates_runtime_error_as_extraction_error(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        _make_agent_mock(mocker, run_sync_side_effect=RuntimeError("kaboom"))

        with pytest.raises(ExtractionError, match="Extraction failed: kaboom"):
            extract_with_usage(schema=_Person, model="openai:gpt-5", input_file=str(local))

    def test_usage_is_frozen_dataclass(self):
        usage = Usage(input_tokens=1, output_tokens=2, total_tokens=3)
        with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError varies by py version
            usage.input_tokens = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# extract retry behavior
# ---------------------------------------------------------------------------


class TestRetryAfterParsing:
    @pytest.mark.parametrize("value", ["2.5", 3, 4.5])
    def test_parses_delta_seconds(self, value):
        assert _parse_retry_after(value) == float(value)

    def test_parses_http_date(self, mocker):
        mocker.patch("openextract._errors.time.time", return_value=1445412450.0)

        assert _parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 30.0

    @pytest.mark.parametrize(
        "value",
        [None, True, -1, float("inf"), "not-a-date"],
    )
    def test_rejects_invalid_values(self, value):
        assert _parse_retry_after(value) is None

    def test_extreme_attempt_count_is_bounded_without_overflow(self):
        assert _retry_delay(1.0, 60.0, 10_000, None) == 60.0


class TestExtractRetry:
    def test_invalid_max_retries_is_rejected_before_extraction(self, mocker):
        once = mocker.patch("openextract._extract._extract_once")

        with pytest.raises(ValueError, match="max_retries"):
            extract(
                schema=_Person,
                model="openai:gpt-5",
                input_file="ignored",
                max_retries=-1,
            )

        once.assert_not_called()

    @pytest.mark.parametrize(
        "retry_backoff",
        [-1.0, float("inf"), float("nan"), "slow", True],
    )
    def test_invalid_retry_backoff_is_rejected_before_extraction(self, mocker, retry_backoff):
        once = mocker.patch("openextract._extract._extract_once")

        with pytest.raises(ValueError, match="retry_backoff"):
            extract(
                schema=_Person,
                model="openai:gpt-5",
                input_file="ignored",
                retry_backoff=retry_backoff,  # type: ignore[arg-type]
            )

        once.assert_not_called()

    @pytest.mark.parametrize(
        "retry_max_backoff",
        [-1.0, float("inf"), float("nan"), "slow", True],
    )
    def test_invalid_retry_max_backoff_is_rejected_before_extraction(
        self, mocker, retry_max_backoff
    ):
        once = mocker.patch("openextract._extract._extract_once")

        with pytest.raises(ValueError, match="retry_max_backoff"):
            extract(
                schema=_Person,
                model="openai:gpt-5",
                input_file="ignored",
                retry_max_backoff=retry_max_backoff,  # type: ignore[arg-type]
            )

        once.assert_not_called()

    def test_zero_backoff_values_are_allowed(self, mocker):
        expected = _Person(name="Grace", age=85)
        mocker.patch(
            "openextract._extract._prepare_extraction",
            return_value=nullcontext((MagicMock(), ["prepared"], None)),
        )
        once = mocker.patch(
            "openextract._extract._extract_once",
            side_effect=[ModelError("flaky"), expected],
        )
        sleep_mock = mocker.patch("openextract._retry.time.sleep")

        result = extract(
            schema=_Person,
            model="openai:gpt-5",
            input_file="ignored",
            max_retries=1,
            retry_backoff=0,
            retry_max_backoff=0,
        )

        assert result is expected
        assert once.call_count == 2
        sleep_mock.assert_called_once_with(0)

    def test_no_retry_by_default_raises_immediately(self, mocker):
        sleep_mock = mocker.patch("openextract._retry.time.sleep")
        prepare = mocker.patch(
            "openextract._extract._prepare_extraction",
            return_value=nullcontext((MagicMock(), ["prepared"], None)),
        )
        once = mocker.patch(
            "openextract._extract._extract_once",
            side_effect=ModelError("upstream down"),
        )

        with pytest.raises(ModelError, match="upstream down"):
            extract(schema=_Person, model="openai:gpt-5", input_file="ignored")

        assert once.call_count == 1
        prepare.assert_called_once()
        sleep_mock.assert_not_called()

    def test_retry_succeeds_after_transient_model_errors(self, mocker):
        sleep_mock = mocker.patch("openextract._retry.time.sleep")
        prepare = mocker.patch(
            "openextract._extract._prepare_extraction",
            return_value=nullcontext((MagicMock(), ["prepared"], None)),
        )
        expected = _Person(name="Grace", age=85)
        once = mocker.patch(
            "openextract._extract._extract_once",
            side_effect=[ModelError("flaky"), ModelError("flaky"), expected],
        )

        result = extract(
            schema=_Person,
            model="openai:gpt-5",
            input_file="ignored",
            max_retries=2,
        )

        assert result is expected
        assert once.call_count == 3
        prepare.assert_called_once()
        assert sleep_mock.call_count == 2

    @pytest.mark.parametrize("status_code", [400, 401, 403, 422])
    def test_permanent_model_errors_are_not_retried(self, mocker, status_code):
        sleep_mock = mocker.patch("openextract._retry.time.sleep")
        mocker.patch(
            "openextract._extract._prepare_extraction",
            return_value=nullcontext((MagicMock(), ["prepared"], None)),
        )
        once = mocker.patch(
            "openextract._extract._extract_once",
            side_effect=ModelError(
                "permanent",
                status_code=status_code,
                retryable=False,
            ),
        )

        with pytest.raises(ModelError, match="permanent"):
            extract(
                schema=_Person,
                model="openai:gpt-5",
                input_file="ignored",
                max_retries=3,
            )

        assert once.call_count == 1
        sleep_mock.assert_not_called()

    def test_retry_exhausted_raises_last_model_error(self, mocker):
        sleep_mock = mocker.patch("openextract._retry.time.sleep")
        prepare = mocker.patch(
            "openextract._extract._prepare_extraction",
            return_value=nullcontext((MagicMock(), ["prepared"], None)),
        )
        once = mocker.patch(
            "openextract._extract._extract_once",
            side_effect=ModelError("persistent"),
        )

        with pytest.raises(ModelError, match="persistent"):
            extract(
                schema=_Person,
                model="openai:gpt-5",
                input_file="ignored",
                max_retries=2,
            )

        assert once.call_count == 3
        prepare.assert_called_once()
        assert sleep_mock.call_count == 2

    def test_backoff_schedule_uses_exponential_jitter(self, mocker):
        sleep_mock = mocker.patch("openextract._retry.time.sleep")
        mocker.patch(
            "openextract._extract._prepare_extraction",
            return_value=nullcontext((MagicMock(), ["prepared"], None)),
        )
        mocker.patch(
            "openextract._extract._extract_once",
            side_effect=ModelError("nope"),
        )

        with pytest.raises(ModelError):
            extract(
                schema=_Person,
                model="openai:gpt-5",
                input_file="ignored",
                max_retries=3,
                retry_backoff=1.0,
            )

        delays = [call.args[0] for call in sleep_mock.call_args_list]
        assert len(delays) == 3
        assert delays == sorted(delays)
        assert 1.0 <= delays[0] <= 1.25
        assert 2.0 <= delays[1] <= 2.5
        assert 4.0 <= delays[2] <= 5.0

    def test_exponential_backoff_is_bounded(self, mocker):
        sleep_mock = mocker.patch("openextract._retry.time.sleep")
        mocker.patch(
            "openextract._extract._prepare_extraction",
            return_value=nullcontext((MagicMock(), ["prepared"], None)),
        )
        mocker.patch(
            "openextract._extract._extract_once",
            side_effect=ModelError("nope"),
        )

        with pytest.raises(ModelError):
            extract(
                schema=_Person,
                model="openai:gpt-5",
                input_file="ignored",
                max_retries=2,
                retry_backoff=10,
                retry_max_backoff=3,
            )

        assert [call.args[0] for call in sleep_mock.call_args_list] == [3, 3]

    def test_retry_after_takes_precedence_and_is_bounded(self, mocker):
        expected = _Person(name="Grace", age=85)
        mocker.patch(
            "openextract._extract._prepare_extraction",
            return_value=nullcontext((MagicMock(), ["prepared"], None)),
        )
        mocker.patch(
            "openextract._extract._extract_once",
            side_effect=[ModelError("rate limited", retry_after=120), expected],
        )
        sleep_mock = mocker.patch("openextract._retry.time.sleep")

        result = extract(
            schema=_Person,
            model="openai:gpt-5",
            input_file="ignored",
            max_retries=1,
            retry_backoff=0.25,
            retry_max_backoff=15,
        )

        assert result is expected
        sleep_mock.assert_called_once_with(15)

    def test_schema_validation_error_is_not_retried(self, mocker):
        sleep_mock = mocker.patch("openextract._retry.time.sleep")
        prepare = mocker.patch(
            "openextract._extract._prepare_extraction",
            return_value=nullcontext((MagicMock(), ["prepared"], None)),
        )
        once = mocker.patch(
            "openextract._extract._extract_once",
            side_effect=SchemaValidationError("bad shape"),
        )

        with pytest.raises(SchemaValidationError, match="bad shape"):
            extract(
                schema=_Person,
                model="openai:gpt-5",
                input_file="ignored",
                max_retries=3,
            )

        assert once.call_count == 1
        prepare.assert_called_once()
        sleep_mock.assert_not_called()

    def test_non_seekable_stream_and_agent_are_prepared_once_for_retries(self, mocker):
        expected = _Person(name="Grace", age=85)
        stream = MagicMock()
        stream.read.side_effect = [b"hello", b""]
        run_result = MagicMock(output=expected)
        agent_cls, agent = _make_agent_mock(mocker, output=expected)
        agent.run_sync.side_effect = [ModelError("flaky"), run_result]
        mocker.patch("openextract._retry.time.sleep")

        result = extract(
            schema=_Person,
            model="openai:gpt-5",
            input_file=stream,
            media_type="text/plain",
            max_retries=1,
        )

        assert result is expected
        assert stream.read.call_count == 2
        agent_cls.assert_called_once()
        assert agent.run_sync.call_count == 2


class TestExtractWithUsageRetry:
    def test_invalid_retry_options_are_rejected_before_extraction(self, mocker):
        run_extraction = mocker.patch("openextract._extract._run_extraction")

        with pytest.raises(ValueError, match="max_retries"):
            extract_with_usage(
                schema=_Person,
                model="openai:gpt-5",
                input_file="ignored",
                max_retries=True,  # type: ignore[arg-type]
            )

        run_extraction.assert_not_called()

    def test_retries_on_model_error(self, mocker):
        sleep_mock = mocker.patch("openextract._retry.time.sleep")
        prepare = mocker.patch(
            "openextract._extract._prepare_extraction",
            return_value=nullcontext((MagicMock(), ["prepared"], None)),
        )
        expected = _Person(name="Ada", age=36)
        usage = MagicMock(input_tokens=1, output_tokens=2, total_tokens=3)
        run_result = MagicMock(output=expected)
        run_result.usage.return_value = usage
        run_extraction = mocker.patch(
            "openextract._extract._run_extraction",
            side_effect=[ModelError("flaky"), run_result],
        )

        output, got_usage = extract_with_usage(
            schema=_Person,
            model="openai:gpt-5",
            input_file="ignored",
            max_retries=1,
        )

        assert output is expected
        assert got_usage.input_tokens == 1
        assert run_extraction.call_count == 2
        prepare.assert_called_once()
        sleep_mock.assert_called_once()


class TestExtractAsyncRetry:
    async def test_invalid_retry_options_are_rejected_before_agent_build(self, mocker):
        agent_cls = mocker.patch("openextract._agent.Agent")

        with pytest.raises(ValueError, match="retry_backoff"):
            await extract_async(
                schema=_Person,
                model="openai:gpt-5",
                input_file="ignored",
                retry_backoff=-1,
            )

        agent_cls.assert_not_called()

    async def test_retries_on_model_error(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hi")
        sleep_mock = mocker.patch(
            "openextract._retry.asyncio.sleep",
            new_callable=AsyncMock,
        )
        expected = _Person(name="Ada", age=36)
        run_result = MagicMock()
        run_result.output = expected
        agent_cls, agent_instance = _make_async_agent_mock(mocker)
        agent_instance.run = AsyncMock(side_effect=[ModelError("flaky"), run_result])

        result = await extract_async(
            schema=_Person,
            model="openai:gpt-5",
            input_file=str(local),
            max_retries=1,
        )

        assert result is expected
        agent_cls.assert_called_once()
        assert agent_instance.run.await_count == 2
        sleep_mock.assert_awaited_once()

    async def test_non_seekable_stream_is_read_once_for_retries(self, mocker):
        expected = _Person(name="Ada", age=36)
        stream = MagicMock()
        stream.read.side_effect = [b"hello", b""]
        run_result = MagicMock(output=expected)
        agent_cls, agent = _make_async_agent_mock(mocker)
        agent.run = AsyncMock(side_effect=[ModelError("flaky"), run_result])
        mocker.patch("openextract._retry.asyncio.sleep", new_callable=AsyncMock)

        result = await extract_async(
            schema=_Person,
            model="openai:gpt-5",
            input_file=stream,
            media_type="text/plain",
            max_retries=1,
        )

        assert result is expected
        assert stream.read.call_count == 2
        agent_cls.assert_called_once()
        assert agent.run.await_count == 2


# ---------------------------------------------------------------------------
# Async helpers and shared mock builder
# ---------------------------------------------------------------------------


def _make_async_agent_mock(mocker, output=None, run_side_effect=None, usage=None):
    """Patch openextract._agent.Agent and stub the async ``run`` method."""
    agent_instance = MagicMock()
    if run_side_effect is not None:
        agent_instance.run = AsyncMock(side_effect=run_side_effect)
    else:
        run_result = MagicMock()
        run_result.output = output
        if usage is not None:
            run_result.usage.return_value = usage
        agent_instance.run = AsyncMock(return_value=run_result)
    agent_cls = mocker.patch("openextract._agent.Agent", return_value=agent_instance)
    return agent_cls, agent_instance


# ---------------------------------------------------------------------------
# extract_async
# ---------------------------------------------------------------------------


class TestExtractAsync:
    async def test_oversized_input_fails_before_agent_build(self, mocker):
        agent = mocker.patch("openextract._agent.Agent")

        with pytest.raises(InputTooLargeError):
            await extract_async(
                schema=_Person,
                model="openai:gpt-5",
                input_file=b"123456",
                media_type="text/plain",
                max_input_bytes=5,
            )

        agent.assert_not_called()

    async def test_async_usage_helper_enforces_input_limit(self, mocker):
        agent = mocker.patch("openextract._agent.Agent")

        with pytest.raises(InputTooLargeError):
            await extract_with_usage_async(
                schema=_Person,
                model="openai:gpt-5",
                input_file=b"123456",
                media_type="text/plain",
                max_input_bytes=5,
            )

        agent.assert_not_called()

    async def test_local_read_does_not_block_event_loop(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hi")

        def slow_read(file_path, *, max_input_bytes):
            time.sleep(0.05)
            return _read_from_path(file_path, max_input_bytes=max_input_bytes)

        mocker.patch("openextract._media._read_from_path", side_effect=slow_read)
        _make_async_agent_mock(mocker, output=_Person(name="Grace", age=85))
        heartbeat = asyncio.create_task(asyncio.sleep(0.01))

        await extract_async(schema=_Person, model="openai:gpt-5", input_file=str(local))

        assert heartbeat.done()

    async def test_returns_schema_instance_from_local_file(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hi")
        expected = _Person(name="Grace", age=85)
        agent_cls, agent_instance = _make_async_agent_mock(mocker, output=expected)

        result = await extract_async(
            schema=_Person,
            model="openai:gpt-5",
            input_file=str(local),
            instructions="extract",
        )

        assert result is expected
        agent_cls.assert_called_once()
        agent_instance.run.assert_awaited_once()

    async def test_ollama_model_wraps_output_in_native_output(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hi")
        expected = _Person(name="Linus", age=54)
        agent_cls, _ = _make_async_agent_mock(mocker, output=expected)

        result = await extract_async(
            schema=_Person,
            model="ollama:llama3",
            input_file=str(local),
        )

        assert result is expected
        output_type = agent_cls.call_args.kwargs["output_type"]
        assert output_type is not _Person
        assert type(output_type).__name__ == "NativeOutput"

    async def test_http_status_error_is_wrapped(self, mocker):
        response = MagicMock()
        response.status_code = 502
        err = httpx.HTTPStatusError("boom", request=MagicMock(), response=response)
        mocker.patch("openextract._extract._get_media_async", side_effect=err)

        with pytest.raises(UrlFetchError, match="502"):
            await extract_async(schema=_Person, model="openai:gpt-5", input_file="https://x/y")

    async def test_request_error_is_wrapped(self, mocker):
        err = httpx.ConnectError("dns failure")
        mocker.patch("openextract._extract._get_media_async", side_effect=err)

        with pytest.raises(UrlFetchError, match="dns failure"):
            await extract_async(schema=_Person, model="openai:gpt-5", input_file="https://x/y")

    async def test_validation_error_is_wrapped(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hi")
        try:
            _Person(name="x", age="not-an-int")  # type: ignore[arg-type]
        except ValidationError as exc:
            validation_error = exc
        _make_async_agent_mock(mocker, run_side_effect=validation_error)

        with pytest.raises(SchemaValidationError, match=r"age: expected int, got str"):
            await extract_async(schema=_Person, model="openai:gpt-5", input_file=str(local))

    async def test_generic_exception_is_wrapped_as_extraction_error(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hi")
        _make_async_agent_mock(mocker, run_side_effect=RuntimeError("kaboom"))

        with pytest.raises(ExtractionError, match="Extraction failed: kaboom"):
            await extract_async(schema=_Person, model="openai:gpt-5", input_file=str(local))

    async def test_model_keyword_exception_is_not_promoted_to_model_error(self, tmp_path, mocker):
        """Mirrors the sync extract() behavior: substring 'model' no longer triggers ModelError."""
        local = tmp_path / "input.txt"
        local.write_bytes(b"hi")
        _make_async_agent_mock(mocker, run_side_effect=RuntimeError("unknown model identifier"))

        with pytest.raises(ExtractionError) as exc_info:
            await extract_async(schema=_Person, model="openai:gpt-5", input_file=str(local))
        assert not isinstance(exc_info.value, ModelError)

    async def test_missing_media_type_for_bytes_input_raises_type_error(self, mocker):
        """TypeError from _get_media must propagate out of extract_async untouched."""
        _make_async_agent_mock(mocker, output=_Person(name="x", age=1))

        with pytest.raises(TypeError, match="media_type is required"):
            await extract_async(schema=_Person, model="openai:gpt-5", input_file=b"abc")

    async def test_passes_through_existing_extraction_error(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hi")
        original = SchemaValidationError("already mapped")
        _make_async_agent_mock(mocker, run_side_effect=original)

        with pytest.raises(SchemaValidationError) as exc_info:
            await extract_async(schema=_Person, model="openai:gpt-5", input_file=str(local))
        assert exc_info.value is original


# ---------------------------------------------------------------------------
# extract_with_result
# ---------------------------------------------------------------------------


class TestExtractWithResult:
    def test_success_shape(self):
        model = TestModel(custom_output_args={"name": "Ada", "age": 36})
        result = extract_with_result(
            _Person,
            model,
            ExtractionInput(b"Ada Lovelace", media_type="text/plain", name="bio.txt"),
        )
        assert isinstance(result, ExtractionResult)
        assert result.output == _Person(name="Ada", age=36)
        assert result.usage.total_tokens >= 0
        assert result.attempts == 1
        assert result.duration >= 0
        assert result.media_type == "text/plain"
        assert result.source == "bio.txt"
        assert result.warnings == ()
        assert result.citations == ()

    def test_cite_attaches_citations(self):
        model = TestModel(
            custom_output_args={
                "output": {"name": "Ada", "age": 36},
                "citations": [{"field": "name", "quote": "Ada", "page": 1}],
            }
        )
        result = extract_with_result(
            _Person,
            model,
            b"Ada Lovelace, 36",
            media_type="text/plain",
            cite=True,
        )
        assert result.output == _Person(name="Ada", age=36)
        assert result.citations[0].field == "name"
        assert result.citations[0].quote == "Ada"
        assert result.citations[0].page == 1

    async def test_async_twin(self):
        model = TestModel(
            custom_output_args={
                "output": {"name": "Grace", "age": 85},
                "citations": [{"field": "name", "quote": "Grace", "page": 1}],
            }
        )
        result = await extract_with_result_async(
            _Person,
            model,
            ExtractionInput(b"Grace Hopper", media_type="text/plain", name="note.txt"),
            cite=True,
        )
        assert isinstance(result, ExtractionResult)
        assert result.output == _Person(name="Grace", age=85)
        assert result.attempts == 1
        assert result.duration >= 0
        assert result.media_type == "text/plain"
        assert result.source == "note.txt"
        assert result.citations[0].field == "name"
        assert result.citations[0].quote == "Grace"


# ---------------------------------------------------------------------------
# extract_with_usage_async
# ---------------------------------------------------------------------------


class TestExtractWithUsageAsync:
    async def test_invalid_retry_options_are_rejected_before_agent_build(self, mocker):
        agent_cls = mocker.patch("openextract._agent.Agent")

        with pytest.raises(ValueError, match="max_retries"):
            await extract_with_usage_async(
                schema=_Person,
                model="openai:gpt-5",
                input_file="ignored",
                max_retries=-1,
            )

        agent_cls.assert_not_called()

    async def test_returns_output_and_usage_tuple(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        expected = _Person(name="Ada", age=36)
        usage_obj = MagicMock(input_tokens=10, output_tokens=20, total_tokens=30)
        _make_async_agent_mock(mocker, output=expected, usage=usage_obj)

        output, usage = await extract_with_usage_async(
            schema=_Person,
            model="openai:gpt-5",
            input_file=str(local),
            instructions="pull the person",
        )

        assert output is expected
        assert usage == Usage(input_tokens=10, output_tokens=20, total_tokens=30)

    async def test_missing_usage_fields_default_to_zero(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        expected = _Person(name="Linus", age=54)

        class _BareUsage:
            pass

        _make_async_agent_mock(mocker, output=expected, usage=_BareUsage())

        _, usage = await extract_with_usage_async(
            schema=_Person,
            model="openai:gpt-5",
            input_file=str(local),
        )

        assert usage == Usage(input_tokens=0, output_tokens=0, total_tokens=0)

    async def test_propagates_runtime_error_as_extraction_error(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        _make_async_agent_mock(mocker, run_side_effect=RuntimeError("kaboom"))

        with pytest.raises(ExtractionError, match="Extraction failed: kaboom"):
            await extract_with_usage_async(
                schema=_Person, model="openai:gpt-5", input_file=str(local)
            )

    async def test_missing_media_type_for_bytes_raises_type_error(self, mocker):
        _make_async_agent_mock(mocker, output=_Person(name="x", age=1))

        with pytest.raises(TypeError, match="media_type is required"):
            await extract_with_usage_async(schema=_Person, model="openai:gpt-5", input_file=b"abc")


# ---------------------------------------------------------------------------
# _run_with_shared_agent (per-item runner used by the batch path)
# ---------------------------------------------------------------------------


class TestRunWithSharedAgent:
    async def test_runs_agent_and_returns_output(self):
        expected = _Person(name="Ada", age=36)
        agent = MagicMock()
        result = MagicMock()
        result.output = expected
        agent.run = AsyncMock(return_value=result)
        inputs = ["prepared"]

        out = await _run_with_shared_agent(agent, inputs)

        assert out is expected
        agent.run.assert_awaited_once_with(inputs)

    async def test_type_error_propagates_unchanged(self):
        agent = MagicMock()
        agent.run = AsyncMock(side_effect=TypeError("bad prepared input"))

        with pytest.raises(TypeError, match="bad prepared input"):
            await _run_with_shared_agent(agent, ["prepared"])

    async def test_existing_extraction_error_passes_through(self):
        original = SchemaValidationError("already mapped")
        agent = MagicMock()
        agent.run = AsyncMock(side_effect=original)

        with pytest.raises(SchemaValidationError) as exc_info:
            await _run_with_shared_agent(agent, ["prepared"])
        assert exc_info.value is original

    async def test_generic_exception_is_wrapped(self):
        agent = MagicMock()
        agent.run = AsyncMock(side_effect=RuntimeError("kaboom"))

        with pytest.raises(ExtractionError, match="Extraction failed: kaboom"):
            await _run_with_shared_agent(agent, ["prepared"])

    async def test_result_variant_returns_raw_run_result(self):
        expected = _Person(name="Ada", age=36)
        raw = MagicMock()
        raw.output = expected
        agent = MagicMock()
        agent.run = AsyncMock(return_value=raw)

        result = await _run_with_shared_agent_result(agent, ["prepared"])

        assert result is raw
        agent.run.assert_awaited_once_with(["prepared"])


# ---------------------------------------------------------------------------
# extract_many
# ---------------------------------------------------------------------------


def _stub_shared_agent(mocker, side_effect):
    """Stub batch preparation and execution while preserving legacy fake signatures.

    The prepared list deliberately carries the source arguments so focused batch
    tests can inspect them without reading actual files.
    """
    mocker.patch("openextract._batch._build_agent", return_value=MagicMock())

    async def get_media(input_file, client=None, media_type=None, *, max_input_bytes=None):
        return (input_file, media_type, client), "text/plain"

    def resolve_inputs(file_bytes, file_type, style_inputs):
        return list(file_bytes)

    async def run(agent, inputs):
        return await side_effect(agent, inputs[0], inputs[1], inputs[2])

    mocker.patch("openextract._batch._get_media_async", side_effect=get_media)
    mocker.patch("openextract._batch._resolve_run_inputs", side_effect=resolve_inputs)
    mocker.patch(
        "openextract._batch._run_with_shared_agent",
        side_effect=run,
    )


class _FakeRunResult:
    """Minimal raw pydantic-ai result for the rich batch path."""

    def __init__(self, output, usage=None):
        self.output = output
        self._usage = usage or SimpleNamespace(input_tokens=1, output_tokens=2, total_tokens=3)

    def usage(self):
        return self._usage


def _stub_shared_agent_result(mocker, side_effect):
    """Stub the rich batch path; ``side_effect`` returns a raw run result.

    Mirrors :func:`_stub_shared_agent` but patches the raw-result runner so
    ``extract_many_with_results*`` can build ``ExtractionResult`` diagnostics.
    """

    mocker.patch("openextract._batch._build_agent", return_value=MagicMock())

    async def get_media(input_file, client=None, media_type=None, *, max_input_bytes=None):
        return (input_file, media_type, client), "text/plain"

    def resolve_inputs(file_bytes, file_type, style_inputs):
        return list(file_bytes)

    async def run(agent, inputs):
        return await side_effect(agent, inputs[0], inputs[1], inputs[2])

    mocker.patch("openextract._batch._get_media_async", side_effect=get_media)
    mocker.patch("openextract._batch._resolve_run_inputs", side_effect=resolve_inputs)
    mocker.patch(
        "openextract._batch._run_with_shared_agent_result",
        side_effect=run,
    )


class TestExtractMany:
    def test_invalid_max_concurrency_is_rejected_before_agent_build(self, mocker):
        build_mock = mocker.patch("openextract._batch._build_agent")

        with pytest.raises(ValueError, match="max_concurrency"):
            extract_many(
                schema=_Person,
                model="openai:gpt-5",
                input_files=["a.txt"],
                max_concurrency=0,
            )

        build_mock.assert_not_called()

    def test_invalid_retry_options_are_rejected_before_agent_build(self, mocker):
        build_mock = mocker.patch("openextract._batch._build_agent")

        with pytest.raises(ValueError, match="max_retries"):
            extract_many(
                schema=_Person,
                model="openai:gpt-5",
                input_files=["a.txt"],
                max_retries=-1,
            )

        build_mock.assert_not_called()

    def test_invalid_max_input_bytes_is_rejected_before_agent_build(self, mocker):
        build_mock = mocker.patch("openextract._batch._build_agent")

        with pytest.raises(ValueError, match="max_input_bytes"):
            extract_many(
                schema=_Person,
                model="openai:gpt-5",
                input_files=[b"x"],
                media_type="text/plain",
                max_input_bytes=0,
            )

        build_mock.assert_not_called()

    def test_size_errors_are_returned_in_place(self, mocker):
        expected = _Person(name="ok", age=1)
        run_result = MagicMock(output=expected)
        agent = MagicMock()
        agent.run = AsyncMock(return_value=run_result)
        mocker.patch("openextract._batch._build_agent", return_value=agent)

        results = extract_many(
            schema=_Person,
            model="openai:gpt-5",
            input_files=[b"ok", b"too-large"],
            media_type="text/plain",
            max_input_bytes=5,
            return_exceptions=True,
        )

        assert results[0] is expected
        assert isinstance(results[1], InputTooLargeError)
        assert agent.run.await_count == 1

    def test_preserves_input_order(self, tmp_path, mocker):
        files = []
        for i in range(4):
            p = tmp_path / f"f{i}.txt"
            p.write_bytes(b"x")
            files.append(str(p))

        people = [_Person(name=f"n{i}", age=i) for i in range(4)]
        path_to_person = dict(zip(files, people, strict=True))

        async def fake_run(agent, input_file, media_type, client):
            # Tiny await yields control so tasks can interleave.
            await asyncio.sleep(0)
            return path_to_person[input_file]

        _stub_shared_agent(mocker, fake_run)

        results = extract_many(schema=_Person, model="openai:gpt-5", input_files=files)

        assert results == people

    def test_max_concurrency_is_respected(self, tmp_path, mocker):
        files = [str(tmp_path / f"f{i}.txt") for i in range(10)]
        for f in files:
            Path(f).write_bytes(b"x")

        in_flight = 0
        peak = 0
        lock = asyncio.Lock()

        async def fake_run(agent, input_file, media_type, client):
            nonlocal in_flight, peak
            async with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            async with lock:
                in_flight -= 1
            return _Person(name=input_file, age=1)

        _stub_shared_agent(mocker, fake_run)

        results = extract_many(
            schema=_Person,
            model="openai:gpt-5",
            input_files=files,
            max_concurrency=3,
        )

        assert len(results) == 10
        assert peak <= 3
        assert peak >= 1

    def test_fail_fast_propagates_first_error(self, tmp_path, mocker):
        files = [str(tmp_path / f"f{i}.txt") for i in range(3)]

        async def fake_run(agent, input_file, media_type, client):
            if input_file.endswith("f1.txt"):
                raise ModelError("boom on f1")
            await asyncio.sleep(0.01)
            return _Person(name=input_file, age=1)

        _stub_shared_agent(mocker, fake_run)

        with pytest.raises(ModelError, match="boom on f1"):
            extract_many(schema=_Person, model="openai:gpt-5", input_files=files)

    def test_return_exceptions_yields_mixed_list(self, tmp_path, mocker):
        files = [str(tmp_path / f"f{i}.txt") for i in range(3)]

        async def fake_run(agent, input_file, media_type, client):
            if input_file.endswith("f1.txt"):
                raise ModelError("boom on f1")
            return _Person(name=input_file, age=1)

        _stub_shared_agent(mocker, fake_run)

        results = extract_many(
            schema=_Person,
            model="openai:gpt-5",
            input_files=files,
            return_exceptions=True,
        )

        assert isinstance(results[0], _Person)
        assert isinstance(results[1], ModelError)
        assert isinstance(results[2], _Person)

    def test_agent_is_built_once_per_batch(self, tmp_path, mocker):
        """The whole point of the batch path: one Agent for N items."""
        files = [str(tmp_path / f"f{i}.txt") for i in range(8)]

        async def fake_run(agent, input_file, media_type, client):
            return _Person(name=input_file, age=1)

        _stub_shared_agent(mocker, fake_run)
        build_mock = mocker.patch("openextract._batch._build_agent", return_value=MagicMock())

        extract_many(schema=_Person, model="openai:gpt-5", input_files=files)

        assert build_mock.call_count == 1

    def test_empty_input_returns_empty_list_without_building_agent(self, mocker):
        build_mock = mocker.patch("openextract._batch._build_agent")
        assert extract_many(schema=_Person, model="openai:gpt-5", input_files=[]) == []
        build_mock.assert_not_called()

    def test_rejects_call_from_running_event_loop(self, mocker):
        build_mock = mocker.patch("openextract._batch._build_agent")

        async def _call_from_loop():
            with pytest.raises(RuntimeError, match="extract_many_async"):
                extract_many(
                    schema=_Person,
                    model="openai:gpt-5",
                    input_files=["a.txt"],
                )

        asyncio.run(_call_from_loop())
        build_mock.assert_not_called()

    def test_media_type_forwarded_to_runner(self, tmp_path, mocker):
        """media_type kwarg is passed through to every per-item runner call."""
        files = [str(tmp_path / f"f{i}.txt") for i in range(2)]
        for f in files:
            from pathlib import Path

            Path(f).write_bytes(b"x")

        received_types: list[str | None] = []

        async def fake_run(agent, input_file, media_type, client):
            received_types.append(media_type)
            return _Person(name=input_file, age=1)

        _stub_shared_agent(mocker, fake_run)

        extract_many(
            schema=_Person,
            model="openai:gpt-5",
            input_files=files,
            media_type="application/pdf",
        )

        assert all(mt == "application/pdf" for mt in received_types)
        assert len(received_types) == 2

    def test_path_inputs_and_per_item_media_types(self, mocker):
        received: list[str | None] = []

        async def fake_run(agent, source, media_type, client):
            received.append(media_type)
            return _Person(name=str(source), age=1)

        _stub_shared_agent(mocker, fake_run)

        results = extract_many(
            schema=_Person,
            model="openai:gpt-5",
            input_files=[
                Path("/tmp/a.pdf"),
                ExtractionInput(source=b"png", media_type="image/png", name="logo"),
                ExtractionInput(source=b"txt"),
            ],
            media_type="text/plain",
        )

        assert received == ["text/plain", "image/png", "text/plain"]
        assert len(results) == 3


# ---------------------------------------------------------------------------
# extract_many_async
# ---------------------------------------------------------------------------


class TestExtractManyAsync:
    async def test_local_media_reads_overlap(self, tmp_path, mocker):
        files = []
        for index in range(3):
            path = tmp_path / f"f{index}.txt"
            path.write_bytes(b"x")
            files.append(str(path))

        barrier = threading.Barrier(3)

        def concurrent_read(file_path, *, max_input_bytes):
            barrier.wait(timeout=1)
            return _read_from_path(file_path, max_input_bytes=max_input_bytes)

        mocker.patch("openextract._media._read_from_path", side_effect=concurrent_read)
        run_result = MagicMock(output=_Person(name="ok", age=1))
        agent = MagicMock()
        agent.run = AsyncMock(return_value=run_result)
        mocker.patch("openextract._batch._build_agent", return_value=agent)

        results = await extract_many_async(
            schema=_Person,
            model="openai:gpt-5",
            input_files=files,
            max_concurrency=3,
        )

        assert results == [_Person(name="ok", age=1)] * 3

    async def test_invalid_options_are_rejected_before_agent_build(self, mocker):
        build_mock = mocker.patch("openextract._batch._build_agent")

        with pytest.raises(ValueError, match="max_concurrency"):
            await extract_many_async(
                schema=_Person,
                model="openai:gpt-5",
                input_files=["a.txt"],
                max_concurrency=False,  # type: ignore[arg-type]
            )

        build_mock.assert_not_called()

    async def test_prepares_each_item_once_when_model_run_retries(self, mocker):
        expected = _Person(name="Ada", age=36)
        stream = MagicMock()
        stream.read.side_effect = [b"hello", b""]
        mocker.patch("openextract._batch._build_agent", return_value=MagicMock())
        run = mocker.patch(
            "openextract._batch._run_with_shared_agent",
            new_callable=AsyncMock,
            side_effect=[ModelError("flaky"), expected],
        )
        mocker.patch("openextract._retry.asyncio.sleep", new_callable=AsyncMock)

        results = await extract_many_async(
            schema=_Person,
            model="openai:gpt-5",
            input_files=[stream],
            media_type="text/plain",
            max_retries=1,
        )

        assert results == [expected]
        assert stream.read.call_count == 2
        assert run.await_count == 2
        assert run.await_args_list[0].args[1] is run.await_args_list[1].args[1]

    async def test_fail_fast_propagates_first_error(self, tmp_path, mocker):
        files = [str(tmp_path / f"f{i}.txt") for i in range(3)]

        async def fake_run(agent, input_file, media_type, client):
            if input_file.endswith("f0.txt"):
                raise ModelError("boom first")
            await asyncio.sleep(0.01)
            return _Person(name=input_file, age=1)

        _stub_shared_agent(mocker, fake_run)

        with pytest.raises(ModelError, match="boom first"):
            await extract_many_async(
                schema=_Person,
                model="openai:gpt-5",
                input_files=files,
            )

    async def test_return_exceptions_yields_mixed_list(self, tmp_path, mocker):
        files = [str(tmp_path / f"f{i}.txt") for i in range(3)]

        async def fake_run(agent, input_file, media_type, client):
            if input_file.endswith("f1.txt"):
                raise SchemaValidationError("bad schema")
            return _Person(name=input_file, age=1)

        _stub_shared_agent(mocker, fake_run)

        results = await extract_many_async(
            schema=_Person,
            model="openai:gpt-5",
            input_files=files,
            return_exceptions=True,
        )

        assert isinstance(results[0], _Person)
        assert isinstance(results[1], SchemaValidationError)
        assert isinstance(results[2], _Person)

    async def test_preserves_input_order(self, tmp_path, mocker):
        files = [str(tmp_path / f"f{i}.txt") for i in range(5)]
        expected = [_Person(name=f, age=i) for i, f in enumerate(files)]
        mapping = dict(zip(files, expected, strict=True))

        async def fake_run(agent, input_file, media_type, client):
            await asyncio.sleep(0)
            return mapping[input_file]

        _stub_shared_agent(mocker, fake_run)

        results = await extract_many_async(
            schema=_Person,
            model="openai:gpt-5",
            input_files=files,
        )

        assert results == expected


# ---------------------------------------------------------------------------
# iter_extract_many_async
# ---------------------------------------------------------------------------


class TestIterExtractManyAsync:
    async def test_yields_in_completion_order_before_batch_finishes(self, mocker):
        release_first = asyncio.Event()

        async def fake_run(agent, input_file, media_type, client):
            if input_file == "slow":
                await release_first.wait()
            return _Person(name=input_file, age=1)

        _stub_shared_agent(mocker, fake_run)
        results = iter_extract_many_async(
            schema=_Person,
            model="openai:gpt-5",
            input_files=["slow", "fast"],
            max_concurrency=2,
        )

        first = await anext(results)
        assert first == (1, _Person(name="fast", age=1))

        release_first.set()
        assert await anext(results) == (0, _Person(name="slow", age=1))
        with pytest.raises(StopAsyncIteration):
            await anext(results)

    async def test_generator_consumption_and_scheduling_are_bounded(self, mocker):
        consumed: list[int] = []
        started: list[int] = []
        all_started = asyncio.Event()
        release = asyncio.Event()

        def inputs():
            for index in range(100):
                consumed.append(index)
                yield index

        async def fake_run(agent, input_file, media_type, client):
            started.append(input_file)
            if len(started) == 3:
                all_started.set()
            await release.wait()
            return _Person(name=str(input_file), age=1)

        _stub_shared_agent(mocker, fake_run)
        results = iter_extract_many_async(
            schema=_Person,
            model="openai:gpt-5",
            input_files=inputs(),
            max_concurrency=3,
        )
        first_result = asyncio.create_task(anext(results))

        await asyncio.wait_for(all_started.wait(), timeout=1)
        assert consumed == [0, 1, 2]
        assert started == [0, 1, 2]

        release.set()
        await first_result
        await results.aclose()

    async def test_fail_fast_cancels_and_awaits_without_starting_more(self, mocker):
        consumed: list[int] = []
        started: list[int] = []
        both_started = asyncio.Event()
        sibling_cancelled = asyncio.Event()

        def inputs():
            for index in range(10):
                consumed.append(index)
                yield index

        async def fake_run(agent, input_file, media_type, client):
            started.append(input_file)
            if len(started) == 2:
                both_started.set()
            await both_started.wait()
            if input_file == 0:
                raise ModelError("stop")
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise

        _stub_shared_agent(mocker, fake_run)
        results = iter_extract_many_async(
            schema=_Person,
            model="openai:gpt-5",
            input_files=inputs(),
            max_concurrency=2,
        )

        with pytest.raises(ModelError, match="stop"):
            await anext(results)

        assert sibling_cancelled.is_set()
        assert consumed == [0, 1]
        assert started == [0, 1]

    async def test_return_exceptions_continues_stream(self, mocker):
        async def fake_run(agent, input_file, media_type, client):
            if input_file == "bad":
                raise SchemaValidationError("bad schema")
            return _Person(name=input_file, age=1)

        _stub_shared_agent(mocker, fake_run)
        results = [
            item
            async for item in iter_extract_many_async(
                schema=_Person,
                model="openai:gpt-5",
                input_files=["first", "bad", "last"],
                max_concurrency=2,
                return_exceptions=True,
            )
        ]
        by_index = dict(results)

        assert by_index[0] == _Person(name="first", age=1)
        assert isinstance(by_index[1], SchemaValidationError)
        assert by_index[2] == _Person(name="last", age=1)

    async def test_input_size_errors_are_yielded_in_place(self, mocker):
        expected = _Person(name="ok", age=1)
        run_result = MagicMock(output=expected)
        agent = MagicMock()
        agent.run = AsyncMock(return_value=run_result)
        mocker.patch("openextract._batch._build_agent", return_value=agent)

        results = [
            item
            async for item in iter_extract_many_async(
                schema=_Person,
                model="openai:gpt-5",
                input_files=[b"ok", b"too-large"],
                media_type="text/plain",
                max_input_bytes=5,
                return_exceptions=True,
            )
        ]
        by_index = dict(results)

        assert by_index[0] is expected
        assert isinstance(by_index[1], InputTooLargeError)
        assert agent.run.await_count == 1

    async def test_child_cancellation_propagates_and_cleans_up(self, mocker):
        sibling_cancelled = asyncio.Event()

        async def fake_run(agent, input_file, media_type, client):
            if input_file == "cancel":
                raise asyncio.CancelledError
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise

        _stub_shared_agent(mocker, fake_run)
        results = iter_extract_many_async(
            schema=_Person,
            model="openai:gpt-5",
            input_files=["cancel", "sibling"],
            max_concurrency=2,
        )

        with pytest.raises(asyncio.CancelledError):
            await anext(results)

        assert sibling_cancelled.is_set()

    async def test_empty_input_does_not_build_agent(self, mocker):
        build_mock = mocker.patch("openextract._batch._build_agent")

        results = [
            item
            async for item in iter_extract_many_async(
                schema=_Person,
                model="openai:gpt-5",
                input_files=iter(()),
            )
        ]

        assert results == []
        build_mock.assert_not_called()

    def test_invalid_options_fail_when_iterator_is_created(self, mocker):
        build_mock = mocker.patch("openextract._batch._build_agent")

        with pytest.raises(ValueError, match="max_concurrency"):
            iter_extract_many_async(
                schema=_Person,
                model="openai:gpt-5",
                input_files=["a.txt"],
                max_concurrency=-1,
            )

        with pytest.raises(ValueError, match="max_input_bytes"):
            iter_extract_many_async(
                schema=_Person,
                model="openai:gpt-5",
                input_files=["a.txt"],
                max_input_bytes=0,
            )

        build_mock.assert_not_called()

    async def test_per_item_media_types_across_heterogeneous_inputs(self, mocker):
        received: list[str | None] = []

        async def fake_run(agent, source, media_type, client):
            received.append(media_type)
            return _Person(name=str(source), age=1)

        _stub_shared_agent(mocker, fake_run)

        results = [
            item
            async for item in iter_extract_many_async(
                schema=_Person,
                model="openai:gpt-5",
                input_files=[
                    "plain.txt",
                    ExtractionInput(source=b"pdf", media_type="application/pdf", name="invoice"),
                    ExtractionInput(source=b"png", media_type="image/png"),
                ],
                media_type="text/plain",
                return_exceptions=True,
            )
        ]

        assert received == ["text/plain", "application/pdf", "image/png"]
        assert [output for _, output in results][1].name == "b'pdf'"


# ---------------------------------------------------------------------------
# Path / ExtractionInput input contracts
# ---------------------------------------------------------------------------


class TestPathInputs:
    def test_get_media_accepts_path(self, tmp_path):
        local = tmp_path / "hello.txt"
        local.write_bytes(b"hello")

        assert _get_media(local) == (b"hello", "text/plain")

    async def test_get_media_async_accepts_path(self, tmp_path):
        local = tmp_path / "hello.txt"
        local.write_bytes(b"hello")

        assert await _get_media_async(local, MagicMock()) == (b"hello", "text/plain")

    async def test_get_media_async_accepts_pathlike_with_override(self, tmp_path):
        local = tmp_path / "hello.txt"
        local.write_bytes(b"hello")

        result = await _get_media_async(local, MagicMock(), media_type="application/custom")

        assert result == (b"hello", "application/custom")

    def test_get_media_accepts_generic_pathlike(self, tmp_path):
        local = tmp_path / "hello.txt"
        local.write_bytes(b"hello")

        class _PathLike(os.PathLike):
            def __init__(self, path):
                self._path = path

            def __fspath__(self):
                return os.fspath(self._path)

        assert _get_media(_PathLike(local)) == (b"hello", "text/plain")

    def test_extract_accepts_path(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        expected = _Person(name="Ada", age=36)
        _, agent_instance = _make_agent_mock(mocker, output=expected)

        result = extract(schema=_Person, model="openai:gpt-5", input_file=local)

        assert result is expected
        binary = _binary_content_arg(agent_instance)
        assert binary.data == b"hello"

    async def test_extract_async_accepts_path(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        expected = _Person(name="Grace", age=85)
        _make_async_agent_mock(mocker, output=expected)

        result = await extract_async(schema=_Person, model="openai:gpt-5", input_file=local)

        assert result is expected

    def test_extract_with_usage_accepts_path(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        expected = _Person(name="Ada", age=36)
        usage = MagicMock(input_tokens=1, output_tokens=2, total_tokens=3)
        _make_agent_mock(mocker, output=expected, usage=usage)

        output, got_usage = extract_with_usage(
            schema=_Person,
            model="openai:gpt-5",
            input_file=local,
        )

        assert output is expected
        assert got_usage == Usage(1, 2, 3)

    async def test_extract_with_usage_async_accepts_path(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        expected = _Person(name="Grace", age=85)
        usage = SimpleNamespace(input_tokens=1, output_tokens=2, total_tokens=3)
        _make_async_agent_mock(mocker, output=expected, usage=usage)

        output, got_usage = await extract_with_usage_async(
            schema=_Person,
            model="openai:gpt-5",
            input_file=local,
        )

        assert output is expected
        assert got_usage == Usage(1, 2, 3)


class TestExtractionInputContract:
    def test_get_media_unwraps_bytes_input(self):
        assert _get_media(ExtractionInput(b"hello", media_type="text/plain")) == (
            b"hello",
            "text/plain",
        )

    def test_get_media_unwraps_path_input(self, tmp_path):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")

        assert _get_media(ExtractionInput(local)) == (b"hello", "text/plain")

    def test_explicit_media_type_wins_over_extraction_input(self):
        input_ = ExtractionInput(b"hello", media_type="text/plain")

        assert _get_media(input_, media_type="application/pdf") == (
            b"hello",
            "application/pdf",
        )

    async def test_get_media_async_unwraps_input(self):
        result = await _get_media_async(
            ExtractionInput(b"hello", media_type="text/plain"),
            MagicMock(),
        )

        assert result == (b"hello", "text/plain")

    def test_extraction_input_size_limit(self):
        with pytest.raises(InputTooLargeError, match=r"5 bytes.*at least 6 bytes"):
            _get_media(
                ExtractionInput(b"123456", media_type="text/plain"),
                max_input_bytes=5,
            )

    def test_single_extract_accepts_extraction_input(self, mocker):
        expected = _Person(name="Ada", age=36)
        _, agent_instance = _make_agent_mock(mocker, output=expected)

        result = extract(
            schema=_Person,
            model="openai:gpt-5",
            input_file=ExtractionInput(b"payload", media_type="application/pdf"),
        )

        assert result is expected
        binary = _binary_content_arg(agent_instance)
        assert binary.data == b"payload"
        assert binary.media_type == "application/pdf"

    def test_single_extract_explicit_override_wins(self, mocker):
        expected = _Person(name="Ada", age=36)
        _, agent_instance = _make_agent_mock(mocker, output=expected)

        extract(
            schema=_Person,
            model="openai:gpt-5",
            input_file=ExtractionInput(b"payload", media_type="application/pdf"),
            media_type="image/png",
        )

        binary = _binary_content_arg(agent_instance)
        assert binary.media_type == "image/png"

    def test_extraction_input_is_frozen(self):
        input_ = ExtractionInput(b"x")
        with pytest.raises(FrozenInstanceError):
            input_.media_type = "text/plain"  # type: ignore[misc]


class TestBatchItemResolution:
    def test_raw_item_uses_global_media_type(self):
        source, media_type, name = _resolve_item("a.pdf", "application/pdf")

        assert (source, media_type, name) == ("a.pdf", "application/pdf", None)

    def test_extraction_input_media_type_wins(self):
        item = ExtractionInput(b"x", media_type="image/png", name="scan")

        source, media_type, name = _resolve_item(item, "application/pdf")

        assert source == b"x"
        assert media_type == "image/png"
        assert name == "scan"

    def test_extraction_input_falls_back_to_global(self):
        item = ExtractionInput(b"x")

        _, media_type, _ = _resolve_item(item, "application/pdf")

        assert media_type == "application/pdf"

    def test_item_source_label_prefers_name(self):
        label = _item_source_label("https://user:secret@example.com/f?q=1", "invoice")

        assert label == "invoice"

    def test_item_source_label_sanitizes_url(self):
        label = _item_source_label("https://user:secret@example.com/f.pdf?token=abc", None)

        assert label == "URL https://example.com/f.pdf"
        assert "secret" not in label
        assert "token" not in label

    def test_item_source_label_path(self):
        assert _item_source_label(Path("/tmp/x.pdf"), None) == "path 'x.pdf'"

    def test_item_source_label_pathlike(self, tmp_path):
        assert _item_source_label(tmp_path / "x.pdf", None) == "path 'x.pdf'"

    def test_item_source_label_bytes_is_none(self):
        assert _item_source_label(b"x", None) is None

    def test_item_source_label_stream_is_none(self):
        assert _item_source_label(io.BytesIO(b"x"), None) is None

    def test_model_identifier_routes_string(self):
        assert _model_identifier("openai:gpt-5", MagicMock()) == "openai-responses:gpt-5"

    def test_model_identifier_uses_model_name(self):
        model = MagicMock()
        model.model_name = "custom-model"

        assert _model_identifier(model, MagicMock()) == "custom-model"

    def test_model_identifier_falls_back_to_agent_model(self):
        agent = MagicMock()
        agent.model.model_name = "agent-model"

        assert _model_identifier(MagicMock(), agent) == "agent-model"

    def test_model_identifier_unknown_returns_none(self):
        assert _model_identifier(MagicMock(), MagicMock()) is None


class TestTotalUsage:
    def test_sums_usage_across_results(self):
        results = [
            ExtractionResult(_Person(name="a", age=1), Usage(1, 2, 3), 1, 0.1, "m", None, None),
            ExtractionResult(_Person(name="b", age=2), Usage(4, 5, 9), 2, 0.2, "m", None, None),
        ]

        assert total_usage(results) == Usage(5, 7, 12)

    def test_empty_results_sum_to_zero(self):
        assert total_usage([]) == Usage(0, 0, 0)

    def test_extraction_result_records_elapsed_time(self, mocker):
        mocker.patch("openextract._types.time.perf_counter", return_value=10.5)
        result = _extraction_result(
            _Person(name="a", age=1),
            Usage(1, 2, 3),
            attempts=2,
            started=10.0,
            model="m",
            media_type="text/plain",
            source="doc",
        )
        assert result.duration == pytest.approx(0.5)
        assert result.citations == ()
        assert result.warnings == ()


# ---------------------------------------------------------------------------
# extract_many_with_results
# ---------------------------------------------------------------------------


class TestExtractManyWithResults:
    def test_invalid_options_rejected_before_agent_build(self, mocker):
        build_mock = mocker.patch("openextract._batch._build_agent")

        with pytest.raises(ValueError, match="max_concurrency"):
            extract_many_with_results(
                schema=_Person,
                model="openai:gpt-5",
                input_files=["a.txt"],
                max_concurrency=0,
            )

        build_mock.assert_not_called()

    def test_rejects_call_from_running_event_loop(self, mocker):
        build_mock = mocker.patch("openextract._batch._build_agent")

        async def _call_from_loop():
            with pytest.raises(RuntimeError, match="extract_many_with_results_async"):
                extract_many_with_results(
                    schema=_Person,
                    model="openai:gpt-5",
                    input_files=["a.txt"],
                )

        asyncio.run(_call_from_loop())
        build_mock.assert_not_called()

    def test_returns_diagnostics_per_item(self, tmp_path, mocker):
        files = [str(tmp_path / f"f{i}.txt") for i in range(2)]

        async def fake_run(agent, source, media_type, client):
            return _FakeRunResult(_Person(name=source, age=1))

        _stub_shared_agent_result(mocker, fake_run)

        results = extract_many_with_results(
            schema=_Person,
            model="openai:gpt-5",
            input_files=files,
        )

        assert len(results) == 2
        for result in results:
            assert isinstance(result, ExtractionResult)
            assert isinstance(result.output, _Person)
            assert result.usage == Usage(1, 2, 3)
            assert result.attempts == 1
            assert result.duration >= 0
            assert result.model == "openai-responses:gpt-5"
            assert result.media_type is None
            assert result.source is not None
            assert result.warnings == ()

    def test_partial_failure_returns_exceptions_in_place(self, tmp_path, mocker):
        files = [str(tmp_path / f"f{i}.txt") for i in range(2)]

        async def fake_run(agent, source, media_type, client):
            if source.endswith("f1.txt"):
                raise ModelError("boom on f1")
            return _FakeRunResult(_Person(name=source, age=1))

        _stub_shared_agent_result(mocker, fake_run)

        results = extract_many_with_results(
            schema=_Person,
            model="openai:gpt-5",
            input_files=files,
            return_exceptions=True,
        )

        assert isinstance(results[0], ExtractionResult)
        assert isinstance(results[1], ModelError)

    def test_fail_fast_propagates_first_error(self, tmp_path, mocker):
        files = [str(tmp_path / f"f{i}.txt") for i in range(3)]

        async def fake_run(agent, source, media_type, client):
            if source.endswith("f1.txt"):
                raise ModelError("boom on f1")
            return _FakeRunResult(_Person(name=source, age=1))

        _stub_shared_agent_result(mocker, fake_run)

        with pytest.raises(ModelError, match="boom on f1"):
            extract_many_with_results(
                schema=_Person,
                model="openai:gpt-5",
                input_files=files,
            )

    def test_retries_record_attempts_and_per_item_metadata(self, mocker):
        flaky_attempts = 0

        async def fake_run(agent, source, media_type, client):
            nonlocal flaky_attempts
            if source == b"flaky":
                flaky_attempts += 1
                if flaky_attempts == 1:
                    raise ModelError("transient", retryable=True)
            return _FakeRunResult(_Person(name=str(source), age=1))

        _stub_shared_agent_result(mocker, fake_run)
        mocker.patch("openextract._retry.asyncio.sleep", new_callable=AsyncMock)

        results = extract_many_with_results(
            schema=_Person,
            model="openai:gpt-5",
            input_files=[
                ExtractionInput(source=b"flaky", media_type="application/pdf", name="invoice-a"),
                ExtractionInput(source=b"ok", media_type="text/plain"),
            ],
            max_retries=1,
        )

        assert isinstance(results[0], ExtractionResult)
        assert results[0].attempts == 2
        assert results[0].media_type == "application/pdf"
        assert results[0].source == "invoice-a"
        assert isinstance(results[1], ExtractionResult)
        assert results[1].attempts == 1
        assert results[1].media_type == "text/plain"

    def test_global_media_type_fallback_and_source_sanitization(self, mocker):
        async def fake_run(agent, source, media_type, client):
            return _FakeRunResult(_Person(name=str(source), age=1))

        _stub_shared_agent_result(mocker, fake_run)

        results = extract_many_with_results(
            schema=_Person,
            model="openai:gpt-5",
            input_files=["https://user:secret@example.com/f.pdf?token=abc"],
            media_type="application/pdf",
        )

        assert results[0].media_type == "application/pdf"
        assert results[0].source == "URL https://example.com/f.pdf"
        assert "secret" not in (results[0].source or "")

    def test_total_usage_aggregates_across_batch(self, mocker):
        async def fake_run(agent, source, media_type, client):
            usage = SimpleNamespace(input_tokens=10, output_tokens=20, total_tokens=30)
            return _FakeRunResult(_Person(name=str(source), age=1), usage=usage)

        _stub_shared_agent_result(mocker, fake_run)

        results = extract_many_with_results(
            schema=_Person,
            model="openai:gpt-5",
            input_files=["a", "b"],
        )

        assert total_usage(results) == Usage(20, 40, 60)


class TestExtractManyWithResultsAsync:
    async def test_returns_rich_results_in_input_order(self, tmp_path, mocker):
        files = [str(tmp_path / f"f{i}.txt") for i in range(3)]

        async def fake_run(agent, source, media_type, client):
            await asyncio.sleep(0)
            return _FakeRunResult(_Person(name=source, age=1))

        _stub_shared_agent_result(mocker, fake_run)

        results = await extract_many_with_results_async(
            schema=_Person,
            model="openai:gpt-5",
            input_files=files,
        )

        assert [result.output.name for result in results] == files
        assert all(isinstance(result, ExtractionResult) for result in results)

    async def test_empty_input_returns_empty_list_without_building_agent(self, mocker):
        build_mock = mocker.patch("openextract._batch._build_agent")

        results = await extract_many_with_results_async(
            schema=_Person,
            model="openai:gpt-5",
            input_files=[],
        )

        assert results == []
        build_mock.assert_not_called()

    async def test_heterogeneous_per_item_media_types(self, mocker):
        received: list[str | None] = []

        async def fake_run(agent, source, media_type, client):
            received.append(media_type)
            return _FakeRunResult(_Person(name=str(source), age=1))

        _stub_shared_agent_result(mocker, fake_run)

        results = await extract_many_with_results_async(
            schema=_Person,
            model="openai:gpt-5",
            input_files=[
                ExtractionInput(source=b"pdf", media_type="application/pdf"),
                ExtractionInput(source=b"png", media_type="image/png"),
                b"raw",
            ],
            media_type="text/plain",
        )

        assert received == ["application/pdf", "image/png", "text/plain"]
        assert [result.media_type for result in results] == received

    async def test_size_errors_are_returned_in_place(self, mocker):
        expected = _Person(name="ok", age=1)
        run_result = _FakeRunResult(expected)
        agent = MagicMock()
        agent.run = AsyncMock(return_value=run_result)
        mocker.patch("openextract._batch._build_agent", return_value=agent)

        results = await extract_many_with_results_async(
            schema=_Person,
            model="openai:gpt-5",
            input_files=[ExtractionInput(b"ok", media_type="text/plain"), b"too-large"],
            media_type="text/plain",
            max_input_bytes=5,
            return_exceptions=True,
        )

        assert isinstance(results[0], ExtractionResult)
        assert results[0].output is expected
        assert isinstance(results[1], InputTooLargeError)
        assert agent.run.await_count == 1
