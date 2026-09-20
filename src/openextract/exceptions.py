"""Exceptions for openextract."""


class ExtractionError(Exception):
    """Base exception for extraction errors."""


class ModelError(ExtractionError):
    """Error communicating with the model API.

    The optional metadata is populated for provider exceptions when available.
    ``retryable`` defaults to ``True`` so manually raised ``ModelError`` values
    retain their historical retry behavior.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
        retryable: bool | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        if retryable is None:
            retryable = (
                status_code is None
                or status_code in {408, 409, 425, 429}
                or 500 <= status_code <= 599
            )
        self.retryable = retryable
        self.retry_after = retry_after


class ProviderNotInstalledError(ExtractionError):
    """A required optional extra is not installed.

    Raised when a model is requested whose provider extra was not installed,
    e.g. calling ``extract(..., model="openai:gpt-4o")`` without first running
    ``pip install 'openextract[openai]'``, or when ``style='search'`` /
    ``style='code'`` is used without ``pydantic-ai-harness``.
    """


class RemoteAgentError(ExtractionError):
    """A remote extraction agent could not be reached or returned a failure.

    ``retryable`` defaults to whether ``status_code`` is transient; transport
    failures (no response at all) are raised as retryable explicitly.
    """

    def __init__(
        self,
        message: str,
        *,
        url: str | None = None,
        status_code: int | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.url = url
        self.status_code = status_code
        if retryable is None:
            retryable = status_code is not None and (
                status_code in {408, 409, 425, 429} or 500 <= status_code <= 599
            )
        self.retryable = retryable


class InputFileError(ExtractionError):
    """Error opening or reading a local path or file-like input."""


class InputTooLargeError(ExtractionError):
    """Input media exceeded the configured byte limit."""


class SchemaValidationError(ExtractionError):
    """Model output did not match the expected schema.

    When raised from a Pydantic ``ValidationError``, the message lists each
    failing field path and expected type (for example
    ``lines[0].qty: expected int, got str``). ``errors`` holds the same
    details as ``({"field", "expected", "received"}, ...)``.
    """

    def __init__(
        self,
        message: str,
        *,
        errors: tuple[dict[str, str], ...] = (),
    ) -> None:
        super().__init__(message)
        self.errors = errors


class UrlFetchError(ExtractionError):
    """Error fetching the URL content."""
