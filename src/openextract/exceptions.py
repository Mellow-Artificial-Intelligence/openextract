"""Exceptions raised by openextract. Underlying errors are chained when available."""


class ExtractionError(Exception):
    """Base class for every openextract error, also raised when a URL cannot be fetched."""


class ModelError(ExtractionError):
    """A model request or generation failed, with provider details when available."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | int | None = None,
        provider: str | None = None,
        finish_reason: str | None = None,
        native_finish_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.provider = provider
        self.finish_reason = finish_reason
        self.native_finish_reason = native_finish_reason


class SchemaValidationError(ExtractionError):
    """The model's output did not match the schema."""


class ProviderNotInstalledError(ExtractionError):
    """Deprecated compatibility export; HTTP adapters do not require provider SDKs."""
