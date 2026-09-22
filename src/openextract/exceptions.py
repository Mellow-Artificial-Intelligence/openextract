"""Exceptions raised by openextract. The underlying error is always chained as ``__cause__``."""


class ExtractionError(Exception):
    """Base class for every openextract error, also raised when a URL cannot be fetched."""


class ModelError(ExtractionError):
    """The model provider returned an error."""


class SchemaValidationError(ExtractionError):
    """The model's output did not match the schema."""


class ProviderNotInstalledError(ExtractionError):
    """The SDK for the requested model provider is not installed."""
