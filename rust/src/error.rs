//! Error types returned by openextract.

use std::fmt;

/// Convenience result alias for openextract operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Provider error code, which may be a string or an integer depending on the API.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum ErrorCode {
    String(String),
    Int(i64),
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String(s) => f.write_str(s),
            Self::Int(n) => write!(f, "{n}"),
        }
    }
}

/// Details of a failed model request or provider-reported generation error.
#[derive(Debug)]
pub struct ModelError {
    pub message: String,
    pub status_code: Option<u16>,
    pub code: Option<ErrorCode>,
    pub provider: Option<String>,
    pub finish_reason: Option<String>,
    pub native_finish_reason: Option<String>,
    pub(crate) source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl ModelError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            status_code: None,
            code: None,
            provider: None,
            finish_reason: None,
            native_finish_reason: None,
            source: None,
        }
    }
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_deref()
            .map(|e| e as &(dyn std::error::Error + 'static))
    }
}

/// Errors produced by openextract.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Operational failure such as missing credentials, missing Bedrock region, or unreachable URL.
    #[error("{message}")]
    Extraction {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// A model request failed or the provider reported an HTTP or generation error.
    #[error(transparent)]
    Model(#[from] Box<ModelError>),

    /// The model output did not match the schema, was refused or truncated, or was malformed.
    #[error("{message}")]
    SchemaValidation {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Invalid configuration, unsupported media type, unsupported schema, or invalid CLI input.
    #[error("{0}")]
    InvalidConfig(String),

    /// Local filesystem I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl Error {
    pub(crate) fn extraction(message: impl Into<String>) -> Self {
        Self::Extraction {
            message: message.into(),
            source: None,
        }
    }

    pub(crate) fn extraction_with_source(
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Extraction {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    pub(crate) fn model(err: ModelError) -> Self {
        Self::Model(Box::new(err))
    }

    pub(crate) fn schema_validation(message: impl Into<String>) -> Self {
        Self::SchemaValidation {
            message: message.into(),
            source: None,
        }
    }

    pub(crate) fn schema_validation_with_source(
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::SchemaValidation {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    /// Conventional process exit code for CLI consumers.
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::InvalidConfig(_) => 2,
            Self::Extraction { .. } | Self::Io(_) => 3,
            Self::Model(_) => 4,
            Self::SchemaValidation { .. } => 5,
        }
    }
}
