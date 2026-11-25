//! Extract structured data from documents, images, audio, and video using LLMs.
//!
//! Every provider is called over plain HTTP with no provider SDKs.
//!
//! # Example
//!
//! ```no_run
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//! use openextract::{extract, Source};
//!
//! #[derive(Debug, Deserialize, JsonSchema)]
//! struct Invoice {
//!     /// Seller, not buyer.
//!     vendor: String,
//!     /// Final amount due, including tax.
//!     total: f64,
//!     /// Three-letter currency code, or null if unclear.
//!     currency: Option<String>,
//! }
//!
//! # async fn run() -> openextract::Result<()> {
//! let source = "Seller: Acme Paper Co.\nTotal due: 120.00 USD\n";
//! let invoice: Invoice = extract("openai:gpt-6.1-sol", source, None).await?;
//! println!("{}: {}", invoice.vendor, invoice.total);
//! # Ok(())
//! # }
//! ```

pub mod content;
pub mod diagnostics;
pub mod error;
pub mod extract;
pub mod field_spec;
pub mod models;
pub mod openrouter_extractions;
pub mod schemas;
pub mod transport;

mod decisions;
mod providers;

pub use content::{Content, PROMPT, Source};
pub use error::{Error, ErrorCode, ModelError, Result};
pub use extract::{
    ExtractOptions, extract, extract_value, extract_value_with_metadata, extract_with_metadata,
    extract_with_options,
};
pub use field_spec::fields_to_schema;
pub use models::{
    Api, Model, OutputMode, ProviderInfo, ResolvedModel, builtin_providers, resolve,
    resolve_with_env,
};
pub use openrouter_extractions::{ExtractionResult, OpenRouterField};
pub use transport::default_client;
