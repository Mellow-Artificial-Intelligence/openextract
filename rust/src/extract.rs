//! Schema + model + source -> a validated instance of that schema.

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

use crate::content::{PROMPT, Source, load};
use crate::diagnostics::{invalid_response_message, validation_details, validation_message};
use crate::error::{Error, Result};
use crate::models::{Api, Model, ResolvedModel, resolve};
use crate::openrouter_extractions::{ExtractionResult, ParsedExtraction};
use crate::schemas::{ValidationFailure, schema_for_type, validate_instance};
use crate::{decisions, openrouter_extractions, providers, transport};

/// Optional per-call extraction settings.
#[derive(Debug, Default, Clone, Copy)]
pub struct ExtractOptions<'a> {
    /// Optional guidance for this extraction call.
    pub instructions: Option<&'a str>,
    /// Override media type detection (e.g. `"application/pdf"`, `"audio/mpeg"`).
    pub media_type: Option<&'a str>,
    /// Caller-owned `reqwest::Client` for model requests (source URLs are always fetched separately).
    pub client: Option<&'a reqwest::Client>,
}

impl<'a> ExtractOptions<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_instructions(mut self, instructions: &'a str) -> Self {
        self.instructions = Some(instructions);
        self
    }

    pub fn with_media_type(mut self, media_type: &'a str) -> Self {
        self.media_type = Some(media_type);
        self
    }

    pub fn with_client(mut self, client: &'a reqwest::Client) -> Self {
        self.client = Some(client);
        self
    }
}

/// Extract a strongly-typed Rust value implementing `JsonSchema + DeserializeOwned`.
pub async fn extract<T>(
    model: impl Into<Model>,
    source: impl Into<Source>,
    instructions: Option<&str>,
) -> Result<T>
where
    T: DeserializeOwned + JsonSchema,
{
    extract_with_options(
        model,
        source,
        ExtractOptions {
            instructions,
            ..Default::default()
        },
    )
    .await
}

/// Extract a strongly-typed Rust value with custom options (`instructions`, `media_type`, `client`).
pub async fn extract_with_options<T>(
    model: impl Into<Model>,
    source: impl Into<Source>,
    options: ExtractOptions<'_>,
) -> Result<T>
where
    T: DeserializeOwned + JsonSchema,
{
    let schema = schema_for_type::<T>()?;
    let resolved = resolve(model)?;
    run_typed(&schema, &resolved, source.into(), options, false)
        .await
        .map(|(data, _, _)| data)
}

/// Extract a strongly-typed Rust value plus OpenRouter Extractions answers and metadata.
pub async fn extract_with_metadata<T>(
    model: impl Into<Model>,
    source: impl Into<Source>,
    options: ExtractOptions<'_>,
) -> Result<ExtractionResult<T>>
where
    T: DeserializeOwned + JsonSchema,
{
    let schema = schema_for_type::<T>()?;
    let resolved = resolve(model)?;
    let (data, answers, metadata) =
        run_typed(&schema, &resolved, source.into(), options, true).await?;
    Ok(ExtractionResult {
        data,
        answers,
        metadata,
    })
}

/// Extract a JSON value validated against a dynamic JSON Schema.
pub async fn extract_value(
    schema: &Value,
    model: impl Into<Model>,
    source: impl Into<Source>,
    options: ExtractOptions<'_>,
) -> Result<Value> {
    let resolved = resolve(model)?;
    run_value(schema, &resolved, source.into(), options, false)
        .await
        .map(|(data, _, _)| data)
}

/// Extract a JSON value validated against a dynamic JSON Schema plus OpenRouter Extractions metadata.
pub async fn extract_value_with_metadata(
    schema: &Value,
    model: impl Into<Model>,
    source: impl Into<Source>,
    options: ExtractOptions<'_>,
) -> Result<ExtractionResult<Value>> {
    let resolved = resolve(model)?;
    let (data, answers, metadata) =
        run_value(schema, &resolved, source.into(), options, true).await?;
    Ok(ExtractionResult {
        data,
        answers,
        metadata,
    })
}

async fn run_typed<T: DeserializeOwned>(
    schema: &Value,
    resolved: &ResolvedModel,
    source: Source,
    options: ExtractOptions<'_>,
    include_metadata: bool,
) -> Result<(T, Map<String, Value>, Map<String, Value>)> {
    run_extraction(schema, resolved, source, options, include_metadata, |val| {
        validate_instance(schema, &val)?;
        serde_json::from_value::<T>(val).map_err(|_| {
            vec![ValidationFailure {
                path: "<root>".to_string(),
                message: "Value did not satisfy target type deserialization.".to_string(),
                kind: "deserialization_error".to_string(),
            }]
        })
    })
    .await
}

async fn run_value(
    schema: &Value,
    resolved: &ResolvedModel,
    source: Source,
    options: ExtractOptions<'_>,
    include_metadata: bool,
) -> Result<(Value, Map<String, Value>, Map<String, Value>)> {
    run_extraction(schema, resolved, source, options, include_metadata, |val| {
        validate_instance(schema, &val)?;
        Ok(val)
    })
    .await
}

async fn run_extraction<T, F>(
    schema: &Value,
    resolved: &ResolvedModel,
    source: Source,
    options: ExtractOptions<'_>,
    include_metadata: bool,
    validate_and_convert: F,
) -> Result<(T, Map<String, Value>, Map<String, Value>)>
where
    F: Fn(Value) -> std::result::Result<T, Vec<ValidationFailure>>,
{
    if include_metadata && resolved.api != Api::OpenRouterExtractions {
        return Err(Error::InvalidConfig(
            "include_metadata=True requires the OpenRouter Extractions API.".to_string(),
        ));
    }

    let content = load(source, options.media_type).await?;
    let owned_client;
    let client = match options.client {
        Some(c) => c,
        None => {
            owned_client = transport::default_client()?;
            &owned_client
        }
    };

    let mut guidance: Option<String> = options.instructions.map(str::to_string);
    let mut attempt: usize = 0;

    loop {
        attempt += 1;
        let (path, body) = build_request(schema, resolved, &content, guidance.as_deref())?;
        let payload = transport::post(resolved, &path, &body, client).await?;

        let parsed = parse_payload(schema, resolved, &payload, include_metadata);
        let ParsedExtraction {
            value: candidate,
            answers,
            metadata,
        } = match parsed {
            Ok(parts) => parts,
            Err(detail) => {
                let context = if resolved.api.is_decisions() {
                    "decision answers"
                } else {
                    "model response"
                };
                return Err(Error::schema_validation(invalid_response_message(
                    &detail, context,
                )));
            }
        };

        match validate_and_convert(candidate) {
            Ok(data) => return Ok((data, answers, metadata)),
            Err(failures) => {
                let retries = if resolved.api.is_decisions() {
                    0
                } else {
                    resolved.config.validation_retries as usize
                };
                if attempt > retries {
                    return Err(Error::schema_validation(validation_message(
                        &failures, attempt,
                    )));
                }
                let base = options.instructions.unwrap_or(PROMPT);
                guidance = Some(format!(
                    "{base}\nCorrect these output validation errors:\n{}",
                    validation_details(&failures)
                ));
            }
        }
    }
}

fn build_request(
    schema: &Value,
    resolved: &ResolvedModel,
    content: &crate::content::Content,
    instructions: Option<&str>,
) -> Result<(String, Value)> {
    if resolved.api == Api::OpenRouterExtractions {
        openrouter_extractions::request(schema, resolved, content, instructions)
    } else if resolved.api.is_decisions() {
        decisions::request(schema, resolved, content, instructions)
    } else {
        providers::request(schema, resolved, content, instructions)
    }
}

fn parse_payload(
    schema: &Value,
    resolved: &ResolvedModel,
    payload: &Map<String, Value>,
    include_metadata: bool,
) -> std::result::Result<ParsedExtraction, String> {
    if resolved.api == Api::OpenRouterExtractions {
        openrouter_extractions::read_response(schema, payload, include_metadata)
    } else if resolved.api.is_decisions() {
        let value = decisions::parse_answers(schema, resolved, payload)?;
        Ok(ParsedExtraction {
            value,
            answers: Map::new(),
            metadata: Map::new(),
        })
    } else {
        let value = providers::read_output(resolved, payload)?;
        Ok(ParsedExtraction {
            value,
            answers: Map::new(),
            metadata: Map::new(),
        })
    }
}
