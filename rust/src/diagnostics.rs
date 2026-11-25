//! Actionable error messages that never dump response bodies, model output, or credentials.

use serde_json::{Map, Value};

use crate::error::{Error, ErrorCode, ModelError};
use crate::schemas::ValidationFailure;

const MAX_VALIDATION_DETAILS: usize = 5;
const TIMED_OUT: &str = "The provider or gateway timed out. Retry later or reduce \
                         the document size or output token budget.";

pub fn validation_details(errors: &[ValidationFailure]) -> String {
    let mut lines = Vec::new();
    for detail in errors.iter().take(MAX_VALIDATION_DETAILS) {
        let path = if detail.path.is_empty() {
            "<root>"
        } else {
            &detail.path
        };
        lines.push(format!("- {path}: {} ({})", detail.message, detail.kind));
    }
    if errors.len() > MAX_VALIDATION_DETAILS {
        let omitted = errors.len() - MAX_VALIDATION_DETAILS;
        lines.push(format!("- {omitted} more validation errors omitted."));
    }
    lines.join("\n")
}

pub fn validation_message(errors: &[ValidationFailure], attempts: usize) -> String {
    format!(
        "The model output did not match the schema after {}.\n\
         {}\n\
         Next step: Check the listed field types, constraints, and validators. \
         Clarify field descriptions or extraction instructions before retrying.",
        count(attempts, "validation attempt"),
        validation_details(errors),
    )
}

pub fn invalid_response_message(detail: &str, context: &str) -> String {
    format!(
        "Invalid {context}: {detail}\n\
         Next step: Check that Model(api=...) and Model(base_url=...) match the provider. \
         Use a model that supports the requested schema and input, and check output token limits."
    )
}

pub fn request_error(err: reqwest::Error, attempts: usize) -> Error {
    let (kind, hint) = if err.is_timeout() {
        (
            "TimeoutException",
            "Check provider availability or increase Model(timeout=...) for slow requests.",
        )
    } else if err.is_decode() {
        (
            "DecodingError",
            "Check the provider or proxy's response encoding and compression settings.",
        )
    } else if err.is_builder() {
        (
            "LocalProtocolError",
            "Check Model(headers=...) and client authentication for invalid header values.",
        )
    } else if err.is_connect() {
        (
            "ConnectError",
            "Check Model(base_url=...), network connectivity, proxy settings, and TLS certificates.",
        )
    } else {
        (
            "RequestError",
            "Check Model(base_url=...), network connectivity, proxy settings, and TLS certificates.",
        )
    };
    let mut model_err = ModelError::new(format!(
        "The model request failed ({kind}) after {}.\nNext step: {hint}",
        count(attempts, "HTTP attempt"),
    ));
    model_err.source = Some(Box::new(err));
    Error::model(model_err)
}

struct ErrorDetails {
    provider: Option<String>,
    code: Option<ErrorCode>,
    finish_reason: Option<String>,
    native_finish_reason: Option<String>,
    message: Option<String>,
}

/// Whether a 2xx response body carries a provider or generation error.
pub fn has_error_envelope(payload: &Map<String, Value>) -> bool {
    if payload.get("error").is_some_and(|v| !v.is_null()) {
        return true;
    }
    let choice = first_choice(payload);
    choice.get("error").is_some_and(|v| !v.is_null())
        || choice.get("finish_reason").and_then(Value::as_str) == Some("error")
}

/// Format an HTTP failure or an error envelope with redacted credentials.
pub fn provider_error(
    payload: &Map<String, Value>,
    status_code: u16,
    secrets: &[String],
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
) -> Error {
    let raw = extract_error_details(payload);
    let next = next_step(
        status_code,
        raw.code.as_ref(),
        raw.native_finish_reason.as_deref(),
    );
    let mut ordered_secrets: Vec<&str> = secrets
        .iter()
        .map(String::as_str)
        .filter(|s| !s.is_empty())
        .collect();
    ordered_secrets.sort_unstable();
    ordered_secrets.dedup();
    ordered_secrets.sort_by_key(|b| std::cmp::Reverse(b.len()));

    let provider = redact(raw.provider.as_deref(), &ordered_secrets);
    let code = raw.code.map(|c| match c {
        ErrorCode::String(s) => ErrorCode::String(redact(Some(&s), &ordered_secrets).unwrap_or(s)),
        ErrorCode::Int(n) => ErrorCode::Int(n),
    });
    let finish_reason = redact(raw.finish_reason.as_deref(), &ordered_secrets);
    let native_finish_reason = redact(raw.native_finish_reason.as_deref(), &ordered_secrets);
    let details = redact(raw.message.as_deref(), &ordered_secrets);

    let heading = if (200..300).contains(&status_code) {
        "Model provider failed.".to_string()
    } else {
        format!("Model provider returned HTTP {status_code}.")
    };

    let mut lines = vec![heading];
    if let Some(ref p) = provider {
        lines.push(format!("Provider: {p}"));
    }
    if let Some(ref c) = code {
        lines.push(format!("Code: {c}"));
    }
    if let Some(ref fr) = finish_reason {
        lines.push(format!("Finish reason: {fr}"));
    }
    if let Some(ref nfr) = native_finish_reason {
        lines.push(format!("Native finish reason: {nfr}"));
    }
    if let Some(ref d) = details {
        lines.push(format!("Details: {d}"));
    }
    lines.push(format!("Next step: {next}"));

    Error::model(ModelError {
        message: lines.join("\n"),
        status_code: Some(status_code),
        code,
        provider,
        finish_reason,
        native_finish_reason,
        source,
    })
}

fn extract_error_details(payload: &Map<String, Value>) -> ErrorDetails {
    let choice = first_choice(payload);
    let error = payload
        .get("error")
        .filter(|v| !v.is_null())
        .or_else(|| choice.get("error").filter(|v| !v.is_null()));
    let fields = error.and_then(Value::as_object);
    let metadata = fields
        .and_then(|f| f.get("metadata"))
        .and_then(Value::as_object);

    let provider = metadata
        .and_then(|m| non_empty_str(m.get("provider_name")))
        .or_else(|| non_empty_str(payload.get("provider")))
        .map(str::to_string);

    let raw_code = fields.and_then(|f| f.get("code").or_else(|| f.get("type")));
    let code = match raw_code {
        Some(Value::String(s)) => Some(ErrorCode::String(s.clone())),
        Some(Value::Number(n)) => n.as_i64().map(ErrorCode::Int),
        _ => None,
    };

    let mut message = fields
        .and_then(|f| non_empty_str(f.get("message")))
        .or_else(|| non_empty_str(error))
        .or_else(|| non_empty_str(payload.get("message")))
        .map(str::to_string);

    if let Some(upstream) = metadata.and_then(|m| upstream_message(m.get("raw"))) {
        message = Some(match message {
            Some(msg) => format!("{msg}: {upstream}"),
            None => upstream,
        });
    }

    ErrorDetails {
        provider,
        code,
        finish_reason: non_empty_str(choice.get("finish_reason")).map(str::to_string),
        native_finish_reason: non_empty_str(choice.get("native_finish_reason")).map(str::to_string),
        message,
    }
}

fn upstream_message(raw: Option<&Value>) -> Option<String> {
    let text = raw?.as_str()?;
    let Ok(parsed) = serde_json::from_str::<Value>(text) else {
        return if text.is_empty() {
            None
        } else {
            Some(text.to_string())
        };
    };
    let obj = parsed.as_object()?;
    let inner = obj.get("error").and_then(Value::as_object).unwrap_or(obj);
    non_empty_str(inner.get("message")).map(str::to_string)
}

fn next_step(status: u16, code: Option<&ErrorCode>, native: Option<&str>) -> &'static str {
    if matches!(code, Some(ErrorCode::String(s)) if s == "invalid_json_schema") {
        return "Simplify the schema to the provider's supported JSON Schema subset \
                or change output mode.";
    }
    if native == Some("MALFORMED_FUNCTION_CALL") {
        return "Simplify the extraction schema or use Model(output_mode='json_schema') \
                with a Chat Completions model that supports it.";
    }
    if (300..400).contains(&status) {
        return "Set Model(base_url=...) to the final API endpoint; \
                provider redirects are not followed.";
    }
    match status {
        401 => "Check the API key or authentication headers for the selected provider.",
        403 => "Check that the account and credentials have permission to use the selected model.",
        404 => "Check the model identifier and Model(base_url=...), including the API version.",
        408 | 504 => TIMED_OUT,
        413 => "Reduce the source document size or split it into smaller inputs.",
        429 => "Wait before retrying; check the provider's rate limits, quota, and billing.",
        500..=599 => "Retry later; check the provider's service status if failures continue.",
        _ => {
            "Check the provider details, model, schema, media type, \
             and Model(settings=...) before retrying."
        }
    }
}

fn redact(value: Option<&str>, secrets: &[&str]) -> Option<String> {
    let mut text = value?.to_string();
    for secret in secrets {
        text = text.replace(secret, "[REDACTED]");
    }
    let collapsed: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    Some(collapsed.chars().take(2000).collect())
}

fn count(number: usize, noun: &str) -> String {
    if number == 1 {
        format!("{number} {noun}")
    } else {
        format!("{number} {noun}s")
    }
}

fn first_choice(payload: &Map<String, Value>) -> Map<String, Value> {
    payload
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|arr| arr.first())
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default()
}

fn non_empty_str(val: Option<&Value>) -> Option<&str> {
    val.and_then(Value::as_str).filter(|s| !s.is_empty())
}
