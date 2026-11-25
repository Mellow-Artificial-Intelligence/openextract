//! POST model requests with bounded retries.

use std::future::Future;
use std::time::{Duration, SystemTime};

use reqwest::redirect::Policy;
use serde_json::{Map, Value};

use crate::diagnostics::{
    has_error_envelope, invalid_response_message, provider_error, request_error,
};
use crate::error::{Error, Result};
use crate::models::ResolvedModel;

const RETRY_STATUSES: &[u16] = &[408, 409, 429, 500, 502, 503, 504];
const KEY_HEADERS: &[&str] = &["authorization", "x-api-key", "x-goog-api-key", "api-key"];

/// Build a default [`reqwest::Client`] configured for provider requests (redirects disabled).
pub fn default_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .redirect(Policy::none())
        .build()
        .map_err(|e| Error::InvalidConfig(format!("Failed to build HTTP client: {e}")))
}

/// POST a JSON request to the resolved model endpoint with bounded retries.
pub async fn post(
    model: &ResolvedModel,
    path: &str,
    body: &Value,
    client: &reqwest::Client,
) -> Result<Map<String, Value>> {
    post_with_sleep(model, path, body, client, |secs| {
        tokio::time::sleep(Duration::from_secs_f64(secs))
    })
    .await
}

pub(crate) async fn post_with_sleep<F, Fut>(
    model: &ResolvedModel,
    path: &str,
    body: &Value,
    client: &reqwest::Client,
    mut sleep_fn: F,
) -> Result<Map<String, Value>>
where
    F: FnMut(f64) -> Fut,
    Fut: Future<Output = ()>,
{
    let url = format!("{}{}", model.base_url, path);
    let timeout = Duration::from_secs_f64(model.config.timeout);
    let mut attempt: u32 = 0;

    loop {
        let result = client
            .post(&url)
            .headers(model.headers.clone())
            .json(body)
            .timeout(timeout)
            .send()
            .await;

        let retry_after = match result {
            Err(err) => {
                if attempt == model.config.max_retries || !is_transient_error(&err) {
                    return Err(request_error(err, (attempt + 1) as usize));
                }
                None
            }
            Ok(response) => {
                let status = response.status().as_u16();
                if attempt == model.config.max_retries || !RETRY_STATUSES.contains(&status) {
                    return read_payload(response, model, &url).await;
                }
                response
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_string)
            }
        };

        let wait = compute_delay(attempt, retry_after.as_deref(), SystemTime::now());
        sleep_fn(wait).await;
        attempt += 1;
    }
}

fn is_transient_error(err: &reqwest::Error) -> bool {
    // Decoding, builder, and redirect errors are not transient.
    if err.is_decode() || err.is_builder() || err.is_redirect() {
        return false;
    }
    err.is_timeout() || err.is_connect() || err.is_request() || err.is_body()
}

/// Honor `Retry-After` up to 30 seconds; otherwise back off exponentially up to 8 seconds.
pub fn compute_delay(attempt: u32, retry_after: Option<&str>, now: SystemTime) -> f64 {
    if let Some(raw) = retry_after
        && let Some(seconds) = parse_retry_after(raw, now)
    {
        return seconds.clamp(0.0, 30.0);
    }
    0.5 * 2_f64.powi(attempt.min(4) as i32)
}

fn parse_retry_after(value: &str, now: SystemTime) -> Option<f64> {
    if let Ok(seconds) = value.trim().parse::<f64>()
        && seconds.is_finite()
    {
        return Some(seconds);
    }
    let target = httpdate::parse_http_date(value.trim()).ok()?;
    match target.duration_since(now) {
        Ok(dur) => Some(dur.as_secs_f64()),
        Err(err) => Some(-err.duration().as_secs_f64()),
    }
}

async fn read_payload(
    response: reqwest::Response,
    model: &ResolvedModel,
    expected_url: &str,
) -> Result<Map<String, Value>> {
    let status = response.status();
    let status_code = status.as_u16();

    // Guard against caller-owned clients that followed a redirect despite policy guidance.
    if url::Url::parse(expected_url)
        .ok()
        .is_some_and(|expected| expected != *response.url())
    {
        return Err(provider_error(
            &Map::new(),
            307,
            &collect_secrets(model),
            None,
        ));
    }

    if !status.is_success() {
        let status_err = response.error_for_status_ref().err();
        let body_map = match response.bytes().await {
            Ok(bytes) => serde_json::from_slice::<Value>(&bytes)
                .ok()
                .and_then(|v| match v {
                    Value::Object(map) => Some(map),
                    _ => None,
                })
                .unwrap_or_default(),
            Err(_) => Map::new(),
        };
        let secrets = collect_secrets(model);
        let source = status_err.map(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>);
        return Err(provider_error(&body_map, status_code, &secrets, source));
    }

    let bytes = response.bytes().await.map_err(|e| request_error(e, 1))?;

    let context = if model.api.is_decisions() {
        "decision answers"
    } else {
        "model response"
    };

    let parsed: Value = serde_json::from_slice(&bytes).map_err(|e| {
        Error::schema_validation_with_source(
            invalid_response_message("The provider response body was not valid JSON.", context),
            e,
        )
    })?;

    let Value::Object(payload) = parsed else {
        return Err(Error::schema_validation(invalid_response_message(
            "Expected a JSON object from the model provider.",
            context,
        )));
    };

    if has_error_envelope(&payload) {
        return Err(provider_error(
            &payload,
            status_code,
            &collect_secrets(model),
            None,
        ));
    }

    Ok(payload)
}

fn collect_secrets(model: &ResolvedModel) -> Vec<String> {
    let mut secrets: Vec<String> = model.config.headers.values().cloned().collect();
    if let Some(ref key) = model.config.api_key
        && !key.is_empty()
    {
        secrets.push(key.clone());
    }
    for header in KEY_HEADERS {
        if let Some(val) = model.headers.get(*header).and_then(|v| v.to_str().ok())
            && !val.is_empty()
        {
            secrets.push(val.to_string());
            if let Some((_, token)) = val.split_once(' ')
                && !token.is_empty()
            {
                secrets.push(token.to_string());
            }
        }
    }
    secrets
}
