mod common;

use std::time::{Duration, UNIX_EPOCH};

use openextract::transport::compute_delay;
use openextract::{
    Error, ErrorCode, ExtractOptions, Model, OutputMode, extract, extract_value,
    extract_with_options,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};

use common::{Invoice, configured, mount_json, output, recorded_bodies};

#[tokio::test]
async fn test_validation_retry_sends_sanitized_feedback_and_recovers() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(output(
            "responses",
            Some(json!({"total": "private-invalid-value"})),
        )))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(output("responses", None)))
        .mount(&server)
        .await;

    let model = configured(&server, "openai:test").with_validation_retries(1);
    let res: Invoice = extract(model, b"text".as_slice(), Some("Use EUR."))
        .await
        .unwrap();
    assert_eq!(res.total, 42.0);

    let bodies = recorded_bodies(&server).await;
    assert_eq!(bodies.len(), 2);
    let feedback = bodies[1]["instructions"].as_str().unwrap();
    assert!(feedback.starts_with("Use EUR."));
    assert!(feedback.contains("total"));
    assert!(!feedback.contains("private-invalid-value"));
}

#[tokio::test]
async fn test_validation_diagnostics_bounded_and_nested_without_raw_values() {
    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct Item {
        quantity: i64,
    }
    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct Receipt {
        items: Vec<Item>,
    }

    let server = MockServer::start().await;
    let bad_items: Vec<Value> = (0..7)
        .map(|_| json!({"quantity": "private-invalid-value"}))
        .collect();
    mount_json(
        &server,
        200,
        output("responses", Some(json!({"items": bad_items}))),
    )
    .await;

    let err = extract::<Receipt>(
        configured(&server, "openai:test"),
        b"Receipt".as_slice(),
        None,
    )
    .await
    .unwrap_err();
    assert!(matches!(err, Error::SchemaValidation { .. }));
    let msg = err.to_string();
    assert!(msg.contains("after 1 validation attempt"));
    assert!(msg.contains("items.0.quantity: Input should be a valid integer"));
    assert!(msg.contains("items.4.quantity"));
    assert!(!msg.contains("items.5.quantity"));
    assert!(msg.contains("2 more validation errors omitted"));
    assert!(!msg.contains("private-invalid-value"));
    assert!(msg.contains("Next step:"));
}

#[tokio::test]
async fn test_transient_status_retries_and_non_transient_fails_immediately() {
    for status in [408, 409, 429, 500, 502, 503, 504] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(status).insert_header("Retry-After", "0"))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(output("responses", None)))
            .mount(&server)
            .await;

        let model = configured(&server, "openai:test").with_max_retries(1);
        let inv: Invoice = extract(model, b"text".as_slice(), None).await.unwrap();
        assert_eq!(inv.total, 42.0);
        let reqs = server.received_requests().await.unwrap();
        assert_eq!(reqs.len(), 2, "expected retry for HTTP {status}");
        assert_eq!(reqs[0].body, reqs[1].body);
    }

    for status in [400, 401, 403] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(status))
            .mount(&server)
            .await;
        let model = configured(&server, "openai:test").with_max_retries(2);
        let err = extract::<Invoice>(model, b"text".as_slice(), None)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Model(_)));
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }
}

#[tokio::test]
async fn test_provider_redirect_is_not_followed() {
    let model_server = MockServer::start().await;
    let untrusted = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(307)
                .insert_header("Location", format!("{}/steal", untrusted.uri())),
        )
        .mount(&model_server)
        .await;

    let err = extract::<Invoice>(
        configured(&model_server, "openai:test"),
        b"text".as_slice(),
        None,
    )
    .await
    .unwrap_err();
    match err {
        Error::Model(me) => {
            assert_eq!(me.status_code, Some(307));
            assert!(me.message.contains("final API endpoint"));
        }
        other => panic!("expected ModelError, got {other:?}"),
    }
    assert!(untrusted.received_requests().await.unwrap().is_empty());

    // Even if a caller supplies a client with redirects enabled, post detects and rejects redirect.
    let following_client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
        .unwrap();
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(output("responses", None)))
        .mount(&untrusted)
        .await;
    let err = extract_with_options::<Invoice>(
        configured(&model_server, "openai:test"),
        b"text".as_slice(),
        ExtractOptions::new().with_client(&following_client),
    )
    .await
    .unwrap_err();
    assert!(matches!(err, Error::Model(me) if me.status_code == Some(307)));
}

#[test]
fn test_retry_after_bounds_http_date_and_backoff_cap() {
    let now = UNIX_EPOCH + Duration::from_secs(1_767_225_600); // 2026-01-01T00:00:00Z
    let cases = [
        ("0", 0.0),
        ("2.5", 2.5),
        ("999", 30.0),
        ("-1", 0.0),
        ("nan", 0.5),
        ("inf", 0.5),
        ("not-a-date", 0.5),
    ];
    for (header, expected) in cases {
        assert_eq!(
            compute_delay(0, Some(header), now),
            expected,
            "header={header}"
        );
    }

    let future = httpdate::fmt_http_date(now + Duration::from_secs(10));
    assert_eq!(compute_delay(0, Some(&future), now), 10.0);
    assert_eq!(compute_delay(10, None, now), 8.0);
}

#[tokio::test]
async fn test_provider_error_envelopes_redaction_and_next_steps() {
    // HTTP 200 application error (MALFORMED_FUNCTION_CALL) is NOT retried
    let server = MockServer::start().await;
    mount_json(
        &server,
        200,
        json!({
            "provider": "Google AI Studio",
            "choices": [{
                "finish_reason": "error",
                "native_finish_reason": "MALFORMED_FUNCTION_CALL",
                "message": {"tool_calls": []}
            }]
        }),
    )
    .await;
    let model = configured(&server, "openrouter:google/gemini-2.5-flash")
        .with_output_mode(OutputMode::JsonSchema)
        .with_max_retries(2);
    let err = extract::<Invoice>(model, b"Invoice".as_slice(), None)
        .await
        .unwrap_err();
    match err {
        Error::Model(me) => {
            assert_eq!(me.status_code, Some(200));
            assert_eq!(me.provider.as_deref(), Some("Google AI Studio"));
            assert_eq!(me.finish_reason.as_deref(), Some("error"));
            assert_eq!(
                me.native_finish_reason.as_deref(),
                Some("MALFORMED_FUNCTION_CALL")
            );
            assert!(me.message.contains("output_mode='json_schema'"));
        }
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(server.received_requests().await.unwrap().len(), 1);

    // Upstream metadata.raw (JSON and plain text) + credential redaction
    for status in [200u16, 401] {
        let server = MockServer::start().await;
        mount_json(
            &server,
            status,
            json!({
                "error": {
                    "code": "private-key",
                    "message": "Invalid credential private-key",
                    "metadata": {
                        "provider_name": "provider-private-key",
                        "raw": "{\"error\":{\"message\":\"upstream rejected private-header\"}}"
                    }
                }
            }),
        )
        .await;
        let model = configured(&server, "openai:test")
            .with_api_key("private-key")
            .with_header("X-Custom-Key", "private-header");
        let err = extract::<Invoice>(model, b"Invoice".as_slice(), None)
            .await
            .unwrap_err();
        match err {
            Error::Model(me) => {
                assert!(!me.message.contains("private-key"));
                assert!(!me.message.contains("private-header"));
                assert!(me.message.contains("[REDACTED]"));
                assert_eq!(me.code, Some(ErrorCode::String("[REDACTED]".to_string())));
                assert_eq!(me.provider.as_deref(), Some("provider-[REDACTED]"));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    // Large multiline error message is collapsed and bounded without hiding Next step
    let server = MockServer::start().await;
    let giant = "private-key\n".repeat(1000);
    mount_json(&server, 401, json!({"error": {"message": giant}})).await;
    let model = configured(&server, "openai:test").with_api_key("private-key");
    let err = extract::<Invoice>(model, b"Invoice".as_slice(), None)
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(!msg.contains("private-key"));
    assert!(msg.len() < 2300);
    assert_eq!(msg.lines().count(), 3);
    assert!(msg.lines().last().unwrap().starts_with("Next step:"));

    // Next-step hints matrix
    let hints: &[(u16, Option<&str>, Option<&str>, &str)] = &[
        (
            400,
            Some("invalid_json_schema"),
            None,
            "supported JSON Schema subset",
        ),
        (
            200,
            None,
            Some("MALFORMED_FUNCTION_CALL"),
            "output_mode='json_schema'",
        ),
        (400, None, None, "Model(settings=...)"),
        (401, None, None, "API key or authentication headers"),
        (403, None, None, "permission"),
        (404, None, None, "model identifier"),
        (408, None, None, "provider or gateway timed out"),
        (413, None, None, "document size"),
        (429, None, None, "rate limits, quota, and billing"),
        (500, None, None, "provider's service status"),
        (504, None, None, "provider or gateway timed out"),
    ];
    for &(status, code, native, expected_hint) in hints {
        let server = MockServer::start().await;
        mount_json(
            &server,
            status,
            json!({
                "error": {"message": "Request was rejected", "code": code},
                "provider": "Example Provider",
                "choices": [{"finish_reason": "error", "native_finish_reason": native}]
            }),
        )
        .await;
        let err = extract::<Invoice>(
            configured(&server, "openai:test"),
            b"Invoice".as_slice(),
            None,
        )
        .await
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains(expected_hint), "status={status} msg={msg}");
    }

    // Non-JSON 503 drops body
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(503).set_body_string("private server details"))
        .mount(&server)
        .await;
    let err = extract::<Invoice>(configured(&server, "openai:test"), b"text".as_slice(), None)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("HTTP 503"));
    assert!(!err.to_string().contains("private"));

    // Integer error code preserved
    let server = MockServer::start().await;
    mount_json(
        &server,
        400,
        json!({"error": {"code": 400, "message": "Bad"}}),
    )
    .await;
    let err = extract_value(
        &json!({"type": "object", "properties": {"total": {"type": "number"}}}),
        Model::new("openai:test")
            .with_base_url(server.uri())
            .with_api_key("k")
            .with_max_retries(0),
        b"text".as_slice(),
        ExtractOptions::new(),
    )
    .await
    .unwrap_err();
    assert!(matches!(err, Error::Model(me) if me.code == Some(ErrorCode::Int(400))));
}
