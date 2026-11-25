mod common;

use openextract::{
    Api, Error, ExtractOptions, PROMPT, extract, extract_value, extract_value_with_metadata,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use common::{Invoice, configured, mount_json, recorded_bodies};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
enum Currency {
    #[serde(rename = "usd")]
    Usd,
    #[serde(rename = "eur")]
    Eur,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
enum DocType {
    #[serde(rename = "invoice")]
    Invoice,
    #[serde(rename = "receipt")]
    Receipt,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
struct Triage {
    /// What kind of document is this?
    document_type: DocType,
    currency: Currency,
    is_paid: bool,
}

fn named_answers() -> Value {
    json!({
        "answers": {
            "document_type": {"type": "choice", "choice": "receipt", "confidence": 0.9},
            "currency": {"type": "choice", "choice": "eur", "confidence": 0.8},
            "is_paid": {"type": "noul", "noul": 0.97}
        }
    })
}

fn openai_answers() -> Value {
    json!({
        "answers": [
            {"type": "choice", "name": "currency", "choice": "eur"},
            {"type": "predicate", "name": "is_paid", "probability": 0.97},
            {"type": "choice", "name": "document_type", "choice": "receipt"}
        ]
    })
}

#[tokio::test]
async fn test_decisions_openrouter_openai_and_gemini_requests() {
    let expected = Triage {
        document_type: DocType::Receipt,
        currency: Currency::Eur,
        is_paid: true,
    };

    // OpenRouter Decisions
    let server = MockServer::start().await;
    mount_json(&server, 200, named_answers()).await;
    let res: Triage = extract(
        configured(&server, "openrouter:typesafe/jev-1.14"),
        "Receipt. Paid in EUR.",
        Some("Be strict."),
    )
    .await
    .unwrap();
    assert_eq!(res, expected);
    let reqs = server.received_requests().await.unwrap();
    assert_eq!(reqs[0].url.path(), "/decisions");
    let body = &recorded_bodies(&server).await[0];
    assert_eq!(body["model"], "typesafe/jev-1.14");
    assert_eq!(
        body["state"],
        json!([PROMPT, "Be strict.", "Receipt. Paid in EUR."])
    );
    assert_eq!(
        body["questions"]["is_paid"],
        json!({"type": "noul", "instructions": "is paid"})
    );
    assert_eq!(
        body["questions"]["document_type"]["instructions"],
        "What kind of document is this?"
    );

    // OpenAI Decisions (text + image)
    let server = MockServer::start().await;
    mount_json(&server, 200, openai_answers()).await;
    let res: Triage = extract(
        configured(&server, "openai-decisions:gpt-6-luna"),
        "Paid receipt in EUR.",
        Some("Be strict."),
    )
    .await
    .unwrap();
    assert_eq!(res, expected);
    let body = &recorded_bodies(&server).await[0];
    assert_eq!(body["model"], "gpt-6-luna");
    assert_eq!(
        body["input"][0],
        json!({"role": "developer", "content": "Be strict."})
    );
    assert_eq!(
        body["questions"][2],
        json!({"name": "is_paid", "type": "predicate", "instructions": "is paid"})
    );

    // Gemini Decisions (PDF + header x-goog-api-key)
    let server = MockServer::start().await;
    mount_json(&server, 200, named_answers()).await;
    let res: Triage = extract(
        configured(&server, "gemini-decisions:gemini-decision-1"),
        b"%PDF-1.7 fake".as_slice(),
        Some("Check carefully."),
    )
    .await
    .unwrap();
    assert_eq!(res, expected);
    let reqs = server.received_requests().await.unwrap();
    assert_eq!(reqs[0].headers.get("x-goog-api-key").unwrap(), "test-key");
    assert!(reqs[0].headers.get("authorization").is_none());
    let body = &recorded_bodies(&server).await[0];
    assert_eq!(body["input"][2]["type"], "document");
    assert_eq!(body["input"][2]["mime_type"], "application/pdf");
}

#[tokio::test]
async fn test_decisions_thresholds_limits_refusals_and_invalid_answers() {
    let server = MockServer::start().await;

    // Threshold: 0.49 -> false, 0.5 -> true
    for (prob, expected_bool) in [(0.49, false), (0.5, true)] {
        let srv = MockServer::start().await;
        mount_json(
            &srv,
            200,
            json!({"answers": [{"type": "predicate", "name": "is_paid", "probability": prob}]}),
        )
        .await;
        let schema = json!({"type": "object", "properties": {"is_paid": {"type": "boolean"}}});
        let val = extract_value(
            &schema,
            configured(&srv, "openai-decisions:gpt-6-luna"),
            b"text".as_slice(),
            ExtractOptions::new(),
        )
        .await
        .unwrap();
        assert_eq!(val["is_paid"], expected_bool);
    }

    // Unsupported schema field & settings rejected before request
    let err = extract::<Invoice>(
        configured(&server, "openai-decisions:gpt-6-luna"),
        b"text".as_slice(),
        None,
    )
    .await
    .unwrap_err();
    assert!(matches!(err, Error::InvalidConfig(_)));
    assert!(err.to_string().contains("only fill"));

    let err = extract::<Triage>(
        configured(&server, "openai-decisions:gpt-6-luna").with_setting("temperature", json!(0)),
        b"text".as_slice(),
        None,
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("settings"));

    // Unsupported media rejected
    let err = extract::<Triage>(
        configured(&server, "openai-decisions:gpt-6-luna"),
        b"%PDF-1.7".as_slice(),
        None,
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("text and images"));

    // Gemini option key limits (including non-breaking space \u{00a0}!)
    for bad_key in [
        "",
        &"x".repeat(129),
        " space",
        "space ",
        "new\nline",
        "no\x7f",
        "a\u{00a0}b",
    ] {
        let schema = json!({
            "type": "object",
            "properties": {"v": {"type": "string", "enum": [bad_key, "ok"]}}
        });
        let err = extract_value(
            &schema,
            configured(&server, "gemini-decisions:gemini-decision-1"),
            "text",
            ExtractOptions::new(),
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("option key"),
            "bad_key={bad_key:?}"
        );
    }

    // OpenAI > 400 fields rejected
    let mut big_props = Map::new();
    for i in 0..401 {
        big_props.insert(format!("q{i}"), json!({"type": "boolean"}));
    }
    let err = extract_value(
        &json!({"type": "object", "properties": big_props}),
        configured(&server, "openai-decisions:gpt-6-luna"),
        "text",
        ExtractOptions::new(),
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("1 and 400"));

    assert!(server.received_requests().await.unwrap().is_empty());

    // Refusal and invalid choice without echoing raw answer
    let srv = MockServer::start().await;
    mount_json(
        &srv,
        200,
        json!({"answers": [{"type": "refusal", "name": "is_paid"}]}),
    )
    .await;
    let err = extract_value(
        &json!({"type": "object", "properties": {"is_paid": {"type": "boolean"}}}),
        configured(&srv, "openai-decisions:gpt-6-luna"),
        "text",
        ExtractOptions::new(),
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("refused to answer 'is_paid'"));

    let srv = MockServer::start().await;
    mount_json(
        &srv,
        200,
        json!({"answers": {"currency": {"type": "choice", "choice": "private-invalid-choice"}}}),
    )
    .await;
    let err = extract_value(
        &json!({"type": "object", "properties": {"currency": {"type": "string", "enum": ["USD", "EUR"]}}}),
        configured(&srv, "openrouter:typesafe/jev-1.14"),
        "text",
        ExtractOptions::new(),
    )
    .await
    .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("Expected one of ['USD', 'EUR'] for field 'currency'"));
    assert!(!msg.contains("private-invalid-choice"));
}

#[tokio::test]
async fn test_openrouter_extractions_text_assets_metadata_truncation_and_retry() {
    let server = MockServer::start().await;
    let schema = json!({
        "type": "object",
        "properties": {
            "vendor": {"type": "string", "description": "Seller named on the invoice."},
            "total": {"type": "number", "exclusiveMinimum": 0, "x-openrouter": {"decimals": 2, "unit": "USD"}},
            "due_date": {"type": "string", "format": "date"},
            "paid_at": {"type": "string", "format": "date-time"},
            "cutoff": {"type": "string", "format": "time"},
            "purchase_order": {"anyOf": [{"type": "string", "minLength": 2}, {"type": "null"}]},
            "quantities": {"anyOf": [{"type": "array", "minItems": 1, "items": {"type": "integer"}}, {"type": "null"}]}
        },
        "required": ["vendor", "total", "due_date", "paid_at", "cutoff", "purchase_order", "quantities"]
    });

    let answers = json!({
        "vendor": {"value": "Acme", "span": [2, 6], "confidence": 0.95},
        "total": {"value": 12.5},
        "due_date": {"value": "2026-11-20"},
        "paid_at": {"value": "2026-11-11T12:00:00"},
        "cutoff": {"value": "17:30:00"},
        "purchase_order": {"value": null, "guess": "INV-1", "p_null": 0.9},
        "quantities": {"value": [1, 2], "spans": [null, [7, 8]]}
    });

    Mock::given(method("POST"))
        .and(path("/extractions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "state_text": "Acme invoice",
            "truncated": false,
            "usage": {"cost": 0.0001},
            "answers": answers
        })))
        .mount(&server)
        .await;

    let model = configured(&server, "openrouter:cloudflare/clairic")
        .with_setting("reference_date", json!("2026-11-11"));

    let res = extract_value_with_metadata(
        &schema,
        model,
        "Acme invoice",
        ExtractOptions::new().with_instructions("Be exact."),
    )
    .await
    .unwrap();

    assert_eq!(res.data["vendor"], "Acme");
    assert_eq!(res.data["total"], 12.5);
    assert_eq!(res.data["quantities"], json!([1, 2]));
    assert_eq!(res.answers["vendor"]["span"], json!([2, 6]));
    assert_eq!(res.metadata["truncated"], false);

    let body = &recorded_bodies(&server).await[0];
    assert_eq!(body["reference_date"], "2026-11-11");
    assert_eq!(body["state"], "Acme invoice");
    assert_eq!(
        body["fields"]["vendor"],
        json!({"type": "text", "instructions": "Seller named on the invoice.\nBe exact."})
    );
    assert_eq!(
        body["fields"]["total"],
        json!({"type": "number", "decimals": 2, "unit": "USD", "instructions": "total\nBe exact."})
    );
    assert_eq!(
        body["fields"]["quantities"],
        json!({"type": "list", "items": "number", "integer": true, "instructions": "quantities\nBe exact."})
    );

    // Asset encoding (PDF) + truncated handling
    let srv = MockServer::start().await;
    mount_json(
        &srv,
        200,
        json!({"truncated": true, "answers": {"vendor": {"value": "Acme"}}}),
    )
    .await;
    let simple = json!({"type": "object", "properties": {"vendor": {"type": "string"}}, "required": ["vendor"]});
    let err = extract_value(
        &simple,
        configured(&srv, "openrouter:scaledown/extract"),
        b"%PDF-1.7".as_slice(),
        ExtractOptions::new(),
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("read only part of the source"));
    let body = &recorded_bodies(&srv).await[0];
    assert_eq!(body["state"], "document");
    assert_eq!(body["assets"]["document"]["type"], "document");
    assert_eq!(body["assets"]["document"]["format"], "pdf");

    // With include_metadata=true, truncated succeeds
    let res = extract_value_with_metadata(
        &simple,
        configured(&srv, "openrouter:scaledown/extract"),
        b"%PDF-1.7".as_slice(),
        ExtractOptions::new(),
    )
    .await
    .unwrap();
    assert_eq!(res.data["vendor"], "Acme");
    assert_eq!(res.metadata["truncated"], true);

    // include_metadata on non-extractions API rejected before request
    let err = extract_value_with_metadata(
        &simple,
        configured(&srv, "openrouter:google/gemini-2.5-flash").with_api(Api::Chat),
        "text",
        ExtractOptions::new(),
    )
    .await
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("requires the OpenRouter Extractions API")
    );

    // Validation retry preserves state untouched and appends feedback to field instructions
    let retry_srv = MockServer::start().await;
    let positive_schema = json!({
        "type": "object",
        "properties": {"total": {"type": "integer", "exclusiveMinimum": 0}},
        "required": ["total"]
    });
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"answers": {"total": {"value": -9876}}})),
        )
        .up_to_n_times(1)
        .mount(&retry_srv)
        .await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({"answers": {"total": {"value": 10}}})),
        )
        .mount(&retry_srv)
        .await;
    let res = extract_value_with_metadata(
        &positive_schema,
        configured(&retry_srv, "openrouter:cloudflare/clairic").with_validation_retries(1),
        "Due: 10",
        ExtractOptions::new(),
    )
    .await
    .unwrap();
    assert_eq!(res.data["total"], 10);
    let bodies = recorded_bodies(&retry_srv).await;
    assert_eq!(bodies[0]["state"], "Due: 10");
    assert_eq!(bodies[1]["state"], "Due: 10");
    let fb = bodies[1]["fields"]["total"]["instructions"]
        .as_str()
        .unwrap();
    assert!(fb.contains("total"));
    assert!(!fb.contains("-9876"));

    // Explicit x-openrouter types (duration, estimate, forecast)
    let ext_srv = MockServer::start().await;
    mount_json(
        &ext_srv,
        200,
        json!({
            "answers": {
                "term_days": {"value": 30},
                "days_to_pay": {"value": 35, "p10": 30, "p90": 40},
                "payments": {"value": [50, 50], "steps": [{"median": 50}, {"median": 50}]}
            }
        }),
    )
    .await;
    let ext_schema = json!({
        "type": "object",
        "properties": {
            "term_days": {"type": "integer", "x-openrouter": {"type": "duration", "integer": true, "unit": "days"}},
            "days_to_pay": {"type": "number", "x-openrouter": {"type": "estimate", "min": 0, "max": 90, "step": 1}},
            "payments": {"type": "array", "items": {"type": "number"}, "x-openrouter": {"type": "forecast", "min": 0, "max": 100, "horizon": 2, "interval": "month"}}
        },
        "required": ["term_days", "days_to_pay", "payments"]
    });
    let res = extract_value_with_metadata(
        &ext_schema,
        configured(&ext_srv, "openrouter:cloudflare/clairic"),
        "Net 30",
        ExtractOptions::new(),
    )
    .await
    .unwrap();
    assert_eq!(res.data["term_days"], 30);
    assert_eq!(res.answers["days_to_pay"]["p10"], 30);
    let body = &recorded_bodies(&ext_srv).await[0];
    assert_eq!(body["fields"]["term_days"]["type"], "duration");
    assert_eq!(body["fields"]["days_to_pay"]["type"], "estimate");
    assert_eq!(body["fields"]["payments"]["type"], "forecast");
}
