mod common;

use std::collections::HashMap;

use openextract::schemas::{anthropic_json, openai_strict, schema_for_type};
use openextract::{
    Api, Error, ExtractOptions, Model, OutputMode, extract, extract_value, extract_with_options,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tempfile::tempdir;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use common::{Invoice, configured, custom, mount_json, output, recorded_bodies};

const PNG_B64: &str = "iVBORw==";
const PNG_URI: &str = "data:image/png;base64,iVBORw==";
const PDF_URI: &str = "data:application/pdf;base64,iVBORw==";

fn document_part(body: &Value, api: Api) -> &Value {
    match api {
        Api::Responses => body["input"].as_array().unwrap().last().unwrap()["content"]
            .as_array()
            .unwrap()
            .last()
            .unwrap(),
        Api::Gemini => body["contents"].as_array().unwrap().last().unwrap()["parts"]
            .as_array()
            .unwrap()
            .last()
            .unwrap(),
        Api::Ollama => body["messages"].as_array().unwrap().last().unwrap(),
        _ => body["messages"].as_array().unwrap().last().unwrap()["content"]
            .as_array()
            .unwrap()
            .last()
            .unwrap(),
    }
}

#[tokio::test]
async fn test_all_general_adapter_request_contracts() {
    let cases = [
        (
            Api::Responses,
            "openai:test",
            "/responses",
            "authorization",
            "responses",
        ),
        (
            Api::Chat,
            "openai-chat:test",
            "/chat/completions",
            "authorization",
            "chat",
        ),
        (
            Api::Anthropic,
            "anthropic:test",
            "/messages",
            "x-api-key",
            "anthropic",
        ),
        (
            Api::Gemini,
            "google-gla:test",
            "/models/test:generateContent",
            "x-goog-api-key",
            "gemini",
        ),
        (
            Api::Ollama,
            "ollama:test",
            "/api/chat",
            "authorization",
            "ollama",
        ),
        (
            Api::Cohere,
            "cohere:test",
            "/chat",
            "authorization",
            "cohere",
        ),
        (
            Api::Bedrock,
            "bedrock:test",
            "/model/test/converse",
            "authorization",
            "bedrock",
        ),
    ];

    for (api, name, expected_path, key_header, reply_api) in cases {
        for instructions in [None, Some("Use EUR.")] {
            let server = MockServer::start().await;
            mount_json(&server, 200, output(reply_api, None)).await;
            let model = configured(&server, name);
            let res: Invoice = extract(model, "Total: 42", instructions).await.unwrap();
            assert_eq!(res.total, 42.0);

            let reqs = server.received_requests().await.unwrap();
            assert_eq!(reqs[0].url.path(), expected_path, "path for {api}");
            let expected_auth = if key_header == "authorization" {
                "Bearer test-key"
            } else {
                "test-key"
            };
            assert_eq!(
                reqs[0].headers.get(key_header).unwrap().to_str().unwrap(),
                expected_auth,
                "header for {api}"
            );
            if api == Api::Anthropic {
                assert_eq!(
                    reqs[0].headers.get("anthropic-version").unwrap(),
                    "2023-06-01"
                );
            }
            let body: Value = serde_json::from_slice(&reqs[0].body).unwrap();
            assert!(document_part(&body, api).to_string().contains("Total: 42"));
            if let Some(instr) = instructions {
                assert!(body.to_string().contains(instr));
            }
            if api == Api::Cohere {
                assert!(
                    body["messages"][body["messages"].as_array().unwrap().len() - 1]["content"][0]
                        ["text"]
                        .as_str()
                        .unwrap()
                        .contains("JSON")
                );
            }
        }
    }
}

#[tokio::test]
async fn test_media_encoding_across_general_adapters() {
    let cases = [
        (
            Api::Responses,
            "responses",
            "image/png",
            json!({"type": "input_image", "image_url": PNG_URI}),
        ),
        (
            Api::Responses,
            "responses",
            "application/pdf",
            json!({"type": "input_file", "filename": "document.pdf", "file_data": PDF_URI}),
        ),
        (
            Api::Chat,
            "chat",
            "image/png",
            json!({"type": "image_url", "image_url": {"url": PNG_URI}}),
        ),
        (
            Api::Chat,
            "chat",
            "application/pdf",
            json!({"type": "file", "file": {"filename": "document.pdf", "file_data": PDF_URI}}),
        ),
        (
            Api::Chat,
            "chat",
            "audio/mpeg",
            json!({"type": "input_audio", "input_audio": {"format": "mp3", "data": PNG_B64}}),
        ),
        (
            Api::Chat,
            "chat",
            "audio/wav",
            json!({"type": "input_audio", "input_audio": {"format": "wav", "data": PNG_B64}}),
        ),
        (
            Api::Cohere,
            "cohere",
            "image/png",
            json!({"type": "image_url", "image_url": {"url": PNG_URI}}),
        ),
        (
            Api::Anthropic,
            "anthropic",
            "image/png",
            json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64}}),
        ),
        (
            Api::Anthropic,
            "anthropic",
            "application/pdf",
            json!({"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": PNG_B64}}),
        ),
        (
            Api::Gemini,
            "gemini",
            "video/mp4",
            json!({"inlineData": {"mimeType": "video/mp4", "data": PNG_B64}}),
        ),
        (
            Api::Gemini,
            "gemini",
            "audio/flac",
            json!({"inlineData": {"mimeType": "audio/flac", "data": PNG_B64}}),
        ),
        (
            Api::Gemini,
            "gemini",
            "application/pdf",
            json!({"inlineData": {"mimeType": "application/pdf", "data": PNG_B64}}),
        ),
        (
            Api::Bedrock,
            "bedrock",
            "image/png",
            json!({"image": {"format": "png", "source": {"bytes": PNG_B64}}}),
        ),
        (
            Api::Bedrock,
            "bedrock",
            "application/pdf",
            json!({"document": {"format": "pdf", "name": "document", "source": {"bytes": PNG_B64}}}),
        ),
    ];

    for (api, reply_api, media, expected) in cases {
        let server = MockServer::start().await;
        mount_json(&server, 200, output(reply_api, None)).await;
        let _: Invoice = extract_with_options(
            custom(&server, api),
            b"\x89PNG".as_slice(),
            ExtractOptions::new().with_media_type(media),
        )
        .await
        .unwrap();
        let bodies = recorded_bodies(&server).await;
        assert_eq!(
            document_part(&bodies[0], api),
            &expected,
            "media {media} on {api}"
        );
    }

    // Ollama images & model tags with colons
    let server = MockServer::start().await;
    mount_json(&server, 200, output("ollama", None)).await;
    let model = Model::new("ollama:llama3:latest")
        .with_base_url(server.uri())
        .with_max_retries(0)
        .with_validation_retries(0);
    let _: Invoice = extract(model, b"\x89PNG".as_slice(), None).await.unwrap();
    let bodies = recorded_bodies(&server).await;
    assert_eq!(bodies[0]["model"], "llama3:latest");
    assert_eq!(bodies[0]["messages"][0]["images"], json!([PNG_B64]));
}

#[tokio::test]
async fn test_unsupported_media_and_settings_rejected_before_request() {
    let server = MockServer::start().await;
    for api in [
        Api::Responses,
        Api::Chat,
        Api::Anthropic,
        Api::Gemini,
        Api::Ollama,
        Api::Cohere,
        Api::Bedrock,
    ] {
        let err = extract_with_options::<Invoice>(
            custom(&server, api),
            b"file".as_slice(),
            ExtractOptions::new().with_media_type("application/x-unknown"),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, Error::InvalidConfig(_)));
        assert!(err.to_string().contains("does not support"));
    }

    let bad_settings = [
        (Api::Responses, "model", json!("wrong")),
        (Api::Responses, "instructions", json!("replace")),
        (Api::Chat, "messages", json!([])),
        (Api::Gemini, "responseJsonSchema", json!({})),
        (Api::Gemini, "responseFormat", json!([])),
        (Api::Chat, "stream", json!(true)),
        (Api::Anthropic, "output_config", json!({})),
        (Api::Anthropic, "output_format", json!({})),
    ];
    for (api, key, val) in bad_settings {
        let err = extract::<Invoice>(
            custom(&server, api).with_setting(key, val),
            b"text".as_slice(),
            None,
        )
        .await
        .unwrap_err();
        assert!(matches!(err, Error::InvalidConfig(_)));
    }

    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn test_native_settings_forwarding() {
    let cases = [
        (
            Api::Responses,
            "responses",
            "temperature",
            json!(0),
            vec!["temperature"],
        ),
        (
            Api::Anthropic,
            "anthropic",
            "max_tokens",
            json!(1024),
            vec!["max_tokens"],
        ),
        (
            Api::Gemini,
            "gemini",
            "maxOutputTokens",
            json!(1024),
            vec!["generationConfig", "maxOutputTokens"],
        ),
        (
            Api::Ollama,
            "ollama",
            "options",
            json!({"temperature": 0}),
            vec!["options"],
        ),
        (
            Api::Bedrock,
            "bedrock",
            "inferenceConfig",
            json!({"maxTokens": 1024}),
            vec!["inferenceConfig"],
        ),
    ];
    for (api, reply_api, key, val, path_keys) in cases {
        let server = MockServer::start().await;
        mount_json(&server, 200, output(reply_api, None)).await;
        let _: Invoice = extract(
            custom(&server, api).with_setting(key, val.clone()),
            b"text".as_slice(),
            None,
        )
        .await
        .unwrap();
        let bodies = recorded_bodies(&server).await;
        let mut cursor = &bodies[0];
        for seg in path_keys {
            cursor = &cursor[seg];
        }
        assert_eq!(cursor, &val);
    }
}

#[tokio::test]
async fn test_chat_json_schema_mode_and_validation_retry() {
    let server = MockServer::start().await;
    let schema = json!({
        "type": "object",
        "properties": {
            "total": {"type": "number"},
            "items": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "quantity": {"type": "integer"}
                    },
                    "required": ["description", "quantity"]
                }
            }
        },
        "required": ["total", "items"]
    });
    let valid = json!({"total": 42.5, "items": [{"description": "Notebook", "quantity": 2}]});
    let invalid = json!({"total": 42.5, "items": []});

    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(output("chat-json", Some(invalid))))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(output("chat-json", Some(valid.clone()))),
        )
        .mount(&server)
        .await;

    let model = configured(&server, "openai-chat:test")
        .with_output_mode(OutputMode::JsonSchema)
        .with_validation_retries(1);

    let res = extract_value(
        &schema,
        model,
        b"Invoice".as_slice(),
        ExtractOptions::new().with_instructions("Include every item."),
    )
    .await
    .unwrap();
    assert_eq!(res, valid);

    let bodies = recorded_bodies(&server).await;
    assert_eq!(bodies.len(), 2);
    assert!(bodies[0].get("tools").is_none());
    assert_eq!(bodies[0]["response_format"]["type"], "json_schema");
    assert_eq!(bodies[0]["response_format"]["json_schema"]["strict"], true);
    assert_eq!(
        bodies[0]["response_format"]["json_schema"]["schema"]["additionalProperties"],
        false
    );
    assert!(
        bodies[1]["messages"][0]["content"]
            .as_str()
            .unwrap()
            .contains("items")
    );
}

#[tokio::test]
async fn test_anthropic_json_schema_mode_and_local_constraint_validation() {
    let server = MockServer::start().await;
    let schema = json!({
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": 2,
                "maxItems": 5,
                "items": {"$ref": "#/$defs/Item"}
            },
            "note": {"type": ["string", "null"], "default": null}
        },
        "required": ["items"],
        "$defs": {
            "Item": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 2, "maxLength": 20, "pattern": "^\\w+$"},
                    "quantity": {"type": "integer", "minimum": 1, "maximum": 10, "multipleOf": 1}
                },
                "required": ["name", "quantity"]
            }
        }
    });

    // Constraint violation caught locally even though Anthropic stripped minLength/minimum
    mount_json(
        &server,
        200,
        output(
            "anthropic-json",
            Some(json!({"items": [{"name": "!", "quantity": 0}]})),
        ),
    )
    .await;

    let model = configured(&server, "anthropic:test")
        .with_output_mode(OutputMode::JsonSchema)
        .with_setting("max_tokens", json!(1234));

    let err = extract_value(
        &schema,
        model,
        b"Invoice".as_slice(),
        ExtractOptions::new().with_instructions("Read all items"),
    )
    .await
    .unwrap_err();
    assert!(matches!(err, Error::SchemaValidation { .. }));
    let msg = err.to_string();
    assert!(msg.contains("string_too_short") || msg.contains("greater_than_equal"));
    assert!(!msg.contains('!')); // no raw value echoed

    let bodies = recorded_bodies(&server).await;
    assert!(bodies[0].get("tools").is_none());
    assert_eq!(bodies[0]["max_tokens"], 1234);
    assert_eq!(bodies[0]["system"], "Read all items");
    let sent_schema = &bodies[0]["output_config"]["format"]["schema"];
    assert_eq!(sent_schema["additionalProperties"], false);
    assert_eq!(sent_schema["required"], json!(["items"]));
    assert_eq!(sent_schema["properties"]["note"]["default"], Value::Null);
    assert!(sent_schema["properties"]["items"].get("minItems").is_none());
    let qty = &sent_schema["$defs"]["Item"]["properties"]["quantity"];
    assert!(qty.get("minimum").is_none());
    assert!(qty["description"].as_str().unwrap().contains("minimum"));
}

#[test]
fn test_schema_adapters_strict_and_anthropic() {
    #[derive(Serialize, Deserialize, JsonSchema)]
    struct MapHolder {
        values: HashMap<String, i64>,
    }

    let map_schema = schema_for_type::<MapHolder>().unwrap();
    assert!(
        openai_strict(&map_schema)
            .unwrap_err()
            .to_string()
            .contains("open-ended dictionaries")
    );
    assert!(
        anthropic_json(&map_schema)
            .unwrap_err()
            .to_string()
            .contains("Anthropic JSON output")
    );

    // Recursive ref rejected by anthropic_json
    let recursive = json!({
        "$defs": {"Node": {"type": "object", "properties": {"next": {"$ref": "#/$defs/Node"}}}},
        "$ref": "#/$defs/Node"
    });
    assert!(
        anthropic_json(&recursive)
            .unwrap_err()
            .to_string()
            .contains("recursive")
    );

    for unsupported in [
        json!({"$ref": "https://example.com/schema"}),
        json!({"allOf": []}),
        json!({"prefixItems": []}),
        json!({"patternProperties": {}}),
        json!({"enum": [{}]}),
    ] {
        assert!(anthropic_json(&unsupported).is_err());
    }

    // Supported vs unsupported formats & minItems
    assert_eq!(
        anthropic_json(&json!({"type": "string", "format": "date"})).unwrap()["format"],
        "date"
    );
    assert!(
        anthropic_json(&json!({"type": "string", "format": "password"}))
            .unwrap()
            .get("format")
            .is_none()
    );
    assert_eq!(
        anthropic_json(&json!({"type": "array", "minItems": 1})).unwrap()["minItems"],
        1
    );

    // Tagged union oneOf -> anyOf, discriminator stripped
    let union = json!({
        "type": "object",
        "properties": {
            "pet": {
                "oneOf": [{"$ref": "#/$defs/Cat"}, {"$ref": "#/$defs/Dog"}],
                "discriminator": {"propertyName": "kind"}
            }
        },
        "$defs": {
            "Cat": {"type": "object", "properties": {"kind": {"const": "cat"}}},
            "Dog": {"type": "object", "properties": {"kind": {"const": "dog"}}}
        }
    });
    for adapted in [
        openai_strict(&union).unwrap(),
        anthropic_json(&union).unwrap(),
    ] {
        let pet = &adapted["properties"]["pet"];
        assert!(pet.get("anyOf").is_some());
        assert!(pet.get("oneOf").is_none());
        assert!(pet.get("discriminator").is_none());
        assert_eq!(adapted["$defs"]["Cat"]["additionalProperties"], false);
    }
}

#[tokio::test]
async fn test_gemini_office_documents_and_spreadsheet_metadata() {
    let server = MockServer::start().await;
    mount_json(&server, 200, output("gemini", None)).await;
    let dir = tempdir().unwrap();

    let selection = json!({
        "sheetIndex": 2,
        "visibleCells": "A1:H50",
        "contentLayers": ["DATA", "FORMULAS"]
    });
    let xlsx_path = dir.path().join("workbook.xlsx");
    std::fs::write(&xlsx_path, b"\x00\xffmock").unwrap();

    let model = configured(&server, "google:gemini-4-hydrogen")
        .with_setting("spreadsheetMetadata", selection.clone())
        .with_setting("temperature", json!(0));

    let res: Invoice = extract(model, xlsx_path.as_path(), Some("Use final total."))
        .await
        .unwrap();
    assert_eq!(res.total, 42.0);
    let bodies = recorded_bodies(&server).await;
    let part = document_part(&bodies[0], Api::Gemini);
    assert_eq!(part["spreadsheetMetadata"], selection);
    assert_eq!(
        part["inlineData"]["mimeType"],
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    );
    assert_eq!(bodies[0]["generationConfig"]["temperature"], 0);
    assert!(
        bodies[0]["generationConfig"]
            .get("spreadsheetMetadata")
            .is_none()
    );

    // Reject spreadsheetMetadata on non-spreadsheet inputs before sending
    let err = extract_with_options::<Invoice>(
        configured(&server, "google:gemini-4-hydrogen")
            .with_setting("spreadsheetMetadata", json!({"sheetIndex": 1})),
        b"text".as_slice(),
        ExtractOptions::new().with_media_type("application/pdf"),
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("spreadsheetMetadata requires"));

    // Reject RTF on Gemini
    let err = extract_with_options::<Invoice>(
        configured(&server, "google:gemini-4-hydrogen"),
        b"rtf".as_slice(),
        ExtractOptions::new().with_media_type("application/rtf"),
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("gemini adapter does not support"));
}

#[tokio::test]
async fn test_mistral_ocr_contract_and_limits() {
    let server = MockServer::start().await;
    let schema = json!({
        "type": "object",
        "properties": {
            "issued": {"type": "string", "format": "date"},
            "total": {"type": "string"}
        },
        "required": ["issued", "total"]
    });

    Mock::given(method("POST"))
        .and(path("/ocr"))
        .respond_with(ResponseTemplate::new(200).set_body_json(output(
            "mistral-ocr",
            Some(json!({"issued": "bad", "total": "42.50"})),
        )))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/ocr"))
        .respond_with(ResponseTemplate::new(200).set_body_json(output(
            "mistral-ocr",
            Some(json!({"issued": "2026-11-14", "total": "42.50"})),
        )))
        .mount(&server)
        .await;

    let model = configured(&server, "mistral:mistral-ocr-latest")
        .with_api(Api::MistralOcr)
        .with_setting("pages", json!([0]))
        .with_validation_retries(1);

    let res = extract_value(
        &schema,
        model,
        b"%PDF-1.7\nscan".as_slice(),
        ExtractOptions::new().with_instructions("Read carefully."),
    )
    .await
    .unwrap();
    assert_eq!(res["issued"], "2026-11-14");

    let bodies = recorded_bodies(&server).await;
    assert_eq!(bodies.len(), 2);
    assert_eq!(bodies[0]["pages"], json!([0]));
    assert_eq!(bodies[0]["include_image_base64"], false);
    assert_eq!(bodies[0]["bbox_annotation_format"], Value::Null);
    assert!(
        bodies[1]["document_annotation_prompt"]
            .as_str()
            .unwrap()
            .contains("Correct these output validation errors")
    );

    // Only PDF accepted
    let err = extract_with_options::<Invoice>(
        configured(&server, "mistral:mistral-ocr-latest").with_api(Api::MistralOcr),
        b"text".as_slice(),
        ExtractOptions::new().with_media_type("text/plain"),
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("only PDF"));
}

#[tokio::test]
async fn test_refusals_incomplete_thoughts_and_chat_natural_stop() {
    let flawed = [
        (
            Api::Responses,
            None,
            json!({"status": "incomplete", "output": []}),
        ),
        (
            Api::Responses,
            None,
            json!({"status": "completed", "output": [{"type": "reasoning"}]}),
        ),
        (
            Api::Responses,
            None,
            json!({"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal"}]}]}),
        ),
        (
            Api::Responses,
            None,
            json!({"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "not JSON"}]}]}),
        ),
        (
            Api::Chat,
            None,
            json!({"choices": [{"finish_reason": "stop", "message": {"content": "plain text"}}]}),
        ),
        (
            Api::Chat,
            None,
            json!({"choices": [{"finish_reason": "tool_calls", "message": {"refusal": "nope", "tool_calls": []}}]}),
        ),
        (
            Api::Chat,
            None,
            json!({"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [{"function": {"name": "wrong"}}]}}]}),
        ),
        (
            Api::Chat,
            Some(OutputMode::JsonSchema),
            json!({"choices": [{"finish_reason": "length", "message": {"content": "{\"total\":1}"}}]}),
        ),
        (
            Api::Chat,
            Some(OutputMode::JsonSchema),
            json!({"choices": [{"finish_reason": "stop", "message": {"content": null}}]}),
        ),
        (
            Api::Anthropic,
            None,
            json!({"stop_reason": "max_tokens", "content": []}),
        ),
        (
            Api::Anthropic,
            Some(OutputMode::JsonSchema),
            json!({"stop_reason": "refusal", "content": []}),
        ),
        (
            Api::Anthropic,
            Some(OutputMode::JsonSchema),
            json!({"stop_reason": "end_turn", "content": [{"type": "text", "text": "{}"}, {"type": "text", "text": "{}"}]}),
        ),
        (
            Api::Bedrock,
            None,
            json!({"stopReason": "guardrail_intervened", "output": {"message": {"content": []}}}),
        ),
        (
            Api::Gemini,
            None,
            json!({"candidates": [{"finishReason": "MAX_TOKENS"}]}),
        ),
        (
            Api::Ollama,
            None,
            json!({"done": false, "message": {"content": "{}"}}),
        ),
        (
            Api::Ollama,
            None,
            json!({"done": true, "done_reason": "length", "message": {"content": "{}"}}),
        ),
        (
            Api::Cohere,
            None,
            json!({"finish_reason": "MAX_TOKENS", "message": {"content": []}}),
        ),
    ];

    for (api, mode, payload) in flawed {
        let server = MockServer::start().await;
        mount_json(&server, 200, payload).await;
        let mut model = custom(&server, api).with_validation_retries(1);
        if let Some(m) = mode {
            model = model.with_output_mode(m);
        }
        let err = extract::<Invoice>(model, b"text".as_slice(), None)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::SchemaValidation { .. }), "api={api}");
        // Refusals / incomplete / empty outputs must fail immediately without validation retry.
        assert_eq!(
            server.received_requests().await.unwrap().len(),
            1,
            "no retry for api={api}"
        );
    }

    // Chat accepts forced tool call with finish_reason="stop"
    let server = MockServer::start().await;
    let mut chat_resp = output("chat", None);
    chat_resp["choices"][0]["finish_reason"] = json!("stop");
    mount_json(&server, 200, chat_resp).await;
    let inv: Invoice = extract(
        configured(&server, "openai-chat:test"),
        b"text".as_slice(),
        None,
    )
    .await
    .unwrap();
    assert_eq!(inv.total, 42.0);

    // Gemini skips thought parts
    let server = MockServer::start().await;
    mount_json(
        &server,
        200,
        json!({
            "candidates": [{
                "finishReason": "STOP",
                "content": {"parts": [
                    {"thought": true, "text": "private thoughts"},
                    {"text": "{\"total\": 42}"}
                ]}
            }]
        }),
    )
    .await;
    let inv: Invoice = extract(configured(&server, "google:test"), b"text".as_slice(), None)
        .await
        .unwrap();
    assert_eq!(inv.total, 42.0);
}
