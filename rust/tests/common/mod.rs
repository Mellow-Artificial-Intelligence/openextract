#![allow(dead_code)]

use openextract::{Api, Model};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Invoice {
    pub total: f64,
}

pub fn configured(server: &MockServer, name: &str) -> Model {
    Model::new(name)
        .with_base_url(server.uri())
        .with_api_key("test-key")
        .with_max_retries(0)
        .with_validation_retries(0)
}

pub fn custom(server: &MockServer, api: Api) -> Model {
    Model::new("custom:test")
        .with_api(api)
        .with_base_url(server.uri())
        .with_api_key("test-key")
        .with_max_retries(0)
        .with_validation_retries(0)
}

pub fn output(api: &str, value: Option<Value>) -> Value {
    let val = value.unwrap_or_else(|| json!({"total": 42.0}));
    let text = serde_json::to_string(&val).unwrap();
    let call = json!({"type": "function", "function": {"name": "extract", "arguments": text}});
    let tool_use = json!({"name": "extract", "input": val});
    match api {
        "responses" => json!({
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
        }),
        "chat" => json!({
            "choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [call]}}]
        }),
        "chat-json" => json!({
            "choices": [{"finish_reason": "stop", "message": {"content": text}}]
        }),
        "anthropic" => json!({
            "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "name": "extract", "input": val}],
        }),
        "anthropic-json" => json!({
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": text}],
        }),
        "gemini" => json!({
            "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": text}]}}]
        }),
        "ollama" => json!({
            "done": true,
            "done_reason": "stop",
            "message": {"content": text},
        }),
        "cohere" => json!({
            "finish_reason": "COMPLETE",
            "message": {"content": [{"type": "text", "text": text}]},
        }),
        "bedrock" => json!({
            "stopReason": "tool_use",
            "output": {"message": {"content": [{"toolUse": tool_use}]}},
        }),
        "mistral-ocr" => json!({"document_annotation": text}),
        other => panic!("unknown output api: {other}"),
    }
}

pub async fn mount_json(server: &MockServer, status: u16, body: Value) {
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(status).set_body_json(body))
        .mount(server)
        .await;
}

pub async fn recorded_bodies(server: &MockServer) -> Vec<Value> {
    server
        .received_requests()
        .await
        .unwrap()
        .into_iter()
        .filter(|r| r.method == "POST")
        .map(|r| serde_json::from_slice(&r.body).unwrap())
        .collect()
}
