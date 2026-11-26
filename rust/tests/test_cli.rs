mod common;

use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::{Value, json};
use tempfile::tempdir;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use common::{mount_json, output};

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_openextract"))
}

#[test]
fn test_list_providers_outputs_valid_json() {
    let assert = bin()
        .arg("--list-providers")
        .env("OPENAI_API_KEY", "dummy-key")
        .assert()
        .success();

    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    assert!(!stdout.contains("dummy-key"));
    let list: Vec<Value> = serde_json::from_str(&stdout).unwrap();
    let openai = list.iter().find(|p| p["prefix"] == "openai").unwrap();
    assert_eq!(openai["api"], "responses");
    assert_eq!(openai["key_env_set"], true);
}

#[test]
fn test_field_shorthand_print_schema() {
    let assert = bin()
        .args([
            "--print-schema",
            "-f",
            "vendor:string=Seller, not buyer.",
            "-f",
            "total:number",
            "-f",
            "currency:string?",
            "-f",
            "due_date:date?",
            "-f",
            "tags:string[]",
            "-f",
            "kind:enum(invoice,receipt,contract)",
            "-f",
            "is_paid:bool",
        ])
        .assert()
        .success();

    let schema: Value = serde_json::from_slice(&assert.get_output().stdout).unwrap();
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["additionalProperties"], false);
    assert_eq!(
        schema["properties"]["vendor"],
        json!({"type": "string", "description": "Seller, not buyer."})
    );
    assert_eq!(
        schema["properties"]["currency"],
        json!({"anyOf": [{"type": "string"}, {"type": "null"}]})
    );
    assert_eq!(
        schema["properties"]["due_date"],
        json!({"anyOf": [{"type": "string", "format": "date"}, {"type": "null"}]})
    );
    assert_eq!(
        schema["properties"]["tags"],
        json!({"type": "array", "items": {"type": "string"}})
    );
    assert_eq!(
        schema["properties"]["kind"],
        json!({"type": "string", "enum": ["invoice", "receipt", "contract"]})
    );
    assert_eq!(schema["properties"]["is_paid"], json!({"type": "boolean"}));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_cli_end_to_end_file_stdin_url_and_text() {
    let server = MockServer::start().await;
    let extracted = json!({"vendor": "Acme Paper Co.", "total": 120.0, "currency": "USD"});
    mount_json(&server, 200, output("responses", Some(extracted.clone()))).await;

    // 1. Local PDF fixture shared with the Python suite. The published crate
    // does not contain that tree, so a packaged `cargo test` uses a tiny PDF.
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let shared = repo_root.join("tests/fixtures/invoices/invoice.pdf");
    let scratch = tempdir().unwrap();
    let pdf_fixture = if shared.is_file() {
        shared
    } else if repo_root.join("pyproject.toml").is_file() {
        panic!("shared invoice fixture missing");
    } else {
        let path = scratch.path().join("invoice.pdf");
        std::fs::write(&path, b"%PDF-1.1\n").unwrap();
        path
    };

    let out = bin()
        .args([
            "-m",
            "openai:gpt-6.1-sol",
            "--base-url",
            &server.uri(),
            "--api-key",
            "test-key",
            "-f",
            "vendor:string",
            "-f",
            "total:number",
            "-f",
            "currency:string?",
            "--compact",
            pdf_fixture.to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let text = String::from_utf8(out).unwrap();
    assert_eq!(
        text.lines().count(),
        1,
        "compact output is single-line JSON"
    );
    assert_eq!(serde_json::from_str::<Value>(&text).unwrap(), extracted);

    // 2. Stdin piped bytes + --schema file + -o output file
    let dir = tempdir().unwrap();
    let schema_file = dir.path().join("schema.json");
    std::fs::write(
        &schema_file,
        serde_json::to_string(&json!({
            "type": "object",
            "properties": {
                "vendor": {"type": "string"},
                "total": {"type": "number"},
                "currency": {"type": ["string", "null"]}
            },
            "required": ["vendor", "total", "currency"]
        }))
        .unwrap(),
    )
    .unwrap();
    let out_file = dir.path().join("result.json");

    bin()
        .args([
            "-m",
            "openai:gpt-6.1-sol",
            "--base-url",
            &server.uri(),
            "--api-key",
            "test-key",
            "--schema",
            &format!("@{}", schema_file.display()),
            "-o",
            out_file.to_str().unwrap(),
        ])
        .write_stdin("Seller: Acme Paper Co.\nTotal due: 120.00 USD\n")
        .assert()
        .success()
        .stdout(predicate::str::is_empty());

    let written: Value =
        serde_json::from_str(&std::fs::read_to_string(&out_file).unwrap()).unwrap();
    assert_eq!(written, extracted);

    // 3. --text never reads a matching file name on disk
    bin()
        .args([
            "-m",
            "openai:gpt-6.1-sol",
            "--base-url",
            &server.uri(),
            "--api-key",
            "test-key",
            "-f",
            "vendor:string",
            "-f",
            "total:number",
            "-f",
            "currency:string?",
            "--text",
            "Cargo.toml",
        ])
        .assert()
        .success();
    let reqs = server.received_requests().await.unwrap();
    let last_body: Value = serde_json::from_slice(&reqs.last().unwrap().body).unwrap();
    assert_eq!(
        last_body["input"][0]["content"][1],
        json!({"type": "input_text", "text": "Cargo.toml"})
    );

    // 4. URL source fetched without model credentials
    let doc_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/doc.txt"))
        .respond_with(ResponseTemplate::new(200).set_body_string("Remote invoice"))
        .mount(&doc_server)
        .await;

    bin()
        .args([
            "-m",
            "openai:gpt-6.1-sol",
            "--base-url",
            &server.uri(),
            "--api-key",
            "secret-model-key",
            "-f",
            "vendor:string",
            "-f",
            "total:number",
            "-f",
            "currency:string?",
            &format!("{}/doc.txt", doc_server.uri()),
        ])
        .assert()
        .success();
    let doc_reqs = doc_server.received_requests().await.unwrap();
    assert!(doc_reqs[0].headers.get("authorization").is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_cli_exit_codes_and_clean_stderr() {
    // Exit 2: invalid field spec
    bin()
        .args(["--print-schema", "-f", "bad:unknown_type"])
        .assert()
        .code(2)
        .stdout(predicate::str::is_empty())
        .stderr(predicate::str::contains("Unsupported field type"));

    // Exit 3: missing API key
    bin()
        .args([
            "-m",
            "openai:gpt-6.1-sol",
            "-f",
            "total:number",
            "--text",
            "42",
        ])
        .env_remove("OPENAI_API_KEY")
        .assert()
        .code(3)
        .stdout(predicate::str::is_empty())
        .stderr(predicate::str::contains("OPENAI_API_KEY"));

    // Exit 4: HTTP 401 provider error without secret leak
    let server = MockServer::start().await;
    mount_json(
        &server,
        401,
        json!({"error": {"code": "authentication_error", "message": "Invalid key super-secret-key"}}),
    )
    .await;
    bin()
        .args([
            "-m",
            "openai:gpt-6.1-sol",
            "--base-url",
            &server.uri(),
            "--api-key",
            "super-secret-key",
            "--max-retries",
            "0",
            "-f",
            "total:number",
            "--text",
            "42",
        ])
        .assert()
        .code(4)
        .stdout(predicate::str::is_empty())
        .stderr(predicate::str::contains("HTTP 401"))
        .stderr(predicate::str::contains("Next step:"))
        .stderr(predicate::str::contains("super-secret-key").not());

    // Exit 5: Schema validation failure
    let server = MockServer::start().await;
    mount_json(
        &server,
        200,
        output("responses", Some(json!({"total": "not-a-number"}))),
    )
    .await;
    bin()
        .args([
            "-m",
            "openai:gpt-6.1-sol",
            "--base-url",
            &server.uri(),
            "--api-key",
            "test-key",
            "--validation-retries",
            "0",
            "-f",
            "total:number",
            "--text",
            "42",
        ])
        .assert()
        .code(5)
        .stdout(predicate::str::is_empty())
        .stderr(predicate::str::contains("- total:"))
        .stderr(predicate::str::contains("not-a-number").not());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_cli_decisions_openrouter_metadata_and_advanced_flags() {
    // OpenAI Decisions via --field enum & bool
    let server = MockServer::start().await;
    mount_json(
        &server,
        200,
        json!({
            "answers": [
                {"type": "choice", "name": "doc_type", "choice": "receipt"},
                {"type": "predicate", "name": "is_paid", "probability": 0.9}
            ]
        }),
    )
    .await;
    let out = bin()
        .args([
            "-m",
            "openai-decisions:gpt-6-luna",
            "--base-url",
            &server.uri(),
            "--api-key",
            "test-key",
            "-f",
            "doc_type:enum(invoice,receipt,contract)=Document kind",
            "-f",
            "is_paid:bool=Paid in full?",
            "--text",
            "Paid receipt",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let val: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(val, json!({"doc_type": "receipt", "is_paid": true}));

    // OpenRouter Extractions with --include-metadata, --settings, and --header
    let server = MockServer::start().await;
    mount_json(
        &server,
        200,
        json!({
            "truncated": false,
            "usage": {"cost": 0.001},
            "answers": {"vendor": {"value": "Acme", "span": [0, 4]}}
        }),
    )
    .await;
    let out = bin()
        .args([
            "-m",
            "openrouter:cloudflare/clairic",
            "--base-url",
            &server.uri(),
            "--api-key",
            "test-key",
            "-H",
            "X-Trace: trace-1",
            "--settings",
            r#"{"reference_date":"2026-11-25"}"#,
            "--include-metadata",
            "-f",
            "vendor:string",
            "--text",
            "Acme invoice",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let val: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(val["data"], json!({"vendor": "Acme"}));
    assert_eq!(val["answers"]["vendor"]["span"], json!([0, 4]));
    assert_eq!(val["metadata"]["truncated"], false);
    let reqs = server.received_requests().await.unwrap();
    assert_eq!(reqs[0].headers.get("x-trace").unwrap(), "trace-1");
}
