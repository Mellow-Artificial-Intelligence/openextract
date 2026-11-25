mod common;

use std::collections::HashMap;

use openextract::{Api, Error, Model, OutputMode, extract, resolve_with_env};
use serde_json::json;
use wiremock::MockServer;

use common::{Invoice, mount_json, output, recorded_bodies};

const OPENAI: &str = "https://api.openai.com/v1";
const GOOGLE: &str = "https://generativelanguage.googleapis.com/v1beta";
const OPENROUTER: &str = "https://openrouter.ai/api/v1";
const OPENROUTER_ALPHA: &str = "https://openrouter.ai/api/alpha";

fn env_map(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> + use<> {
    let map: HashMap<String, String> = pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
    move |k: &str| map.get(k).cloned()
}

#[test]
fn test_provider_defaults() {
    let cases = [
        ("openai:x", Api::Responses, OPENAI, "OPENAI_API_KEY"),
        (
            "openai-responses:x",
            Api::Responses,
            OPENAI,
            "OPENAI_API_KEY",
        ),
        ("openai-chat:x", Api::Chat, OPENAI, "OPENAI_API_KEY"),
        (
            "openai-decisions:x",
            Api::OpenAiDecisions,
            OPENAI,
            "OPENAI_API_KEY",
        ),
        (
            "anthropic:x",
            Api::Anthropic,
            "https://api.anthropic.com/v1",
            "ANTHROPIC_API_KEY",
        ),
        ("google:x", Api::Gemini, GOOGLE, "GOOGLE_API_KEY"),
        ("gemini:x", Api::Gemini, GOOGLE, "GEMINI_API_KEY"),
        ("google-gla:x", Api::Gemini, GOOGLE, "GOOGLE_API_KEY"),
        (
            "groq:x",
            Api::Chat,
            "https://api.groq.com/openai/v1",
            "GROQ_API_KEY",
        ),
        (
            "mistral:x",
            Api::Chat,
            "https://api.mistral.ai/v1",
            "MISTRAL_API_KEY",
        ),
        ("xai:x", Api::Chat, "https://api.x.ai/v1", "XAI_API_KEY"),
        (
            "cerebras:x",
            Api::Chat,
            "https://api.cerebras.ai/v1",
            "CEREBRAS_API_KEY",
        ),
        (
            "huggingface:x",
            Api::Chat,
            "https://router.huggingface.co/v1",
            "HF_TOKEN",
        ),
        (
            "cohere:x",
            Api::Cohere,
            "https://api.cohere.com/v2",
            "CO_API_KEY",
        ),
    ];

    for (name, expected_api, expected_url, key_var) in cases {
        let resolved =
            resolve_with_env(Model::new(name), env_map(&[(key_var, "env-key")])).unwrap();
        assert_eq!(resolved.api, expected_api, "api for {name}");
        assert_eq!(resolved.base_url, expected_url, "url for {name}");
        let (header, expected_val) = match expected_api {
            Api::Anthropic => ("x-api-key", "env-key"),
            Api::Gemini | Api::GeminiDecisions => ("x-goog-api-key", "env-key"),
            _ => ("authorization", "Bearer env-key"),
        };
        assert_eq!(resolved.headers[header], expected_val, "header for {name}");
    }
}

#[test]
fn test_openrouter_routes_by_model_name() {
    let cases = [
        ("openrouter:x", Api::Chat, OPENROUTER),
        (
            "openrouter:typesafe/jev-1.14",
            Api::OpenRouterDecisions,
            OPENROUTER_ALPHA,
        ),
        (
            "openrouter:~typesafe/jev-latest",
            Api::OpenRouterDecisions,
            OPENROUTER_ALPHA,
        ),
        (
            "openrouter:cloudflare/clairic",
            Api::OpenRouterExtractions,
            OPENROUTER_ALPHA,
        ),
        (
            "openrouter:inception/mercury-decide",
            Api::OpenRouterExtractions,
            OPENROUTER_ALPHA,
        ),
        (
            "openrouter:levanto/sage-2",
            Api::OpenRouterExtractions,
            OPENROUTER_ALPHA,
        ),
        (
            "openrouter:scaledown/extract",
            Api::OpenRouterExtractions,
            OPENROUTER_ALPHA,
        ),
        (
            "openrouter-extractions:custom/preview",
            Api::OpenRouterExtractions,
            OPENROUTER_ALPHA,
        ),
    ];

    let env = env_map(&[("OPENROUTER_API_KEY", "env-key")]);
    for (name, expected_api, expected_url) in cases {
        let resolved = resolve_with_env(Model::new(name), &env).unwrap();
        assert_eq!(resolved.api, expected_api);
        assert_eq!(resolved.base_url, expected_url);
        assert_eq!(resolved.model_name, name.split_once(':').unwrap().1);
        assert_eq!(resolved.headers["authorization"], "Bearer env-key");
    }
}

#[test]
fn test_explicit_api_overrides_openrouter_routing() {
    let cases = [
        ("openrouter:typesafe/model", Api::Chat, OPENROUTER),
        (
            "openrouter:inception/mercury-decide",
            Api::OpenRouterDecisions,
            OPENROUTER_ALPHA,
        ),
        (
            "openrouter:custom/preview",
            Api::OpenRouterExtractions,
            OPENROUTER_ALPHA,
        ),
    ];
    for (name, api, url) in cases {
        let model = Model::new(name).with_api(api).with_api_key("test-key");
        let resolved = resolve_with_env(model, env_map(&[])).unwrap();
        assert_eq!(resolved.api, api);
        assert_eq!(resolved.base_url, url);
    }
}

#[test]
fn test_openrouter_alpha_translates_env_base_url() {
    for base in [
        "https://proxy.example/api/v1",
        "https://proxy.example/api/alpha/",
    ] {
        let env = env_map(&[
            ("OPENROUTER_BASE_URL", base),
            ("OPENROUTER_API_KEY", "env-key"),
        ]);
        let r1 = resolve_with_env(Model::new("openrouter:cloudflare/clairic"), &env).unwrap();
        assert_eq!(r1.base_url, "https://proxy.example/api/alpha");
        let r2 = resolve_with_env(Model::new("openrouter:x"), &env).unwrap();
        assert_eq!(r2.base_url, base.trim_end_matches('/'));
    }
}

#[test]
fn test_bedrock_region_and_key() {
    for var in ["AWS_REGION", "AWS_DEFAULT_REGION"] {
        let env = env_map(&[
            (var, "us-west-2"),
            ("AWS_BEARER_TOKEN_BEDROCK", "bedrock-key"),
        ]);
        let resolved = resolve_with_env(Model::new("bedrock:us.anthropic.model"), env).unwrap();
        assert_eq!(
            resolved.base_url,
            "https://bedrock-runtime.us-west-2.amazonaws.com"
        );
        assert_eq!(resolved.headers["authorization"], "Bearer bedrock-key");
    }
}

#[test]
fn test_env_overrides_and_explicit_precedence() {
    let env = env_map(&[
        ("OPENAI_BASE_URL", "https://proxy.example/v1/"),
        ("OPENAI_API_KEY", "env-key"),
    ]);
    let r = resolve_with_env(Model::new("openai:test"), &env).unwrap();
    assert_eq!(r.base_url, "https://proxy.example/v1");

    let explicit = Model::new("openai:test")
        .with_base_url("https://explicit.example/v1/")
        .with_api_key("explicit-key");
    let r = resolve_with_env(explicit, &env).unwrap();
    assert_eq!(r.base_url, "https://explicit.example/v1");
    assert_eq!(r.headers["authorization"], "Bearer explicit-key");
}

#[test]
fn test_custom_headers_precedence_and_secrets_redacted_in_debug() {
    let model = Model::new("anthropic:test")
        .with_api_key("secret-key")
        .with_header("X-API-Key", "header-secret")
        .with_header("Anthropic-Version", "custom-version");
    let resolved = resolve_with_env(model.clone(), env_map(&[])).unwrap();
    assert_eq!(resolved.headers["x-api-key"], "header-secret");
    assert_eq!(resolved.headers["anthropic-version"], "custom-version");
    assert!(!format!("{model:?}").contains("secret"));
    assert!(!format!("{resolved:?}").contains("secret"));
}

#[test]
fn test_custom_protocol_endpoint_without_auth() {
    let model = Model::new("local:model")
        .with_api(Api::Chat)
        .with_base_url("http://localhost:8000/v1");
    let resolved = resolve_with_env(model, env_map(&[])).unwrap();
    assert_eq!(resolved.model_name, "model");
    assert!(!resolved.headers.contains_key("authorization"));
}

#[tokio::test]
async fn test_custom_chat_endpoint_and_settings() {
    let server = MockServer::start().await;
    mount_json(&server, 200, output("chat", None)).await;
    let model = Model::new("custom:tenant/model:latest")
        .with_api(Api::Chat)
        .with_base_url(format!("{}/custom/v1", server.uri()))
        .with_header("X-Tenant", "tenant")
        .with_setting("temperature", json!(0))
        .with_max_retries(0)
        .with_validation_retries(0);

    let inv: Invoice = extract(model, b"text".as_slice(), None).await.unwrap();
    assert_eq!(inv.total, 42.0);
    let reqs = server.received_requests().await.unwrap();
    assert_eq!(reqs[0].url.path(), "/custom/v1/chat/completions");
    assert_eq!(reqs[0].headers.get("x-tenant").unwrap(), "tenant");
    let bodies = recorded_bodies(&server).await;
    assert_eq!(bodies[0]["model"], "tenant/model:latest");
    assert_eq!(bodies[0]["temperature"], 0);
}

#[test]
fn test_output_mode_scoping_and_validation() {
    for mode in [OutputMode::Tool, OutputMode::JsonSchema] {
        let m = Model::new("openai-chat:test")
            .with_api_key("k")
            .with_output_mode(mode);
        assert_eq!(
            resolve_with_env(m, env_map(&[]))
                .unwrap()
                .config
                .output_mode,
            Some(mode)
        );
    }

    for name in [
        "openai:test",
        "google:test",
        "openai-decisions:test",
        "openrouter:typesafe/model",
    ] {
        let m = Model::new(name)
            .with_api_key("k")
            .with_output_mode(OutputMode::JsonSchema);
        let err = resolve_with_env(m, env_map(&[])).unwrap_err();
        assert!(matches!(err, Error::InvalidConfig(_)));
        assert!(err.to_string().contains("only configurable"));
    }
}

#[test]
fn test_invalid_limits_and_urls_and_models() {
    for bad_timeout in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let err = resolve_with_env(
            Model::new("openai:test").with_timeout(bad_timeout),
            env_map(&[]),
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidConfig(_)));
    }

    for bad_url in [
        "ftp://example.com",
        "https:///",
        "https://user:secret@example.com/v1",
        "https://example.com?token=secret",
        "https://example.com#fragment",
        "https://example.com:invalid",
    ] {
        let m = Model::new("custom:model")
            .with_api(Api::Chat)
            .with_base_url(bad_url);
        let err = resolve_with_env(m, env_map(&[])).unwrap_err();
        assert!(err.to_string().contains("base_url"), "{bad_url}: {err}");
    }

    for bad_name in ["model", "", ":model", "openai:"] {
        let err = resolve_with_env(Model::new(bad_name), env_map(&[])).unwrap_err();
        assert!(err.to_string().contains("provider:model"));
    }

    for m in [
        Model::new("unknown:model"),
        Model::new("custom:model").with_api(Api::Chat),
        Model::new("custom:model").with_base_url("https://example.com"),
    ] {
        let err = resolve_with_env(m, env_map(&[])).unwrap_err();
        assert!(err.to_string().contains("Unknown provider"));
    }

    assert!("invalid".parse::<Api>().is_err());
    assert!("invalid".parse::<OutputMode>().is_err());

    for name in ["openai:test", "anthropic:test", "gemini:test"] {
        let err = resolve_with_env(Model::new(name), env_map(&[])).unwrap_err();
        assert!(matches!(err, Error::Extraction { .. }));
        assert!(err.to_string().contains("API_KEY"));
    }

    let err =
        resolve_with_env(Model::new("bedrock:model").with_api_key("k"), env_map(&[])).unwrap_err();
    assert!(matches!(err, Error::Extraction { .. }));
    assert!(err.to_string().contains("AWS_REGION"));
}
