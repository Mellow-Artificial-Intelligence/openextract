//! Model configuration and provider defaults, without provider SDKs.

use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use url::Url;

use crate::error::{Error, Result};

/// Supported HTTP provider APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Api {
    #[serde(rename = "responses")]
    Responses,
    #[serde(rename = "chat")]
    Chat,
    #[serde(rename = "anthropic")]
    Anthropic,
    #[serde(rename = "gemini")]
    Gemini,
    #[serde(rename = "ollama")]
    Ollama,
    #[serde(rename = "cohere")]
    Cohere,
    #[serde(rename = "bedrock")]
    Bedrock,
    #[serde(rename = "mistral-ocr")]
    MistralOcr,
    #[serde(rename = "gemini-decisions")]
    GeminiDecisions,
    #[serde(rename = "openai-decisions")]
    OpenAiDecisions,
    #[serde(rename = "openrouter-decisions")]
    OpenRouterDecisions,
    #[serde(rename = "openrouter-extractions")]
    OpenRouterExtractions,
}

impl Api {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Responses => "responses",
            Self::Chat => "chat",
            Self::Anthropic => "anthropic",
            Self::Gemini => "gemini",
            Self::Ollama => "ollama",
            Self::Cohere => "cohere",
            Self::Bedrock => "bedrock",
            Self::MistralOcr => "mistral-ocr",
            Self::GeminiDecisions => "gemini-decisions",
            Self::OpenAiDecisions => "openai-decisions",
            Self::OpenRouterDecisions => "openrouter-decisions",
            Self::OpenRouterExtractions => "openrouter-extractions",
        }
    }

    pub fn is_decisions(&self) -> bool {
        matches!(
            self,
            Self::GeminiDecisions | Self::OpenAiDecisions | Self::OpenRouterDecisions
        )
    }
}

impl fmt::Display for Api {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Api {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "responses" => Ok(Self::Responses),
            "chat" => Ok(Self::Chat),
            "anthropic" => Ok(Self::Anthropic),
            "gemini" => Ok(Self::Gemini),
            "ollama" => Ok(Self::Ollama),
            "cohere" => Ok(Self::Cohere),
            "bedrock" => Ok(Self::Bedrock),
            "mistral-ocr" => Ok(Self::MistralOcr),
            "gemini-decisions" => Ok(Self::GeminiDecisions),
            "openai-decisions" => Ok(Self::OpenAiDecisions),
            "openrouter-decisions" => Ok(Self::OpenRouterDecisions),
            "openrouter-extractions" => Ok(Self::OpenRouterExtractions),
            other => Err(Error::InvalidConfig(format!(
                "Unsupported HTTP API: '{other}'."
            ))),
        }
    }
}

/// Structured output mode for Chat Completions and Anthropic Messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputMode {
    #[serde(rename = "tool")]
    Tool,
    #[serde(rename = "json_schema")]
    JsonSchema,
}

impl OutputMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Tool => "tool",
            Self::JsonSchema => "json_schema",
        }
    }
}

impl fmt::Display for OutputMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for OutputMode {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "tool" => Ok(Self::Tool),
            "json_schema" => Ok(Self::JsonSchema),
            _ => Err(Error::InvalidConfig(
                "output_mode must be 'tool' or 'json_schema'.".to_string(),
            )),
        }
    }
}

/// Configure a model, optionally overriding its HTTP API and connection settings.
///
/// `base_url` includes the API version (for example, `https://host/v1`).
/// `settings` contains provider-native generation parameters. Request structure
/// and structured-output fields are managed by openextract and cannot be overridden.
#[derive(Clone, PartialEq)]
pub struct Model {
    pub name: String,
    pub api: Option<Api>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub headers: BTreeMap<String, String>,
    pub settings: Map<String, Value>,
    pub timeout: f64,
    pub max_retries: u32,
    pub validation_retries: u32,
    pub output_mode: Option<OutputMode>,
}

impl fmt::Debug for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Deliberately omit `api_key` and `headers` to prevent credential leaks in logs.
        f.debug_struct("Model")
            .field("name", &self.name)
            .field("api", &self.api)
            .field("base_url", &self.base_url)
            .field("settings", &self.settings)
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .field("validation_retries", &self.validation_retries)
            .field("output_mode", &self.output_mode)
            .finish()
    }
}

impl Model {
    /// Create a model configuration from a `"provider:model"` identifier.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            api: None,
            base_url: None,
            api_key: None,
            headers: BTreeMap::new(),
            settings: Map::new(),
            timeout: 60.0,
            max_retries: 2,
            validation_retries: 1,
            output_mode: None,
        }
    }

    pub fn with_api(mut self, api: Api) -> Self {
        self.api = Some(api);
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    pub fn with_headers<I, K, V>(mut self, headers: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        for (k, v) in headers {
            self.headers.insert(k.into(), v.into());
        }
        self
    }

    pub fn with_setting(mut self, key: impl Into<String>, value: Value) -> Self {
        self.settings.insert(key.into(), value);
        self
    }

    pub fn with_settings(mut self, settings: Map<String, Value>) -> Self {
        self.settings = settings;
        self
    }

    pub fn with_timeout(mut self, seconds: f64) -> Self {
        self.timeout = seconds;
        self
    }

    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    pub fn with_validation_retries(mut self, validation_retries: u32) -> Self {
        self.validation_retries = validation_retries;
        self
    }

    pub fn with_output_mode(mut self, output_mode: OutputMode) -> Self {
        self.output_mode = Some(output_mode);
        self
    }

    fn validate_limits(&self) -> Result<()> {
        if !self.timeout.is_finite() || self.timeout <= 0.0 {
            return Err(Error::InvalidConfig(
                "timeout must be a positive, finite number of seconds.".to_string(),
            ));
        }
        Ok(())
    }
}

impl From<&str> for Model {
    fn from(name: &str) -> Self {
        Self::new(name)
    }
}

impl From<String> for Model {
    fn from(name: String) -> Self {
        Self::new(name)
    }
}

/// A model configuration combined with its provider's defaults and environment.
#[derive(Clone)]
pub struct ResolvedModel {
    pub config: Model,
    pub model_name: String,
    pub api: Api,
    pub base_url: String,
    pub headers: HeaderMap,
}

impl fmt::Debug for ResolvedModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Deliberately omit `headers` to prevent credential leaks in logs.
        f.debug_struct("ResolvedModel")
            .field("config", &self.config)
            .field("model_name", &self.model_name)
            .field("api", &self.api)
            .field("base_url", &self.base_url)
            .finish()
    }
}

/// Built-in provider metadata exposed for CLI discovery and resolution.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct ProviderInfo {
    pub prefix: &'static str,
    pub aliases: &'static [&'static str],
    pub api: Api,
    pub base_url: &'static str,
    pub key_env: Option<&'static str>,
    pub url_env: &'static str,
    pub inputs: &'static str,
}

const OPENAI_URL: &str = "https://api.openai.com/v1";
const GOOGLE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
const OPENROUTER_URL: &str = "https://openrouter.ai/api/v1";

const PROVIDERS: &[ProviderInfo] = &[
    ProviderInfo {
        prefix: "openai",
        aliases: &["openai-responses"],
        api: Api::Responses,
        base_url: OPENAI_URL,
        key_env: Some("OPENAI_API_KEY"),
        url_env: "OPENAI_BASE_URL",
        inputs: "Text, images, PDF, Word, RTF, ODT, PowerPoint, Excel",
    },
    ProviderInfo {
        prefix: "openai-chat",
        aliases: &[],
        api: Api::Chat,
        base_url: OPENAI_URL,
        key_env: Some("OPENAI_API_KEY"),
        url_env: "OPENAI_BASE_URL",
        inputs: "Text, images, PDF, MP3, WAV",
    },
    ProviderInfo {
        prefix: "openai-decisions",
        aliases: &[],
        api: Api::OpenAiDecisions,
        base_url: OPENAI_URL,
        key_env: Some("OPENAI_API_KEY"),
        url_env: "OPENAI_BASE_URL",
        inputs: "Text, images (bool/enum schemas, 1-400 fields)",
    },
    ProviderInfo {
        prefix: "anthropic",
        aliases: &[],
        api: Api::Anthropic,
        base_url: "https://api.anthropic.com/v1",
        key_env: Some("ANTHROPIC_API_KEY"),
        url_env: "ANTHROPIC_BASE_URL",
        inputs: "Text, PNG, JPEG, GIF, WebP, PDF",
    },
    ProviderInfo {
        prefix: "google",
        aliases: &["google-gla", "gemini"],
        api: Api::Gemini,
        base_url: GOOGLE_URL,
        key_env: Some("GOOGLE_API_KEY"),
        url_env: "GOOGLE_BASE_URL",
        inputs: "Text, images, PDF, audio, video, Office, OpenDocument",
    },
    ProviderInfo {
        prefix: "gemini-decisions",
        aliases: &[],
        api: Api::GeminiDecisions,
        base_url: GOOGLE_URL,
        key_env: Some("GOOGLE_API_KEY"),
        url_env: "GOOGLE_BASE_URL",
        inputs: "Text, images, PDF (bool/enum schemas, 1-1000 fields)",
    },
    ProviderInfo {
        prefix: "openrouter",
        aliases: &[],
        api: Api::Chat,
        base_url: OPENROUTER_URL,
        key_env: Some("OPENROUTER_API_KEY"),
        url_env: "OPENROUTER_BASE_URL",
        inputs: "Chat Completions, Extractions, or Decisions by model",
    },
    ProviderInfo {
        prefix: "openrouter-extractions",
        aliases: &[],
        api: Api::OpenRouterExtractions,
        base_url: OPENROUTER_URL,
        key_env: Some("OPENROUTER_API_KEY"),
        url_env: "OPENROUTER_BASE_URL",
        inputs: "Text, PDF, PNG, JPEG, WebP (typed fields + metadata)",
    },
    ProviderInfo {
        prefix: "ollama",
        aliases: &[],
        api: Api::Ollama,
        base_url: "http://localhost:11434",
        key_env: None,
        url_env: "OLLAMA_BASE_URL",
        inputs: "Text, images (local)",
    },
    ProviderInfo {
        prefix: "groq",
        aliases: &[],
        api: Api::Chat,
        base_url: "https://api.groq.com/openai/v1",
        key_env: Some("GROQ_API_KEY"),
        url_env: "GROQ_BASE_URL",
        inputs: "Encoded as openai-chat",
    },
    ProviderInfo {
        prefix: "mistral",
        aliases: &[],
        api: Api::Chat,
        base_url: "https://api.mistral.ai/v1",
        key_env: Some("MISTRAL_API_KEY"),
        url_env: "MISTRAL_BASE_URL",
        inputs: "Encoded as openai-chat; PDF with --api mistral-ocr",
    },
    ProviderInfo {
        prefix: "xai",
        aliases: &[],
        api: Api::Chat,
        base_url: "https://api.x.ai/v1",
        key_env: Some("XAI_API_KEY"),
        url_env: "XAI_BASE_URL",
        inputs: "Encoded as openai-chat",
    },
    ProviderInfo {
        prefix: "cerebras",
        aliases: &[],
        api: Api::Chat,
        base_url: "https://api.cerebras.ai/v1",
        key_env: Some("CEREBRAS_API_KEY"),
        url_env: "CEREBRAS_BASE_URL",
        inputs: "Encoded as openai-chat",
    },
    ProviderInfo {
        prefix: "huggingface",
        aliases: &[],
        api: Api::Chat,
        base_url: "https://router.huggingface.co/v1",
        key_env: Some("HF_TOKEN"),
        url_env: "HF_BASE_URL",
        inputs: "Encoded as openai-chat",
    },
    ProviderInfo {
        prefix: "cohere",
        aliases: &[],
        api: Api::Cohere,
        base_url: "https://api.cohere.com/v2",
        key_env: Some("CO_API_KEY"),
        url_env: "COHERE_BASE_URL",
        inputs: "Text, images",
    },
    ProviderInfo {
        prefix: "bedrock",
        aliases: &[],
        api: Api::Bedrock,
        base_url: "",
        key_env: Some("AWS_BEARER_TOKEN_BEDROCK"),
        url_env: "BEDROCK_BASE_URL",
        inputs: "Text, PNG, JPEG, GIF, WebP, PDF (needs AWS_REGION)",
    },
];

const OPENROUTER_EXTRACTION_MODELS: &[&str] = &[
    "cloudflare/clairic",
    "inception/mercury-decide",
    "levanto/sage-2",
    "scaledown/extract",
];

/// Return the built-in provider table.
pub fn builtin_providers() -> &'static [ProviderInfo] {
    PROVIDERS
}

fn find_provider(prefix: &str) -> Option<&'static ProviderInfo> {
    let canonical = match prefix {
        "openai-responses" => "openai",
        "google-gla" | "gemini" => "google",
        other => other,
    };
    PROVIDERS.iter().find(|p| p.prefix == canonical)
}

/// Fill in a model's API, endpoint, and credentials from its provider prefix and environment.
pub fn resolve(model: impl Into<Model>) -> Result<ResolvedModel> {
    resolve_with_env(model.into(), |key| std::env::var(key).ok())
}

/// Same as [`resolve`], reading environment variables from the supplied closure.
pub fn resolve_with_env<F>(model: Model, env: F) -> Result<ResolvedModel>
where
    F: Fn(&str) -> Option<String>,
{
    model.validate_limits()?;
    let (prefix, name) = match model.name.split_once(':') {
        Some((p, n)) if !p.is_empty() && !n.is_empty() => (p, n),
        _ => {
            return Err(Error::InvalidConfig(
                "Use a model identifier in the form 'provider:model'.".to_string(),
            ));
        }
    };
    let model_name = name.to_string();

    let Some(provider) = find_provider(prefix) else {
        let (Some(explicit_api), Some(explicit_url)) = (model.api, model.base_url.as_deref())
        else {
            return Err(Error::InvalidConfig(format!(
                "Unknown provider '{prefix}'; configure Model(api=..., base_url=...)."
            )));
        };
        let api = resolve_api(&model, prefix, name, explicit_api)?;
        let base_url = checked_url(explicit_url)?;
        let headers = build_headers(&model, api, None, &env)?;
        return Ok(ResolvedModel {
            config: model,
            model_name,
            api,
            base_url,
            headers,
        });
    };

    let api = resolve_api(&model, prefix, name, provider.api)?;
    let raw_url = match &model.base_url {
        Some(url) => url.clone(),
        None => {
            let env_or_default = env(provider.url_env)
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| provider.base_url.to_string());
            default_url(api, &env_or_default, &env)?
        }
    };
    let base_url = checked_url(&raw_url)?;
    let headers = build_headers(&model, api, provider.key_env, &env)?;
    Ok(ResolvedModel {
        config: model,
        model_name,
        api,
        base_url,
        headers,
    })
}

/// Copy caller generation settings into `target`, rejecting managed keys.
pub(crate) fn merge_settings(
    target: &mut Map<String, Value>,
    settings: &Map<String, Value>,
    reserved: &[&str],
    kind: &str,
) -> Result<()> {
    let mut overridden: Vec<&str> = settings
        .keys()
        .filter(|k| reserved.contains(&k.as_str()))
        .map(String::as_str)
        .collect();
    if !overridden.is_empty() {
        overridden.sort_unstable();
        let formatted: Vec<String> = overridden.iter().map(|k| format!("'{k}'")).collect();
        return Err(Error::InvalidConfig(format!(
            "settings cannot override {kind}: [{}].",
            formatted.join(", ")
        )));
    }
    for (k, v) in settings {
        target.insert(k.clone(), v.clone());
    }
    Ok(())
}

fn resolve_api(model: &Model, prefix: &str, name: &str, default: Api) -> Result<Api> {
    let api = if let Some(explicit) = model.api {
        explicit
    } else if prefix == "openrouter" && OPENROUTER_EXTRACTION_MODELS.contains(&name) {
        Api::OpenRouterExtractions
    } else if prefix == "openrouter"
        && (name.starts_with("typesafe/") || name.starts_with("~typesafe/"))
    {
        Api::OpenRouterDecisions
    } else {
        default
    };

    if model.output_mode.is_some() && !matches!(api, Api::Chat | Api::Anthropic) {
        return Err(Error::InvalidConfig(
            "output_mode is only configurable for Chat Completions and Anthropic Messages."
                .to_string(),
        ));
    }
    Ok(api)
}

fn default_url<F>(api: Api, base_url: &str, env: &F) -> Result<String>
where
    F: Fn(&str) -> Option<String>,
{
    if matches!(api, Api::OpenRouterDecisions | Api::OpenRouterExtractions) {
        // OpenRouter serves its alpha APIs beside /v1.
        let trimmed = base_url.trim_end_matches('/');
        let trimmed = trimmed.strip_suffix("/v1").unwrap_or(trimmed);
        let trimmed = trimmed.strip_suffix("/alpha").unwrap_or(trimmed);
        return Ok(format!("{trimmed}/alpha"));
    }
    if api == Api::Bedrock && base_url.is_empty() {
        let region = env("AWS_REGION")
            .filter(|v| !v.is_empty())
            .or_else(|| env("AWS_DEFAULT_REGION").filter(|v| !v.is_empty()))
            .ok_or_else(|| {
                Error::extraction("Set AWS_REGION or configure Model(base_url=...) for Bedrock.")
            })?;
        return Ok(format!("https://bedrock-runtime.{region}.amazonaws.com"));
    }
    Ok(base_url.to_string())
}

fn checked_url(base_url: &str) -> Result<String> {
    let parsed = Url::parse(base_url)
        .map_err(|_| Error::InvalidConfig("base_url must be a valid HTTP(S) URL.".to_string()))?;
    let is_http = matches!(parsed.scheme(), "http" | "https")
        && parsed.host_str().is_some_and(|h| !h.is_empty());
    if !is_http
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        return Err(Error::InvalidConfig(
            "base_url must be an HTTP(S) URL without credentials, query, or fragment.".to_string(),
        ));
    }
    Ok(base_url.trim_end_matches('/').to_string())
}

fn key_header_for(api: Api) -> &'static str {
    match api {
        Api::Anthropic => "x-api-key",
        Api::Gemini | Api::GeminiDecisions => "x-goog-api-key",
        _ => "authorization",
    }
}

fn build_headers<F>(model: &Model, api: Api, key_env: Option<&str>, env: &F) -> Result<HeaderMap>
where
    F: Fn(&str) -> Option<String>,
{
    let mut headers = HeaderMap::new();
    for (k, v) in &model.headers {
        let name = HeaderName::from_bytes(k.as_bytes()).map_err(|_| {
            Error::InvalidConfig(
                "Invalid HTTP header name in Model(headers=...); check header formatting."
                    .to_string(),
            )
        })?;
        let val = HeaderValue::from_str(v).map_err(|_| {
            Error::InvalidConfig(
                "Invalid HTTP header value in Model(headers=...); check header formatting."
                    .to_string(),
            )
        })?;
        headers.insert(name, val);
    }

    if api == Api::Anthropic && !headers.contains_key("anthropic-version") {
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
    }

    let key_header = key_header_for(api);
    if headers.contains_key(key_header) {
        return Ok(headers);
    }

    if let Some(key) = resolve_api_key(model.api_key.as_deref(), key_env, env)?
        && !key.is_empty()
    {
        let raw_val = if key_header == "authorization" {
            format!("Bearer {key}")
        } else {
            key
        };
        let val = HeaderValue::from_str(&raw_val).map_err(|_| {
            Error::InvalidConfig("API key contains invalid HTTP header characters.".to_string())
        })?;
        let name = HeaderName::from_static(key_header);
        headers.insert(name, val);
    }

    Ok(headers)
}

fn resolve_api_key<F>(
    explicit_key: Option<&str>,
    key_env: Option<&str>,
    env: &F,
) -> Result<Option<String>>
where
    F: Fn(&str) -> Option<String>,
{
    if let Some(key) = explicit_key {
        return Ok(Some(key.to_string()));
    }
    let Some(var) = key_env else {
        return Ok(None);
    };
    let mut key = env(var).filter(|v| !v.is_empty());
    if key.is_none() && var == "GOOGLE_API_KEY" {
        key = env("GEMINI_API_KEY").filter(|v| !v.is_empty());
    }
    match key {
        Some(k) => Ok(Some(k)),
        None => Err(Error::extraction(format!(
            "Set {var} or configure Model(api_key=..., headers=...)."
        ))),
    }
}
