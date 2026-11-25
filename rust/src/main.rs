use std::collections::BTreeMap;
use std::io::{self, Read};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use serde::Serialize;
use serde_json::{Map, Value};

use openextract::{
    Api, Error, ExtractOptions, Model, OutputMode, Result, Source, builtin_providers,
    extract_value, extract_value_with_metadata, fields_to_schema,
};

#[derive(Parser, Debug)]
#[command(
    name = "openextract",
    version,
    about = "Extract structured JSON from documents, images, audio, video, URLs, and text using LLMs.",
    long_about = "Extract structured JSON from documents, images, audio, video, URLs, and text using LLMs.\n\n\
        Stdout receives valid JSON only; diagnostics and errors are written to stderr.\n\
        Exit codes: 0=ok, 2=config/usage, 3=env/io/fetch, 4=provider/http, 5=validation/refusal.",
    after_help = "EXAMPLES:\n  \
        # Quick extraction with --field shorthand from a local PDF\n  \
        openextract -m openai:gpt-6.1-sol -f 'vendor:string=Seller, not buyer' -f total:number -f currency:string? invoice.pdf\n\n  \
        # Full JSON Schema from a file on an image or URL\n  \
        openextract -m anthropic:claude-sonnet-5-5 --schema invoice.schema.json https://example.com/2291.pdf\n\n  \
        # Pipe text or binary bytes from stdin\n  \
        cat email.txt | openextract -m google:gemini-2.5-flash -f sender:string -f action_items:string[]\n\n  \
        # Closed-set Decisions API\n  \
        openextract -m openai-decisions:gpt-6-luna -f 'doc_type:enum(invoice,receipt,contract)' -f is_paid:bool receipt.png\n\n  \
        # Inspect configured providers and environment keys\n  \
        openextract --list-providers"
)]
struct Cli {
    /// Source file path, http(s):// URL, or '-' (default: read from stdin).
    /// Use --text to pass literal document text directly.
    #[arg(value_name = "SOURCE")]
    source: Option<String>,

    /// Pass literal UTF-8 document text directly (never interpreted as a file path or URL).
    #[arg(long, conflicts_with = "source")]
    text: Option<String>,

    /// Model identifier in 'provider:model' form (e.g. openai:gpt-6.1-sol, anthropic:claude-sonnet-5-5).
    #[arg(short, long, env = "OPENEXTRACT_MODEL")]
    model: Option<String>,

    /// JSON Schema as a file path, '@path', inline JSON object '{...}', or '-' for stdin.
    #[arg(short, long, conflicts_with = "fields")]
    schema: Option<String>,

    /// Shorthand field spec: NAME[:TYPE][?][=DESCRIPTION] (repeatable).
    /// Types: string, number, integer, boolean, date, time, datetime, email, uri, uuid, enum(a,b), T[].
    #[arg(short = 'f', long = "field", value_name = "SPEC")]
    fields: Vec<String>,

    /// Optional extraction instructions for this call.
    #[arg(short, long)]
    instructions: Option<String>,

    /// Override media type detection (e.g. application/pdf, audio/mpeg, text/markdown).
    #[arg(long)]
    media_type: Option<String>,

    /// Override the HTTP API protocol (responses, chat, anthropic, gemini, ollama, cohere,
    /// bedrock, mistral-ocr, openai-decisions, gemini-decisions, openrouter-decisions, openrouter-extractions).
    #[arg(long)]
    api: Option<String>,

    /// Override provider base URL, including API version (e.g. https://host/v1).
    #[arg(long)]
    base_url: Option<String>,

    /// Explicit API key (prefer provider environment variables instead).
    #[arg(long)]
    api_key: Option<String>,

    /// Extra HTTP header in 'Name: Value' format (repeatable; overrides generated headers).
    #[arg(short = 'H', long = "header", value_name = "NAME:VALUE")]
    headers: Vec<String>,

    /// Provider-native generation settings as a JSON object (e.g. '{"temperature":0}').
    #[arg(long, value_name = "JSON")]
    settings: Option<String>,

    /// Structured output mode for Chat Completions and Anthropic Messages ('tool' or 'json_schema').
    #[arg(long)]
    output_mode: Option<String>,

    /// Request timeout in seconds.
    #[arg(long, default_value_t = 60.0)]
    timeout: f64,

    /// Maximum HTTP retries for transient errors and 408/409/429/5xx statuses.
    #[arg(long, default_value_t = 2)]
    max_retries: u32,

    /// Maximum retries after schema validation failure.
    #[arg(long, default_value_t = 1)]
    validation_retries: u32,

    /// Return {data, answers, metadata} with OpenRouter Extractions.
    #[arg(long)]
    include_metadata: bool,

    /// Emit compact single-line JSON instead of pretty-printed JSON.
    #[arg(long)]
    compact: bool,

    /// Write JSON output to a file instead of stdout.
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Print the resolved JSON Schema (from --schema or --field) and exit without calling a model.
    #[arg(long)]
    print_schema: bool,

    /// List built-in providers, APIs, env vars, and which API keys are set, then exit.
    #[arg(long)]
    list_providers: bool,
}

#[derive(Serialize)]
struct ProviderStatus {
    prefix: &'static str,
    aliases: &'static [&'static str],
    api: Api,
    base_url: &'static str,
    key_env: Option<&'static str>,
    key_env_set: bool,
    url_env: &'static str,
    inputs: &'static str,
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(err.exit_code() as u8)
        }
    }
}

async fn run(cli: Cli) -> Result<()> {
    if cli.list_providers {
        let list: Vec<ProviderStatus> = builtin_providers()
            .iter()
            .map(|p| {
                let set = match p.key_env {
                    None => true,
                    Some("GOOGLE_API_KEY") => {
                        env_non_empty("GOOGLE_API_KEY") || env_non_empty("GEMINI_API_KEY")
                    }
                    Some(var) => env_non_empty(var),
                };
                ProviderStatus {
                    prefix: p.prefix,
                    aliases: p.aliases,
                    api: p.api,
                    base_url: p.base_url,
                    key_env: p.key_env,
                    key_env_set: set,
                    url_env: p.url_env,
                    inputs: p.inputs,
                }
            })
            .collect();
        let val = serde_json::to_value(list)
            .map_err(|e| Error::InvalidConfig(format!("Failed to serialize providers: {e}")))?;
        return write_json(&val, cli.compact, cli.output.as_deref());
    }

    let schema = load_schema(&cli)?;
    if cli.print_schema {
        return write_json(&schema, cli.compact, cli.output.as_deref());
    }

    let model_name = cli.model.as_deref().ok_or_else(|| {
        Error::InvalidConfig(
            "Missing --model (or OPENEXTRACT_MODEL). Example: -m openai:gpt-6.1-sol".to_string(),
        )
    })?;

    let model = build_model(model_name, &cli)?;
    let source = load_source(&cli)?;
    let options = ExtractOptions {
        instructions: cli.instructions.as_deref(),
        media_type: cli.media_type.as_deref(),
        client: None,
    };

    let output_value = if cli.include_metadata {
        let result = extract_value_with_metadata(&schema, model, source, options).await?;
        serde_json::to_value(result)
            .map_err(|e| Error::InvalidConfig(format!("Failed to serialize result: {e}")))?
    } else {
        extract_value(&schema, model, source, options).await?
    };

    write_json(&output_value, cli.compact, cli.output.as_deref())
}

fn load_schema(cli: &Cli) -> Result<Value> {
    if !cli.fields.is_empty() {
        return fields_to_schema(&cli.fields);
    }
    let Some(raw) = cli.schema.as_deref() else {
        return Err(Error::InvalidConfig(
            "Provide a schema with --schema <file-or-json> or one or more --field specs."
                .to_string(),
        ));
    };

    let trimmed = raw.trim();
    let text = if trimmed.starts_with('{') {
        trimmed.to_string()
    } else if trimmed == "-" {
        let mut buf = String::new();
        io::stdin().read_to_string(&mut buf)?;
        buf
    } else {
        let path = trimmed.strip_prefix('@').unwrap_or(trimmed);
        std::fs::read_to_string(path)?
    };

    let val: Value = serde_json::from_str(&text)
        .map_err(|e| Error::InvalidConfig(format!("Invalid JSON in --schema: {e}")))?;
    if !val.is_object() {
        return Err(Error::InvalidConfig(
            "--schema must be a JSON Schema object.".to_string(),
        ));
    }
    Ok(val)
}

fn load_source(cli: &Cli) -> Result<Source> {
    if let Some(ref text) = cli.text {
        return Ok(Source::Text(text.clone()));
    }
    match cli.source.as_deref() {
        None | Some("-") => {
            if cli.schema.as_deref().map(str::trim) == Some("-") {
                return Err(Error::InvalidConfig(
                    "Cannot read both --schema and SOURCE from stdin ('-').".to_string(),
                ));
            }
            let mut bytes = Vec::new();
            io::stdin().read_to_end(&mut bytes)?;
            Ok(Source::Bytes(bytes))
        }
        Some(arg) if arg.starts_with("http://") || arg.starts_with("https://") => {
            Ok(Source::Url(arg.to_string()))
        }
        Some(path) => Ok(Source::Path(PathBuf::from(path))),
    }
}

fn build_model(model_name: &str, cli: &Cli) -> Result<Model> {
    let mut model = Model::new(model_name)
        .with_timeout(cli.timeout)
        .with_max_retries(cli.max_retries)
        .with_validation_retries(cli.validation_retries);

    if let Some(ref api_str) = cli.api {
        model = model.with_api(api_str.parse::<Api>()?);
    }
    if let Some(ref url) = cli.base_url {
        model = model.with_base_url(url.clone());
    }
    if let Some(ref key) = cli.api_key {
        model = model.with_api_key(key.clone());
    }
    if let Some(ref mode_str) = cli.output_mode {
        model = model.with_output_mode(mode_str.parse::<OutputMode>()?);
    }

    if !cli.headers.is_empty() {
        let mut parsed = BTreeMap::new();
        for header in &cli.headers {
            let Some((k, v)) = header.split_once(':') else {
                return Err(Error::InvalidConfig(format!(
                    "Invalid --header '{header}'; expected 'Name: Value'."
                )));
            };
            let name = k.trim();
            if name.is_empty() {
                return Err(Error::InvalidConfig(format!(
                    "Empty header name in --header '{header}'."
                )));
            }
            parsed.insert(name.to_string(), v.trim().to_string());
        }
        model = model.with_headers(parsed);
    }

    if let Some(ref raw_settings) = cli.settings {
        let parsed: Value = serde_json::from_str(raw_settings)
            .map_err(|e| Error::InvalidConfig(format!("Invalid JSON in --settings: {e}")))?;
        let Value::Object(map) = parsed else {
            return Err(Error::InvalidConfig(
                "--settings must be a JSON object.".to_string(),
            ));
        };
        let settings: Map<String, Value> = map;
        model = model.with_settings(settings);
    }

    Ok(model)
}

fn write_json(value: &Value, compact: bool, output_path: Option<&std::path::Path>) -> Result<()> {
    let serialized = if compact {
        serde_json::to_string(value)
    } else {
        serde_json::to_string_pretty(value)
    }
    .map_err(|e| Error::InvalidConfig(format!("Failed to format JSON output: {e}")))?;

    match output_path {
        Some(path) => {
            std::fs::write(path, format!("{serialized}\n"))?;
        }
        None => {
            println!("{serialized}");
        }
    }
    Ok(())
}

fn env_non_empty(key: &str) -> bool {
    std::env::var(key).is_ok_and(|v| !v.is_empty())
}
