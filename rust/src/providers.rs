//! Request builders and response readers for general structured-output APIs.

use std::collections::HashSet;

use percent_encoding::{AsciiSet, NON_ALPHANUMERIC, utf8_percent_encode};
use serde_json::{Map, Value, json};

use crate::content::{Content, DOCUMENT_EXTENSIONS, PROMPT, document_extension};
use crate::error::{Error, Result};
use crate::models::{Api, OutputMode, ResolvedModel, merge_settings};
use crate::schemas::{anthropic_json, openai_strict};

const TOOL_DESCRIPTION: &str = "Return the extracted information.";
const INCOMPLETE: &str = "The model response was not completed.";
const INCOMPLETE_TOOL_CALL: &str = "The model did not complete the extraction tool call.";
const WEB_IMAGES: &[&str] = &["image/png", "image/jpeg", "image/gif", "image/webp"];

const GEMINI_OUTPUT_SETTINGS: &[&str] = &[
    "responseMimeType",
    "responseJsonSchema",
    "responseSchema",
    "responseFormat",
];

// Top-level fields that carry the request structure in at least one API.
const STRUCTURE_FIELDS: &[&str] = &[
    "model",
    "input",
    "instructions",
    "messages",
    "system",
    "contents",
    "systemInstruction",
    "tools",
    "tool_choice",
    "toolConfig",
    "text",
    "response_format",
    "output_config",
    "output_format",
    "format",
];

// Matches Python's `urllib.parse.quote(s, safe='')` (RFC 3986 unreserved characters only).
const PATH_SEGMENT_ENCODE: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'_')
    .remove(b'.')
    .remove(b'~');

/// Intermediate output parsed from a provider response before schema validation.
pub(crate) enum RawOutput {
    JsonText(String),
    JsonObject(Value),
}

/// Build the provider request path and JSON body, then merge caller generation settings.
pub(crate) fn request(
    schema: &Value,
    model: &ResolvedModel,
    content: &Content,
    instructions: Option<&str>,
) -> Result<(String, Value)> {
    let (path, mut body) = match model.api {
        Api::Responses => build_responses(model, schema, content, instructions)?,
        Api::Chat => build_chat(model, schema, content, instructions)?,
        Api::Anthropic => build_anthropic(model, schema, content, instructions)?,
        Api::Gemini => build_gemini(model, schema, content, instructions)?,
        Api::Ollama => build_ollama(model, schema, content, instructions)?,
        Api::Cohere => build_cohere(model, schema, content, instructions)?,
        Api::Bedrock => build_bedrock(model, schema, content, instructions)?,
        Api::MistralOcr => build_mistral_ocr(model, schema, content, instructions)?,
        other => {
            return Err(Error::InvalidConfig(format!(
                "Unsupported general adapter API: '{other}'."
            )));
        }
    };

    if model.api != Api::Gemini {
        // Anthropic's max_tokens is a default, not a field callers are kept from configuring.
        let mut managed_set: HashSet<String> = body.keys().cloned().collect();
        for &field in STRUCTURE_FIELDS {
            managed_set.insert(field.to_string());
        }
        managed_set.remove("max_tokens");
        let managed_refs: Vec<&str> = managed_set.iter().map(String::as_str).collect();
        merge_settings(
            &mut body,
            &model.config.settings,
            &managed_refs,
            "request fields",
        )?;
    }

    if body.get("stream").is_some_and(is_truthy) {
        return Err(Error::InvalidConfig(
            "Streaming is not supported by extract; omit settings['stream'].".to_string(),
        ));
    }

    Ok((path, Value::Object(body)))
}

/// Extract structured output from a provider response payload (before JSON Schema validation).
pub(crate) fn read_output(
    model: &ResolvedModel,
    payload: &Map<String, Value>,
) -> std::result::Result<Value, String> {
    let raw = match (model.api, model.config.output_mode) {
        (Api::Responses, _) => read_responses(payload)?,
        (Api::Chat, Some(OutputMode::JsonSchema)) => read_chat_json(payload)?,
        (Api::Chat, _) => read_chat(payload)?,
        (Api::Anthropic, Some(OutputMode::JsonSchema)) => read_anthropic_json(payload)?,
        (Api::Anthropic, _) => read_anthropic(payload)?,
        (Api::Gemini, _) => read_gemini(payload)?,
        (Api::Ollama, _) => read_ollama(payload)?,
        (Api::Cohere, _) => read_cohere(payload)?,
        (Api::Bedrock, _) => read_bedrock(payload)?,
        (Api::MistralOcr, _) => read_mistral_ocr(payload)?,
        (other, _) => return Err(format!("Unsupported general adapter API: '{other}'.")),
    };

    match raw {
        RawOutput::JsonText(text) => {
            if text.is_empty() {
                return Err("The model returned no structured output.".to_string());
            }
            serde_json::from_str::<Value>(&text).map_err(|_| {
                "The provider returned invalid JSON in the structured output.".to_string()
            })
        }
        RawOutput::JsonObject(obj) => {
            if !obj.is_object() {
                return Err("The model returned no structured output.".to_string());
            }
            Ok(obj)
        }
    }
}

fn build_responses(
    model: &ResolvedModel,
    schema: &Value,
    content: &Content,
    instructions: Option<&str>,
) -> Result<(String, Map<String, Value>)> {
    let document = if content.is_text() {
        json!({"type": "input_text", "text": content.text()})
    } else if content.is_image() {
        json!({"type": "input_image", "image_url": content.data_uri()})
    } else if let Some(ext) = responses_document_extension(&content.media_type) {
        json!({
            "type": "input_file",
            "file_data": content.data_uri(),
            "filename": format!("document{ext}"),
        })
    } else {
        return Err(unsupported_media_error("responses", content));
    };

    let strict_schema = openai_strict(schema)?;
    let mut body = Map::new();
    body.insert("model".to_string(), Value::String(model.model_name.clone()));
    body.insert(
        "input".to_string(),
        json!([{
            "role": "user",
            "content": [{"type": "input_text", "text": PROMPT}, document],
        }]),
    );
    body.insert(
        "text".to_string(),
        json!({
            "format": {
                "type": "json_schema",
                "name": "extract",
                "schema": strict_schema,
                "strict": true,
            }
        }),
    );
    if let Some(instr) = instructions
        && !instr.is_empty()
    {
        body.insert("instructions".to_string(), Value::String(instr.to_string()));
    }
    Ok(("/responses".to_string(), body))
}

fn read_responses(payload: &Map<String, Value>) -> std::result::Result<RawOutput, String> {
    if payload.get("status").and_then(Value::as_str) != Some("completed") {
        return Err(INCOMPLETE.to_string());
    }
    let output = payload
        .get("output")
        .and_then(Value::as_array)
        .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
    let mut text = String::new();
    for item in output {
        let item_obj = item
            .as_object()
            .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
        let item_type = item_obj
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
        if item_type != "message" {
            continue;
        }
        let parts = item_obj
            .get("content")
            .and_then(Value::as_array)
            .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
        for part in parts {
            let part_obj = part.as_object().ok_or_else(|| {
                "The provider returned an unexpected response structure.".to_string()
            })?;
            let part_type = part_obj
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    "The provider returned an unexpected response structure.".to_string()
                })?;
            if part_type == "refusal" {
                return Err("The model refused to extract the document.".to_string());
            }
            if part_type == "output_text" {
                let chunk = part_obj
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        "The provider returned an unexpected response structure.".to_string()
                    })?;
                text.push_str(chunk);
            }
        }
    }
    Ok(RawOutput::JsonText(text))
}

fn build_chat(
    model: &ResolvedModel,
    schema: &Value,
    content: &Content,
    instructions: Option<&str>,
) -> Result<(String, Map<String, Value>)> {
    let document = chat_document(content, "chat")?;
    let mut messages = system_messages(instructions);
    messages.push(user_message(PROMPT, document));

    let mut body = Map::new();
    body.insert("model".to_string(), Value::String(model.model_name.clone()));
    body.insert("messages".to_string(), Value::Array(messages));
    if model.config.output_mode == Some(OutputMode::JsonSchema) {
        let strict_schema = openai_strict(schema)?;
        body.insert(
            "response_format".to_string(),
            json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "extract",
                    "strict": true,
                    "schema": strict_schema,
                }
            }),
        );
    } else {
        body.insert(
            "tools".to_string(),
            json!([{
                "type": "function",
                "function": {
                    "name": "extract",
                    "description": TOOL_DESCRIPTION,
                    "parameters": schema,
                }
            }]),
        );
        body.insert(
            "tool_choice".to_string(),
            json!({"type": "function", "function": {"name": "extract"}}),
        );
    }
    Ok(("/chat/completions".to_string(), body))
}

fn read_chat(payload: &Map<String, Value>) -> std::result::Result<RawOutput, String> {
    let choice = payload
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|arr| arr.first())
        .and_then(Value::as_object)
        .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
    let message = choice
        .get("message")
        .and_then(Value::as_object)
        .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
    let finish_reason = choice
        .get("finish_reason")
        .and_then(Value::as_str)
        .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
    // A forced function call can report a natural stop even with complete tool arguments.
    if !matches!(finish_reason, "tool_calls" | "stop")
        || message.get("refusal").is_some_and(is_truthy)
    {
        return Err(INCOMPLETE_TOOL_CALL.to_string());
    }
    let tool_calls = match message.get("tool_calls") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(arr)) => arr.clone(),
        _ => return Err("The provider returned an unexpected response structure.".to_string()),
    };
    let mut functions = Vec::with_capacity(tool_calls.len());
    for call in &tool_calls {
        let func = call
            .get("function")
            .and_then(Value::as_object)
            .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
        functions.push(func);
    }
    let args = extract_tool_arguments(&functions, "arguments")?;
    match args {
        Value::String(s) => Ok(RawOutput::JsonText(s.clone())),
        Value::Object(_) => Ok(RawOutput::JsonObject(args.clone())),
        _ => Err("The model returned no structured output.".to_string()),
    }
}

fn read_chat_json(payload: &Map<String, Value>) -> std::result::Result<RawOutput, String> {
    let choice = payload
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|arr| arr.first())
        .and_then(Value::as_object)
        .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
    let message = choice
        .get("message")
        .and_then(Value::as_object)
        .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
    let finish_reason = choice
        .get("finish_reason")
        .and_then(Value::as_str)
        .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
    if finish_reason != "stop"
        || message.get("refusal").is_some_and(is_truthy)
        || message.get("tool_calls").is_some_and(is_truthy)
    {
        return Err("The model did not complete a JSON Schema response.".to_string());
    }
    let content = message
        .get("content")
        .and_then(Value::as_str)
        .ok_or_else(|| "Expected JSON text in the model response.".to_string())?;
    Ok(RawOutput::JsonText(content.to_string()))
}

fn build_cohere(
    model: &ResolvedModel,
    schema: &Value,
    content: &Content,
    instructions: Option<&str>,
) -> Result<(String, Map<String, Value>)> {
    let prompt = format!("{PROMPT} Return the extracted information as JSON.");
    let document = chat_document(content, "cohere")?;
    let mut messages = system_messages(instructions);
    messages.push(user_message(&prompt, document));

    let mut body = Map::new();
    body.insert("model".to_string(), Value::String(model.model_name.clone()));
    body.insert("messages".to_string(), Value::Array(messages));
    body.insert(
        "response_format".to_string(),
        json!({"type": "json_object", "schema": schema}),
    );
    Ok(("/chat".to_string(), body))
}

fn read_cohere(payload: &Map<String, Value>) -> std::result::Result<RawOutput, String> {
    let finish = payload
        .get("finish_reason")
        .and_then(Value::as_str)
        .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
    if !finish.eq_ignore_ascii_case("complete") {
        return Err(INCOMPLETE.to_string());
    }
    let parts = payload
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(Value::as_array)
        .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
    let mut text = String::new();
    for part in parts {
        if part.get("type").and_then(Value::as_str) == Some("text") {
            let chunk = part.get("text").and_then(Value::as_str).ok_or_else(|| {
                "The provider returned an unexpected response structure.".to_string()
            })?;
            text.push_str(chunk);
        }
    }
    Ok(RawOutput::JsonText(text))
}

fn build_anthropic(
    model: &ResolvedModel,
    schema: &Value,
    content: &Content,
    instructions: Option<&str>,
) -> Result<(String, Map<String, Value>)> {
    let document = if content.is_text() {
        text_block(&content.text())
    } else if content.is_pdf() || WEB_IMAGES.contains(&content.media_type.as_str()) {
        let kind = if content.is_pdf() {
            "document"
        } else {
            "image"
        };
        json!({
            "type": kind,
            "source": {
                "type": "base64",
                "media_type": content.media_type,
                "data": content.base64(),
            }
        })
    } else {
        return Err(unsupported_media_error("anthropic", content));
    };

    let mut body = Map::new();
    body.insert("model".to_string(), Value::String(model.model_name.clone()));
    body.insert("max_tokens".to_string(), json!(4096));
    body.insert(
        "messages".to_string(),
        Value::Array(vec![user_message(PROMPT, document)]),
    );
    if let Some(instr) = instructions
        && !instr.is_empty()
    {
        body.insert("system".to_string(), Value::String(instr.to_string()));
    }
    if model.config.output_mode == Some(OutputMode::JsonSchema) {
        let adapted = anthropic_json(schema)?;
        body.insert(
            "output_config".to_string(),
            json!({"format": {"type": "json_schema", "schema": adapted}}),
        );
    } else {
        body.insert(
            "tools".to_string(),
            json!([{
                "name": "extract",
                "description": TOOL_DESCRIPTION,
                "input_schema": schema,
            }]),
        );
        body.insert(
            "tool_choice".to_string(),
            json!({"type": "tool", "name": "extract"}),
        );
    }
    Ok(("/messages".to_string(), body))
}

fn read_anthropic(payload: &Map<String, Value>) -> std::result::Result<RawOutput, String> {
    let stop = payload
        .get("stop_reason")
        .and_then(Value::as_str)
        .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
    if stop != "tool_use" {
        return Err(INCOMPLETE_TOOL_CALL.to_string());
    }
    let parts = payload
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
    let mut calls = Vec::new();
    for part in parts {
        let obj = part
            .as_object()
            .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
        if obj.get("type").and_then(Value::as_str) == Some("tool_use") {
            calls.push(obj);
        }
    }
    let input = extract_tool_arguments(&calls, "input")?;
    match input {
        Value::Object(_) => Ok(RawOutput::JsonObject(input.clone())),
        Value::String(s) => Ok(RawOutput::JsonText(s.clone())),
        _ => Err("The model returned no structured output.".to_string()),
    }
}

fn read_anthropic_json(payload: &Map<String, Value>) -> std::result::Result<RawOutput, String> {
    let stop = payload
        .get("stop_reason")
        .and_then(Value::as_str)
        .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
    if stop != "end_turn" {
        return Err("The model refused or did not complete a JSON Schema response.".to_string());
    }
    let parts = payload
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
    let mut texts: Vec<&Value> = Vec::new();
    for part in parts {
        let obj = part
            .as_object()
            .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
        let part_type = obj
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
        if part_type == "tool_use" {
            return Err("Unexpected tool call in a JSON Schema response.".to_string());
        }
        if part_type == "text" {
            let text_val = obj.get("text").ok_or_else(|| {
                "The provider returned an unexpected response structure.".to_string()
            })?;
            texts.push(text_val);
        }
    }
    if texts.len() != 1 {
        return Err("Expected one JSON text block in the model response.".to_string());
    }
    let Some(text) = texts[0].as_str() else {
        return Err("Expected one JSON text block in the model response.".to_string());
    };
    Ok(RawOutput::JsonText(text.to_string()))
}

fn build_gemini(
    model: &ResolvedModel,
    schema: &Value,
    content: &Content,
    instructions: Option<&str>,
) -> Result<(String, Map<String, Value>)> {
    let media = content.media_type.as_str();
    let mut document = if content.is_text() {
        let mut m = Map::new();
        m.insert(
            "text".to_string(),
            Value::String(content.text().into_owned()),
        );
        m
    } else if media.starts_with("image/")
        || media.starts_with("audio/")
        || media.starts_with("video/")
        || is_gemini_document(media)
    {
        let mut m = Map::new();
        m.insert(
            "inlineData".to_string(),
            json!({"mimeType": media, "data": content.base64()}),
        );
        m
    } else {
        return Err(unsupported_media_error("gemini", content));
    };

    let mut settings = model.config.settings.clone();
    if let Some(spreadsheet_meta) = settings.shift_remove("spreadsheetMetadata") {
        if !is_gemini_spreadsheet(media) {
            return Err(Error::InvalidConfig(
                "spreadsheetMetadata requires an XLSX, XLS, or ODS input.".to_string(),
            ));
        }
        document.insert("spreadsheetMetadata".to_string(), spreadsheet_meta);
    }

    let mut generation_config = Map::new();
    generation_config.insert(
        "responseMimeType".to_string(),
        Value::String("application/json".to_string()),
    );
    generation_config.insert("responseJsonSchema".to_string(), schema.clone());
    merge_settings(
        &mut generation_config,
        &settings,
        GEMINI_OUTPUT_SETTINGS,
        "structured output",
    )?;

    let mut body = Map::new();
    body.insert(
        "contents".to_string(),
        json!([{"role": "user", "parts": [{"text": PROMPT}, Value::Object(document)]}]),
    );
    body.insert(
        "generationConfig".to_string(),
        Value::Object(generation_config),
    );
    if let Some(instr) = instructions
        && !instr.is_empty()
    {
        body.insert(
            "systemInstruction".to_string(),
            json!({"parts": [{"text": instr}]}),
        );
    }
    let encoded = utf8_percent_encode(&model.model_name, PATH_SEGMENT_ENCODE);
    Ok((format!("/models/{encoded}:generateContent"), body))
}

fn read_gemini(payload: &Map<String, Value>) -> std::result::Result<RawOutput, String> {
    let candidate = payload
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|arr| arr.first())
        .and_then(Value::as_object)
        .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
    if candidate.get("finishReason").and_then(Value::as_str) != Some("STOP") {
        return Err("The model response was blocked or incomplete.".to_string());
    }
    let parts = candidate
        .get("content")
        .and_then(|c| c.get("parts"))
        .and_then(Value::as_array)
        .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
    let mut text = String::new();
    for part in parts {
        let obj = part
            .as_object()
            .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
        if obj.get("thought").is_some_and(is_truthy) {
            continue;
        }
        if let Some(chunk) = obj.get("text").and_then(Value::as_str) {
            text.push_str(chunk);
        }
    }
    Ok(RawOutput::JsonText(text))
}

fn build_ollama(
    model: &ResolvedModel,
    schema: &Value,
    content: &Content,
    instructions: Option<&str>,
) -> Result<(String, Map<String, Value>)> {
    let mut message = Map::new();
    message.insert("role".to_string(), Value::String("user".to_string()));
    if content.is_text() {
        message.insert(
            "content".to_string(),
            Value::String(format!("{PROMPT}\n\n{}", content.text())),
        );
    } else if content.is_image() {
        message.insert("content".to_string(), Value::String(PROMPT.to_string()));
        message.insert("images".to_string(), json!([content.base64()]));
    } else {
        return Err(unsupported_media_error("ollama", content));
    }

    let mut messages = system_messages(instructions);
    messages.push(Value::Object(message));

    let mut body = Map::new();
    body.insert("model".to_string(), Value::String(model.model_name.clone()));
    body.insert("messages".to_string(), Value::Array(messages));
    body.insert("format".to_string(), schema.clone());
    body.insert("stream".to_string(), Value::Bool(false));
    Ok(("/api/chat".to_string(), body))
}

fn read_ollama(payload: &Map<String, Value>) -> std::result::Result<RawOutput, String> {
    let done = payload.get("done").and_then(Value::as_bool) == Some(true);
    let reason = payload
        .get("done_reason")
        .and_then(Value::as_str)
        .unwrap_or("stop");
    if !done || reason != "stop" {
        return Err(INCOMPLETE.to_string());
    }
    let content = payload
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(Value::as_str)
        .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
    Ok(RawOutput::JsonText(content.to_string()))
}

fn build_mistral_ocr(
    model: &ResolvedModel,
    schema: &Value,
    content: &Content,
    instructions: Option<&str>,
) -> Result<(String, Map<String, Value>)> {
    if !content.is_pdf() {
        return Err(Error::InvalidConfig(
            "The mistral-ocr adapter supports only PDF inputs.".to_string(),
        ));
    }
    let annotation_prompt = match instructions {
        Some(instr) if !instr.is_empty() => format!("{PROMPT}\n{instr}").trim().to_string(),
        _ => PROMPT.to_string(),
    };
    let mut body = Map::new();
    body.insert("model".to_string(), Value::String(model.model_name.clone()));
    body.insert(
        "document".to_string(),
        json!({"type": "document_url", "document_url": content.data_uri()}),
    );
    body.insert(
        "document_annotation_format".to_string(),
        json!({
            "type": "json_schema",
            "json_schema": {"name": "extract", "schema": schema, "strict": true}
        }),
    );
    body.insert(
        "document_annotation_prompt".to_string(),
        Value::String(annotation_prompt),
    );
    body.insert("bbox_annotation_format".to_string(), Value::Null);
    body.insert("include_image_base64".to_string(), Value::Bool(false));
    Ok(("/ocr".to_string(), body))
}

fn read_mistral_ocr(payload: &Map<String, Value>) -> std::result::Result<RawOutput, String> {
    match payload.get("document_annotation") {
        Some(Value::String(s)) => Ok(RawOutput::JsonText(s.clone())),
        Some(val @ Value::Object(_)) => Ok(RawOutput::JsonObject(val.clone())),
        Some(_) => Err("The model returned no structured output.".to_string()),
        None => Err("The provider returned an unexpected response structure.".to_string()),
    }
}

fn build_bedrock(
    model: &ResolvedModel,
    schema: &Value,
    content: &Content,
    instructions: Option<&str>,
) -> Result<(String, Map<String, Value>)> {
    let document = if content.is_text() {
        json!({"text": content.text()})
    } else if WEB_IMAGES.contains(&content.media_type.as_str()) {
        let image_format = content.media_type.split('/').nth(1).unwrap_or("png");
        json!({"image": {"format": image_format, "source": {"bytes": content.base64()}}})
    } else if content.is_pdf() {
        json!({"document": {"format": "pdf", "name": "document", "source": {"bytes": content.base64()}}})
    } else {
        return Err(unsupported_media_error("bedrock", content));
    };

    let mut body = Map::new();
    body.insert(
        "messages".to_string(),
        json!([{"role": "user", "content": [{"text": PROMPT}, document]}]),
    );
    body.insert(
        "toolConfig".to_string(),
        json!({
            "tools": [{
                "toolSpec": {
                    "name": "extract",
                    "description": TOOL_DESCRIPTION,
                    "inputSchema": {"json": schema},
                }
            }],
            "toolChoice": {"tool": {"name": "extract"}},
        }),
    );
    if let Some(instr) = instructions
        && !instr.is_empty()
    {
        body.insert("system".to_string(), json!([{"text": instr}]));
    }
    let encoded = utf8_percent_encode(&model.model_name, PATH_SEGMENT_ENCODE);
    Ok((format!("/model/{encoded}/converse"), body))
}

fn read_bedrock(payload: &Map<String, Value>) -> std::result::Result<RawOutput, String> {
    if payload.get("stopReason").and_then(Value::as_str) != Some("tool_use") {
        return Err(INCOMPLETE_TOOL_CALL.to_string());
    }
    let parts = payload
        .get("output")
        .and_then(|o| o.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(Value::as_array)
        .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
    let mut calls = Vec::new();
    for part in parts {
        if let Some(tool_use) = part.get("toolUse").and_then(Value::as_object) {
            calls.push(tool_use);
        }
    }
    let input = extract_tool_arguments(&calls, "input")?;
    match input {
        Value::Object(_) => Ok(RawOutput::JsonObject(input.clone())),
        Value::String(s) => Ok(RawOutput::JsonText(s.clone())),
        _ => Err("The model returned no structured output.".to_string()),
    }
}

fn chat_document(content: &Content, api: &str) -> Result<Value> {
    if content.is_text() {
        return Ok(text_block(&content.text()));
    }
    if content.is_image() {
        return Ok(json!({"type": "image_url", "image_url": {"url": content.data_uri()}}));
    }
    if api == "chat" && content.is_pdf() {
        return Ok(json!({
            "type": "file",
            "file": {"filename": "document.pdf", "file_data": content.data_uri()},
        }));
    }
    if api == "chat" {
        let format = match content.media_type.as_str() {
            "audio/mpeg" => Some("mp3"),
            "audio/wav" => Some("wav"),
            _ => None,
        };
        if let Some(fmt) = format {
            return Ok(json!({
                "type": "input_audio",
                "input_audio": {"data": content.base64(), "format": fmt},
            }));
        }
    }
    Err(unsupported_media_error(api, content))
}

fn responses_document_extension(media_type: &str) -> Option<&'static str> {
    let ext = document_extension(media_type)?;
    if matches!(ext, ".dotx" | ".odp" | ".ods") {
        return None;
    }
    Some(ext)
}

fn is_gemini_document(media_type: &str) -> bool {
    media_type != "application/rtf"
        && DOCUMENT_EXTENSIONS
            .iter()
            .any(|(media, _)| *media == media_type)
}

fn is_gemini_spreadsheet(media_type: &str) -> bool {
    matches!(
        document_extension(media_type),
        Some(".xlsx" | ".xls" | ".ods")
    )
}

fn text_block(text: &str) -> Value {
    json!({"type": "text", "text": text})
}

fn user_message(prompt: &str, document: Value) -> Value {
    json!({"role": "user", "content": [text_block(prompt), document]})
}

fn system_messages(instructions: Option<&str>) -> Vec<Value> {
    match instructions {
        Some(instr) if !instr.is_empty() => vec![json!({"role": "system", "content": instr})],
        _ => Vec::new(),
    }
}

fn unsupported_media_error(api: &str, content: &Content) -> Error {
    Error::InvalidConfig(format!(
        "The {api} adapter does not support {} inputs.",
        content.media_type
    ))
}

fn extract_tool_arguments<'a>(
    calls: &[&'a Map<String, Value>],
    arguments_key: &str,
) -> std::result::Result<&'a Value, String> {
    if calls.len() != 1 || calls[0].get("name").and_then(Value::as_str) != Some("extract") {
        return Err("Expected exactly one extraction tool call.".to_string());
    }
    calls[0]
        .get(arguments_key)
        .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())
}

fn is_truthy(val: &Value) -> bool {
    match val {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}
