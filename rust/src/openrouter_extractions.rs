//! Typed field extraction through OpenRouter's alpha Extractions API.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::content::Content;
use crate::error::{Error, Result};
use crate::models::{ResolvedModel, merge_settings};
use crate::schemas::{ensure_schema_object, resolve_ref};

const LIST_ITEMS: &[&str] = &["text", "number", "date"];
const REQUEST_FIELDS: &[&str] = &["model", "state", "assets", "fields"];

/// Native OpenRouter Extractions field options, attachable in JSON Schema via `"x-openrouter"`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct OpenRouterField {
    pub options: Map<String, Value>,
}

impl OpenRouterField {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_option(mut self, key: impl Into<String>, value: Value) -> Self {
        self.options.insert(key.into(), value);
        self
    }

    pub fn from_map(options: Map<String, Value>) -> Self {
        Self { options }
    }
}

/// Validated data together with OpenRouter's per-field answer envelopes and response metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractionResult<T> {
    pub data: T,
    pub answers: Map<String, Value>,
    pub metadata: Map<String, Value>,
}

/// Build an OpenRouter Extractions request path and JSON body from a JSON Schema.
pub(crate) fn request(
    schema: &Value,
    model: &ResolvedModel,
    content: &Content,
    instructions: Option<&str>,
) -> Result<(String, Value)> {
    let obj = ensure_schema_object(schema)?;
    if obj.get("type").and_then(Value::as_str) != Some("object") && !obj.contains_key("properties")
    {
        return Err(Error::InvalidConfig(
            "OpenRouter Extractions requires an object schema with named fields.".to_string(),
        ));
    }
    let props = obj
        .get("properties")
        .and_then(Value::as_object)
        .filter(|p| !p.is_empty())
        .ok_or_else(|| {
            Error::InvalidConfig(
                "OpenRouter Extractions requires at least one schema field.".to_string(),
            )
        })?;

    let mut fields = Map::new();
    for (name, prop) in props {
        let spec = field_spec(schema, name, prop, instructions)?;
        fields.insert(name.clone(), Value::Object(spec));
    }

    let mut body = Map::new();
    body.insert("model".to_string(), Value::String(model.model_name.clone()));
    body.insert("fields".to_string(), Value::Object(fields));
    merge_settings(
        &mut body,
        &model.config.settings,
        REQUEST_FIELDS,
        "request fields",
    )?;

    if content.is_text() {
        body.insert(
            "state".to_string(),
            Value::String(content.text().into_owned()),
        );
    } else if let Some((kind, file_format)) = asset_format(&content.media_type) {
        body.insert("state".to_string(), Value::String("document".to_string()));
        body.insert(
            "assets".to_string(),
            json!({
                "document": {
                    "type": kind,
                    "format": file_format,
                    "data_base64": content.base64(),
                }
            }),
        );
    } else {
        return Err(Error::InvalidConfig(format!(
            "OpenRouter Extractions supports text, PDF, PNG, JPEG, and WebP. Received {}.",
            content.media_type
        )));
    }

    Ok(("/extractions".to_string(), Value::Object(body)))
}

/// Intermediate parsed OpenRouter Extractions response before schema validation.
pub(crate) struct ParsedExtraction {
    pub value: Value,
    pub answers: Map<String, Value>,
    pub metadata: Map<String, Value>,
}

/// Extract the answer values, raw answer envelopes, and response metadata from a response payload.
pub(crate) fn read_response(
    schema: &Value,
    payload: &Map<String, Value>,
    include_metadata: bool,
) -> std::result::Result<ParsedExtraction, String> {
    let props = schema
        .get("properties")
        .and_then(Value::as_object)
        .ok_or_else(|| "OpenRouter Extractions requires an object schema.".to_string())?;

    let Some(answers) = payload.get("answers").and_then(Value::as_object) else {
        return Err(
            "OpenRouter Extractions must return one answer for every schema field.".to_string(),
        );
    };

    if answers.len() != props.len() || !props.keys().all(|k| answers.contains_key(k)) {
        return Err(
            "OpenRouter Extractions must return one answer for every schema field.".to_string(),
        );
    }

    if !include_metadata && payload.get("truncated").and_then(Value::as_bool) == Some(true) {
        return Err(
            "OpenRouter read only part of the source. Use include_metadata=True to inspect \
             partial results or retry with a shorter source."
                .to_string(),
        );
    }

    let mut values = Map::new();
    for (name, answer) in answers {
        let Some(answer_obj) = answer.as_object() else {
            return Err("Every OpenRouter Extractions answer must contain a value.".to_string());
        };
        let Some(val) = answer_obj.get("value") else {
            return Err("Every OpenRouter Extractions answer must contain a value.".to_string());
        };
        values.insert(name.clone(), val.clone());
    }

    let mut metadata = Map::new();
    for (k, v) in payload {
        if k != "answers" {
            metadata.insert(k.clone(), v.clone());
        }
    }

    Ok(ParsedExtraction {
        value: Value::Object(values),
        answers: answers.clone(),
        metadata,
    })
}

fn asset_format(media_type: &str) -> Option<(&'static str, &'static str)> {
    match media_type {
        "application/pdf" => Some(("document", "pdf")),
        "image/jpeg" => Some(("image", "jpeg")),
        "image/png" => Some(("image", "png")),
        "image/webp" => Some(("image", "webp")),
        _ => None,
    }
}

fn field_spec(
    root: &Value,
    name: &str,
    prop: &Value,
    instructions: Option<&str>,
) -> Result<Map<String, Value>> {
    let mut options = Map::new();
    if let Some(ext) = prop.get("x-openrouter").and_then(Value::as_object) {
        for (k, v) in ext {
            options.insert(k.clone(), v.clone());
        }
    }

    let mut spec = if options.contains_key("type") {
        options
    } else {
        let mut inferred = infer_field(root, name, prop)?;
        for (k, v) in options {
            inferred.insert(k, v);
        }
        inferred
    };

    if !spec.contains_key("instructions") {
        let desc = prop
            .get("description")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| name.replace('_', " "));
        spec.insert("instructions".to_string(), Value::String(desc));
    }

    if let Some(instr) = instructions
        && !instr.is_empty()
    {
        // Keep global instructions out of state so they don't shift source spans.
        let current = spec
            .get("instructions")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        spec.insert(
            "instructions".to_string(),
            Value::String(format!("{current}\n{instr}")),
        );
    }

    Ok(spec)
}

fn infer_field(root: &Value, name: &str, prop: &Value) -> Result<Map<String, Value>> {
    let unwrapped = unwrap_nullable(root, prop)?;
    if let Some(spec) = scalar_spec(unwrapped) {
        return Ok(spec);
    }
    if unwrapped.get("type").and_then(Value::as_str) == Some("array")
        && let Some(items) = unwrapped.get("items")
    {
        let resolved_items = resolve_if_ref(root, items)?;
        if let Some(mut item_spec) = scalar_spec(resolved_items)
            && let Some(item_type) = item_spec
                .get("type")
                .and_then(Value::as_str)
                .map(str::to_string)
            && LIST_ITEMS.contains(&item_type.as_str())
        {
            item_spec.insert("type".to_string(), Value::String("list".to_string()));
            item_spec.insert("items".to_string(), Value::String(item_type));
            return Ok(item_spec);
        }
    }
    Err(Error::InvalidConfig(format!(
        "OpenRouter Extractions cannot infer a field type for '{name}'. \
         Use a supported scalar or list, set a compatible OpenRouterField(type=...) / x-openrouter, \
         or use a structured-output model for nested schemas."
    )))
}

fn resolve_if_ref<'a>(root: &'a Value, node: &'a Value) -> Result<&'a Value> {
    let mut current = node;
    let mut depth = 0;
    while let Some(r) = current.get("$ref").and_then(Value::as_str) {
        if depth > 16 {
            return Err(Error::InvalidConfig(
                "Recursive JSON Schema reference in OpenRouter Extractions schema.".to_string(),
            ));
        }
        current = resolve_ref(root, r)?;
        depth += 1;
    }
    Ok(current)
}

fn unwrap_nullable<'a>(root: &'a Value, prop: &'a Value) -> Result<&'a Value> {
    let node = resolve_if_ref(root, prop)?;
    for union_key in ["anyOf", "oneOf"] {
        if let Some(branches) = node.get(union_key).and_then(Value::as_array) {
            let non_null: Vec<&Value> = branches
                .iter()
                .filter(|b| b.get("type").and_then(Value::as_str) != Some("null"))
                .collect();
            if non_null.len() == 1 && non_null.len() < branches.len() {
                return resolve_if_ref(root, non_null[0]);
            }
        }
    }
    Ok(node)
}

fn scalar_spec(node: &Value) -> Option<Map<String, Value>> {
    let obj = node.as_object()?;
    let type_name = match obj.get("type") {
        Some(Value::String(s)) => s.as_str(),
        Some(Value::Array(arr)) => {
            let non_null: Vec<&str> = arr
                .iter()
                .filter_map(Value::as_str)
                .filter(|t| *t != "null")
                .collect();
            if non_null.len() != 1 {
                return None;
            }
            non_null[0]
        }
        _ => return None,
    };

    let mut spec = Map::new();
    match type_name {
        "string" => {
            let openrouter_type = match obj.get("format").and_then(Value::as_str) {
                Some("date") => "date",
                Some("time") => "time",
                Some("date-time") => "datetime",
                _ => "text",
            };
            spec.insert(
                "type".to_string(),
                Value::String(openrouter_type.to_string()),
            );
            Some(spec)
        }
        "integer" => {
            spec.insert("type".to_string(), Value::String("number".to_string()));
            spec.insert("integer".to_string(), Value::Bool(true));
            Some(spec)
        }
        "number" => {
            spec.insert("type".to_string(), Value::String("number".to_string()));
            Some(spec)
        }
        _ => None,
    }
}
