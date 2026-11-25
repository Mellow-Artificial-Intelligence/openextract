//! Concise `--field` shorthand parser for building JSON Schemas on the CLI.
//!
//! Syntax: `NAME[:TYPE][?][=DESCRIPTION]`
//!
//! Examples:
//! - `vendor` -> required string
//! - `vendor:string=Seller, not buyer.` -> required string with description
//! - `total:number` -> required number
//! - `currency:string?` -> nullable string (still required so the model can return `null`)
//! - `tags:string[]` -> array of strings
//! - `due_date:date?` -> nullable ISO date
//! - `kind:enum(invoice,receipt,contract)` -> closed-set string enum

use serde_json::{Map, Value, json};

use crate::error::{Error, Result};

/// Build a standard JSON Schema object from one or more `--field` shorthand specifications.
pub fn fields_to_schema(specs: &[String]) -> Result<Value> {
    if specs.is_empty() {
        return Err(Error::InvalidConfig(
            "Provide at least one --field specification or pass --schema.".to_string(),
        ));
    }

    let mut properties = Map::new();
    let mut required = Vec::with_capacity(specs.len());

    for raw in specs {
        let (name, schema) = parse_field_spec(raw)?;
        if properties.contains_key(&name) {
            return Err(Error::InvalidConfig(format!(
                "Duplicate --field name: '{name}'."
            )));
        }
        required.push(Value::String(name.clone()));
        properties.insert(name, schema);
    }

    Ok(json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
    }))
}

fn parse_field_spec(raw: &str) -> Result<(String, Value)> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(Error::InvalidConfig(
            "Empty --field specification; expected NAME[:TYPE][?][=DESCRIPTION].".to_string(),
        ));
    }

    let (lhs, description) = match trimmed.split_once('=') {
        Some((left, desc)) => (left.trim(), Some(desc.trim())),
        None => (trimmed, None),
    };

    let (raw_name, raw_type, nullable) = match lhs.split_once(':') {
        Some((name_part, type_part)) => {
            let t = type_part.trim();
            if let Some(inner) = t.strip_suffix('?') {
                (name_part.trim(), inner.trim(), true)
            } else {
                (name_part.trim(), t, false)
            }
        }
        None => {
            if let Some(inner) = lhs.strip_suffix('?') {
                (inner.trim(), "string", true)
            } else {
                (lhs, "string", false)
            }
        }
    };

    if raw_name.is_empty() || raw_name.chars().any(char::is_whitespace) {
        return Err(Error::InvalidConfig(format!(
            "Invalid field name '{raw_name}' in --field '{raw}'. Field names cannot be empty or contain whitespace."
        )));
    }
    if raw_type.is_empty() {
        return Err(Error::InvalidConfig(format!(
            "Missing field type after ':' in --field '{raw}'."
        )));
    }

    let base_schema = parse_type_expr(raw_type, raw)?;
    let mut prop = if nullable {
        let mut map = Map::new();
        map.insert(
            "anyOf".to_string(),
            Value::Array(vec![base_schema, json!({"type": "null"})]),
        );
        map
    } else {
        match base_schema {
            Value::Object(map) => map,
            _ => unreachable!("type schemas are always JSON objects"),
        }
    };

    if let Some(desc) = description
        && !desc.is_empty()
    {
        prop.insert("description".to_string(), Value::String(desc.to_string()));
    }

    Ok((raw_name.to_string(), Value::Object(prop)))
}

fn parse_type_expr(expr: &str, full_spec: &str) -> Result<Value> {
    if let Some(inner) = expr.strip_suffix("[]") {
        let item_schema = parse_scalar_or_enum(inner.trim(), full_spec)?;
        return Ok(json!({"type": "array", "items": item_schema}));
    }
    if let Some(inner) = expr
        .strip_prefix("list<")
        .and_then(|s| s.strip_suffix('>'))
        .or_else(|| expr.strip_prefix("list[").and_then(|s| s.strip_suffix(']')))
        .or_else(|| {
            expr.strip_prefix("array<")
                .and_then(|s| s.strip_suffix('>'))
        })
        .or_else(|| {
            expr.strip_prefix("array[")
                .and_then(|s| s.strip_suffix(']'))
        })
    {
        let item_schema = parse_scalar_or_enum(inner.trim(), full_spec)?;
        return Ok(json!({"type": "array", "items": item_schema}));
    }
    parse_scalar_or_enum(expr, full_spec)
}

fn parse_scalar_or_enum(expr: &str, full_spec: &str) -> Result<Value> {
    if let Some(inner) = expr.strip_prefix("enum(").and_then(|s| s.strip_suffix(')')) {
        let sep = if inner.contains('|') { '|' } else { ',' };
        let values: Vec<Value> = inner
            .split(sep)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| Value::String(s.to_string()))
            .collect();
        if values.is_empty() {
            return Err(Error::InvalidConfig(format!(
                "Empty enum(...) in --field '{full_spec}'; provide at least one option."
            )));
        }
        return Ok(json!({"type": "string", "enum": values}));
    }

    match expr.to_ascii_lowercase().as_str() {
        "string" | "str" | "text" => Ok(json!({"type": "string"})),
        "number" | "float" | "decimal" => Ok(json!({"type": "number"})),
        "integer" | "int" => Ok(json!({"type": "integer"})),
        "boolean" | "bool" => Ok(json!({"type": "boolean"})),
        "date" => Ok(json!({"type": "string", "format": "date"})),
        "time" => Ok(json!({"type": "string", "format": "time"})),
        "datetime" | "date-time" => Ok(json!({"type": "string", "format": "date-time"})),
        "email" => Ok(json!({"type": "string", "format": "email"})),
        "uri" | "url" => Ok(json!({"type": "string", "format": "uri"})),
        "uuid" => Ok(json!({"type": "string", "format": "uuid"})),
        _ => Err(Error::InvalidConfig(format!(
            "Unsupported field type '{expr}' in --field '{full_spec}'. \
             Use string, number, integer, boolean, date, time, datetime, email, uri, uuid, \
             enum(a,b,...), or an array like string[] (append '?' for nullable)."
        ))),
    }
}
