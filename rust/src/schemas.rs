//! Adapt JSON Schema to provider dialects and validate model output locally.

use jsonschema::error::ValidationErrorKind;
use serde_json::{Map, Value, json};

use crate::error::{Error, Result};

const ANTHROPIC_FORMATS: &[&str] = &[
    "date-time",
    "time",
    "date",
    "duration",
    "email",
    "hostname",
    "uri",
    "ipv4",
    "ipv6",
    "uuid",
];

// Anthropic rejects these keywords, so they move into descriptions and are enforced locally.
const ANTHROPIC_UNSUPPORTED: &[&str] = &[
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "maxItems",
    "uniqueItems",
    "pattern",
];

/// A single sanitized schema validation failure (never contains raw invalid model values).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationFailure {
    pub path: String,
    pub message: String,
    pub kind: String,
}

/// Generate a JSON Schema for a Rust type implementing [`schemars::JsonSchema`].
pub fn schema_for_type<T: schemars::JsonSchema>() -> Result<Value> {
    let root = schemars::schema_for!(T);
    let mut value = serde_json::to_value(root)
        .map_err(|e| Error::InvalidConfig(format!("Failed to serialize JSON Schema: {e}")))?;
    if let Some(obj) = value.as_object_mut() {
        obj.shift_remove("$schema");
    }
    Ok(value)
}

/// Require every declared field and close fixed objects, as OpenAI strict output expects.
pub fn openai_strict(schema: &Value) -> Result<Value> {
    let mut adapted = schema.clone();
    adapt_openai_node(&mut adapted)?;
    Ok(adapted)
}

fn adapt_openai_node(node: &mut Value) -> Result<()> {
    let Some(obj) = node.as_object_mut() else {
        return Ok(());
    };
    obj.shift_remove("default");
    close_object(
        obj,
        "Strict JSON Schema output",
        "use Model(api='chat') for tool-based output",
    )?;
    if obj.get("type").and_then(Value::as_str) == Some("object") {
        let keys: Vec<Value> = obj
            .get("properties")
            .and_then(Value::as_object)
            .map(|props| props.keys().cloned().map(Value::String).collect())
            .unwrap_or_default();
        obj.insert("required".to_string(), Value::Array(keys));
    }
    visit_children_mut(obj, adapt_openai_node)
}

/// Close fixed objects, keep optional defaults, and describe constraints Anthropic rejects.
///
/// Maps, recursion, and unsupported compositions fail rather than silently changing meaning.
pub fn anthropic_json(schema: &Value) -> Result<Value> {
    let mut adapted = schema.clone();
    reject_recursion(&adapted, &adapted, &[])?;
    adapt_anthropic_node(&mut adapted)?;
    Ok(adapted)
}

fn adapt_anthropic_node(node: &mut Value) -> Result<()> {
    let Some(obj) = node.as_object_mut() else {
        return Ok(());
    };
    close_object(
        obj,
        "Anthropic JSON output",
        "use output_mode='tool' or a list of key/value objects",
    )?;
    for forbidden in ["allOf", "prefixItems", "patternProperties"] {
        if obj.contains_key(forbidden) {
            return Err(Error::InvalidConfig(
                "Anthropic JSON output does not support this schema composition.".to_string(),
            ));
        }
    }
    if let Some(enum_vals) = obj.get("enum").and_then(Value::as_array)
        && enum_vals.iter().any(|v| v.is_object() || v.is_array())
    {
        return Err(Error::InvalidConfig(
            "Anthropic JSON output does not support complex enum values.".to_string(),
        ));
    }

    let keys_to_move: Vec<String> = obj
        .iter()
        .filter(|(k, v)| is_local_constraint(k, v))
        .map(|(k, _)| k.clone())
        .collect();
    if !keys_to_move.is_empty() {
        let mut constraints = Map::new();
        for key in keys_to_move {
            if let Some(val) = obj.shift_remove(&key) {
                constraints.insert(key, val);
            }
        }
        let serialized = serde_json::to_string(&Value::Object(constraints)).unwrap_or_default();
        let existing = obj.get("description").and_then(Value::as_str).unwrap_or("");
        let described = format!("{existing} Constraints: {serialized}");
        obj.insert(
            "description".to_string(),
            Value::String(described.trim().to_string()),
        );
    }

    visit_children_mut(obj, adapt_anthropic_node)
}

fn close_object(obj: &mut Map<String, Value>, output: &str, alternative: &str) -> Result<()> {
    obj.shift_remove("discriminator");
    if let Some(one_of) = obj.shift_remove("oneOf") {
        obj.insert("anyOf".to_string(), one_of);
    }
    if obj.get("type").and_then(Value::as_str) == Some("object") {
        match obj.get("additionalProperties") {
            None | Some(Value::Bool(false)) => {}
            Some(_) => {
                return Err(Error::InvalidConfig(format!(
                    "{output} does not support open-ended dictionaries; {alternative}."
                )));
            }
        }
        obj.insert("additionalProperties".to_string(), Value::Bool(false));
    }
    Ok(())
}

fn is_local_constraint(keyword: &str, value: &Value) -> bool {
    if ANTHROPIC_UNSUPPORTED.contains(&keyword) {
        return true;
    }
    if keyword == "minItems" {
        return !matches!(value.as_u64(), Some(0 | 1));
    }
    if keyword == "format" {
        return value
            .as_str()
            .is_none_or(|fmt| !ANTHROPIC_FORMATS.contains(&fmt));
    }
    false
}

fn visit_children_mut<F>(obj: &mut Map<String, Value>, f: F) -> Result<()>
where
    F: Fn(&mut Value) -> Result<()> + Copy,
{
    for keyword in ["properties", "$defs", "definitions"] {
        if let Some(map) = obj.get_mut(keyword).and_then(Value::as_object_mut) {
            for child in map.values_mut() {
                f(child)?;
            }
        }
    }
    for keyword in ["items", "additionalProperties"] {
        if let Some(child) = obj.get_mut(keyword)
            && child.is_object()
        {
            f(child)?;
        }
    }
    for keyword in ["anyOf", "allOf", "oneOf", "prefixItems"] {
        if let Some(arr) = obj.get_mut(keyword).and_then(Value::as_array_mut) {
            for child in arr.iter_mut() {
                f(child)?;
            }
        }
    }
    Ok(())
}

fn collect_subschemas<'a>(node: &'a Value, out: &mut Vec<&'a Value>) {
    out.push(node);
    let Some(obj) = node.as_object() else {
        return;
    };
    for keyword in ["properties", "$defs", "definitions"] {
        if let Some(map) = obj.get(keyword).and_then(Value::as_object) {
            for child in map.values() {
                collect_subschemas(child, out);
            }
        }
    }
    for keyword in ["items", "additionalProperties"] {
        if let Some(child) = obj.get(keyword)
            && child.is_object()
        {
            collect_subschemas(child, out);
        }
    }
    for keyword in ["anyOf", "allOf", "oneOf", "prefixItems"] {
        if let Some(arr) = obj.get(keyword).and_then(Value::as_array) {
            for child in arr {
                collect_subschemas(child, out);
            }
        }
    }
}

fn reject_recursion(root: &Value, node: &Value, followed: &[String]) -> Result<()> {
    let mut subschemas = Vec::new();
    collect_subschemas(node, &mut subschemas);
    for schema in subschemas {
        let Some(r) = schema.get("$ref").and_then(Value::as_str) else {
            continue;
        };
        if !r.starts_with("#/") || followed.iter().any(|seen| seen == r) {
            return Err(Error::InvalidConfig(
                "Anthropic JSON output does not support external or recursive refs.".to_string(),
            ));
        }
        let target = resolve_ref(root, r)?;
        let mut next_followed = followed.to_vec();
        next_followed.push(r.to_string());
        reject_recursion(root, target, &next_followed)?;
    }
    Ok(())
}

/// Resolve a local JSON Pointer `$ref` (e.g. `#/$defs/Item`) against `root`.
pub fn resolve_ref<'a>(root: &'a Value, reference: &str) -> Result<&'a Value> {
    let pointer = reference.strip_prefix("#/").ok_or_else(|| {
        Error::InvalidConfig(format!(
            "External JSON Schema reference is not supported: '{reference}'."
        ))
    })?;
    let mut target = root;
    for raw_key in pointer.split('/') {
        let key = raw_key.replace("~1", "/").replace("~0", "~");
        target = target.get(&key).ok_or_else(|| {
            Error::InvalidConfig(format!(
                "Unresolvable JSON Schema reference: '{reference}'."
            ))
        })?;
    }
    Ok(target)
}

/// Validate a JSON instance against a JSON Schema locally, returning sanitized field errors.
pub fn validate_instance(
    schema: &Value,
    instance: &Value,
) -> std::result::Result<(), Vec<ValidationFailure>> {
    let validator = jsonschema::options()
        .should_validate_formats(true)
        .with_format("date", is_iso_date)
        .with_format("time", is_iso_time)
        .with_format("date-time", is_iso_datetime)
        .build(schema)
        .map_err(|e| {
            vec![ValidationFailure {
                path: "<root>".to_string(),
                message: format!("Invalid JSON Schema: {e}"),
                kind: "schema_error".to_string(),
            }]
        })?;

    let failures: Vec<ValidationFailure> = validator
        .iter_errors(instance)
        .map(|err| {
            let base_path = format_instance_path(err.instance_path.as_str());
            let (path, message, kind) = describe_error(&base_path, &err.kind);
            ValidationFailure {
                path,
                message,
                kind,
            }
        })
        .collect();

    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}

fn format_instance_path(pointer: &str) -> String {
    if pointer.is_empty() {
        return "<root>".to_string();
    }
    pointer
        .trim_start_matches('/')
        .split('/')
        .map(|s| s.replace("~1", "/").replace("~0", "~"))
        .collect::<Vec<_>>()
        .join(".")
}

fn join_path(base: &str, segment: &str) -> String {
    if base == "<root>" || base.is_empty() {
        segment.to_string()
    } else {
        format!("{base}.{segment}")
    }
}

/// Convert a `jsonschema::error::ValidationErrorKind` into a safe message without raw input values.
fn describe_error(base_path: &str, kind: &ValidationErrorKind) -> (String, String, String) {
    match kind {
        ValidationErrorKind::Required { property } => {
            let field = property.as_str().unwrap_or("");
            let path = if field.is_empty() {
                base_path.to_string()
            } else {
                join_path(base_path, field)
            };
            (path, "Field required".to_string(), "missing".to_string())
        }
        ValidationErrorKind::Type { kind: type_kind } => {
            let msg = match type_kind {
                jsonschema::error::TypeKind::Single(t) => {
                    format!("Input should be a valid {t}")
                }
                jsonschema::error::TypeKind::Multiple(types) => {
                    let names: Vec<String> = types.into_iter().map(|t| t.to_string()).collect();
                    format!("Input should match one of: {}", names.join(", "))
                }
            };
            (base_path.to_string(), msg, "type_error".to_string())
        }
        ValidationErrorKind::Minimum { limit } => (
            base_path.to_string(),
            format!("Input should be greater than or equal to {limit}"),
            "greater_than_equal".to_string(),
        ),
        ValidationErrorKind::Maximum { limit } => (
            base_path.to_string(),
            format!("Input should be less than or equal to {limit}"),
            "less_than_equal".to_string(),
        ),
        ValidationErrorKind::ExclusiveMinimum { limit } => (
            base_path.to_string(),
            format!("Input should be greater than {limit}"),
            "greater_than".to_string(),
        ),
        ValidationErrorKind::ExclusiveMaximum { limit } => (
            base_path.to_string(),
            format!("Input should be less than {limit}"),
            "less_than".to_string(),
        ),
        ValidationErrorKind::MultipleOf { multiple_of } => (
            base_path.to_string(),
            format!("Input should be a multiple of {multiple_of}"),
            "multiple_of".to_string(),
        ),
        ValidationErrorKind::MinLength { limit } => (
            base_path.to_string(),
            format!("String should have at least {limit} characters"),
            "string_too_short".to_string(),
        ),
        ValidationErrorKind::MaxLength { limit } => (
            base_path.to_string(),
            format!("String should have at most {limit} characters"),
            "string_too_long".to_string(),
        ),
        ValidationErrorKind::Pattern { pattern } => (
            base_path.to_string(),
            format!("String should match pattern '{pattern}'"),
            "string_pattern_mismatch".to_string(),
        ),
        ValidationErrorKind::MinItems { limit } => (
            base_path.to_string(),
            format!("List should have at least {limit} items"),
            "too_short".to_string(),
        ),
        ValidationErrorKind::MaxItems { limit } => (
            base_path.to_string(),
            format!("List should have at most {limit} items"),
            "too_long".to_string(),
        ),
        ValidationErrorKind::UniqueItems => (
            base_path.to_string(),
            "List items must be unique".to_string(),
            "unique_items".to_string(),
        ),
        ValidationErrorKind::MinProperties { limit } => (
            base_path.to_string(),
            format!("Object should have at least {limit} properties"),
            "min_properties".to_string(),
        ),
        ValidationErrorKind::MaxProperties { limit } => (
            base_path.to_string(),
            format!("Object should have at most {limit} properties"),
            "max_properties".to_string(),
        ),
        ValidationErrorKind::AdditionalProperties { unexpected } => {
            let names = unexpected.join(", ");
            (
                base_path.to_string(),
                format!("Extra inputs are not permitted ({names})"),
                "extra_forbidden".to_string(),
            )
        }
        ValidationErrorKind::Enum { .. } => (
            base_path.to_string(),
            "Input should be one of the allowed enum values".to_string(),
            "enum".to_string(),
        ),
        ValidationErrorKind::Constant { .. } => (
            base_path.to_string(),
            "Input does not match the required constant value".to_string(),
            "const".to_string(),
        ),
        ValidationErrorKind::Format { format } => (
            base_path.to_string(),
            format!("Input should be a valid {format}"),
            "format".to_string(),
        ),
        ValidationErrorKind::AnyOf
        | ValidationErrorKind::OneOfNotValid
        | ValidationErrorKind::OneOfMultipleValid => (
            base_path.to_string(),
            "Value does not match an allowed union schema".to_string(),
            "union_invalid".to_string(),
        ),
        _ => (
            base_path.to_string(),
            "Value did not satisfy a schema constraint".to_string(),
            "validation_error".to_string(),
        ),
    }
}

/// Validate `YYYY-MM-DD` with calendar month/day checks.
pub(crate) fn is_iso_date(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return false;
    }
    let Ok(year) = s[0..4].parse::<i32>() else {
        return false;
    };
    let Ok(month) = s[5..7].parse::<u32>() else {
        return false;
    };
    let Ok(day) = s[8..10].parse::<u32>() else {
        return false;
    };
    if !(1..=12).contains(&month) || day == 0 {
        return false;
    }
    let leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    let max_day = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    day <= max_day
}

/// Validate `HH:MM:SS[.frac][Z|+HH:MM|-HH:MM]` (allowing both naive and offset times).
pub(crate) fn is_iso_time(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() < 8 || b[2] != b':' || b[5] != b':' {
        return false;
    }
    let Ok(hour) = s[0..2].parse::<u32>() else {
        return false;
    };
    let Ok(minute) = s[3..5].parse::<u32>() else {
        return false;
    };
    let Ok(second) = s[6..8].parse::<u32>() else {
        return false;
    };
    if hour > 23 || minute > 59 || second > 60 {
        return false;
    }
    let mut rest = &s[8..];
    if let Some(after_dot) = rest.strip_prefix('.') {
        let digits = after_dot.bytes().take_while(u8::is_ascii_digit).count();
        if digits == 0 {
            return false;
        }
        rest = &after_dot[digits..];
    }
    if rest.is_empty() || rest == "Z" || rest == "z" {
        return true;
    }
    let Some(tz) = rest.strip_prefix('+').or_else(|| rest.strip_prefix('-')) else {
        return false;
    };
    let tb = tz.as_bytes();
    if tb.len() != 5 || tb[2] != b':' {
        return false;
    }
    let Ok(th) = tz[0..2].parse::<u32>() else {
        return false;
    };
    let Ok(tm) = tz[3..5].parse::<u32>() else {
        return false;
    };
    th <= 23 && tm <= 59
}

/// Validate ISO 8601 `date-time` (allowing both naive and offset timestamps).
pub(crate) fn is_iso_datetime(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() < 19 || !matches!(b[10], b'T' | b't' | b' ') {
        return false;
    }
    is_iso_date(&s[0..10]) && is_iso_time(&s[11..])
}

/// Ensure a root JSON Schema is a JSON object, returning an error if not.
pub fn ensure_schema_object(schema: &Value) -> Result<&Map<String, Value>> {
    schema
        .as_object()
        .ok_or_else(|| Error::InvalidConfig("Schema must be a JSON Schema object.".to_string()))
}

/// Build an empty object schema helper.
pub fn empty_object_schema() -> Value {
    json!({"type": "object", "properties": {}})
}
