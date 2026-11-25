//! Fill boolean and enum schemas through closed-set Decisions APIs.
//!
//! Each field becomes one question. OpenRouter and Gemini share a named question and
//! answer format; OpenAI uses lists and its own answer types.

use serde_json::{Map, Value, json};

use crate::content::{Content, PROMPT};
use crate::error::{Error, Result};
use crate::models::{Api, ResolvedModel};
use crate::schemas::{ensure_schema_object, resolve_ref};

const YES_THRESHOLD: f64 = 0.5;
const GEMINI_IMAGES: &[&str] = &[
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/heic",
    "image/heif",
    "image/gif",
    "image/bmp",
    "image/tiff",
];

#[derive(Debug, Clone)]
enum DecisionField {
    Bool {
        instructions: String,
    },
    Choice {
        instructions: String,
        /// Ordered list of `(wire_key, original_json_value)` pairs.
        options: Vec<(String, Value)>,
    },
}

/// Build the Decisions request path and JSON body.
pub(crate) fn request(
    schema: &Value,
    model: &ResolvedModel,
    content: &Content,
    instructions: Option<&str>,
) -> Result<(String, Value)> {
    let fields = extract_decision_fields(schema)?;
    let body = match model.api {
        Api::OpenRouterDecisions => build_openrouter(&fields, content, instructions)?,
        Api::GeminiDecisions => build_gemini(&fields, content, instructions)?,
        Api::OpenAiDecisions => build_openai(&fields, content, instructions)?,
        other => {
            return Err(Error::InvalidConfig(format!(
                "Unsupported decisions API: '{other}'."
            )));
        }
    };
    if !model.config.settings.is_empty() {
        return Err(Error::InvalidConfig(
            "Model settings are not supported by decisions adapters.".to_string(),
        ));
    }
    let mut full = Map::new();
    full.insert("model".to_string(), Value::String(model.model_name.clone()));
    for (k, v) in body {
        full.insert(k, v);
    }
    Ok(("/decisions".to_string(), Value::Object(full)))
}

/// Parse one answer per schema field into a JSON object ready for schema validation.
pub(crate) fn parse_answers(
    schema: &Value,
    model: &ResolvedModel,
    payload: &Map<String, Value>,
) -> std::result::Result<Value, String> {
    let is_openai = model.api == Api::OpenAiDecisions;
    let fields = extract_decision_fields(schema).map_err(|e| e.to_string())?;
    let raw_answers = payload
        .get("answers")
        .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
    let answers = answer_map(raw_answers, is_openai)?;

    if answers.len() != fields.len() || !fields.iter().all(|(name, _)| answers.contains_key(name)) {
        return Err("Expected exactly one answer for every schema field.".to_string());
    }

    let mut values = Map::new();
    for (name, field) in &fields {
        let answer = &answers[name];
        let val = parse_single_answer(name, field, answer, is_openai)?;
        values.insert(name.clone(), val);
    }
    Ok(Value::Object(values))
}

fn build_openrouter(
    fields: &[(String, DecisionField)],
    content: &Content,
    instructions: Option<&str>,
) -> Result<Map<String, Value>> {
    require_text_or_image(content)?;
    let document = if content.is_text() {
        Value::String(content.text().into_owned())
    } else {
        json!({"type": "image_url", "image_url": {"url": content.data_uri()}})
    };
    let mut state: Vec<Value> = guidance(instructions)
        .into_iter()
        .map(Value::String)
        .collect();
    state.push(document);

    let mut body = Map::new();
    body.insert("state".to_string(), Value::Array(state));
    body.insert(
        "questions".to_string(),
        Value::Object(named_questions(fields)),
    );
    Ok(body)
}

fn build_gemini(
    fields: &[(String, DecisionField)],
    content: &Content,
    instructions: Option<&str>,
) -> Result<Map<String, Value>> {
    let questions = named_questions(fields);
    check_gemini_limits(fields)?;

    let document = if content.is_text() {
        json!({"type": "text", "text": content.text()})
    } else if content.is_pdf() || GEMINI_IMAGES.contains(&content.media_type.as_str()) {
        let kind = if content.is_pdf() {
            "document"
        } else {
            "image"
        };
        json!({"type": kind, "data": content.base64(), "mime_type": content.media_type})
    } else {
        return Err(Error::InvalidConfig(format!(
            "Gemini Decisions supports text, images and PDF, not {}.",
            content.media_type
        )));
    };

    let mut input: Vec<Value> = guidance(instructions)
        .into_iter()
        .map(|text| json!({"type": "text", "text": format!("{text}\n")}))
        .collect();
    input.push(document);

    let mut body = Map::new();
    body.insert("input".to_string(), Value::Array(input));
    body.insert("questions".to_string(), Value::Object(questions));
    Ok(body)
}

fn build_openai(
    fields: &[(String, DecisionField)],
    content: &Content,
    instructions: Option<&str>,
) -> Result<Map<String, Value>> {
    if !(1..=400).contains(&fields.len()) {
        return Err(Error::InvalidConfig(
            "OpenAI Decisions requires between 1 and 400 schema fields.".to_string(),
        ));
    }
    require_text_or_image(content)?;

    let questions: Vec<Value> = fields
        .iter()
        .map(|(name, field)| openai_question(name, field))
        .collect();

    let document = if content.is_text() {
        json!({"type": "input_text", "text": content.text()})
    } else {
        json!({"type": "input_image", "image_url": content.data_uri()})
    };

    let developer_content = match instructions {
        Some(instr) if !instr.is_empty() => instr,
        _ => PROMPT,
    };

    let mut body = Map::new();
    body.insert(
        "input".to_string(),
        json!([
            {"role": "developer", "content": developer_content},
            {"role": "user", "content": [{"type": "input_text", "text": PROMPT}, document]},
        ]),
    );
    body.insert("questions".to_string(), Value::Array(questions));
    Ok(body)
}

fn check_gemini_limits(fields: &[(String, DecisionField)]) -> Result<()> {
    if !(1..=1000).contains(&fields.len()) {
        return Err(Error::InvalidConfig(
            "Gemini Decisions requires between 1 and 1000 schema fields.".to_string(),
        ));
    }
    for (name, field) in fields {
        let valid_name = !name.is_empty()
            && name.len() <= 64
            && name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
        if !valid_name {
            return Err(Error::InvalidConfig(
                "Gemini Decisions field names must be 1–64 ASCII letters, digits, _ or -."
                    .to_string(),
            ));
        }
        if let DecisionField::Choice { options, .. } = field {
            if !(2..=255).contains(&options.len()) {
                return Err(Error::InvalidConfig(
                    "Gemini Decisions requires between 2 and 255 choices per field.".to_string(),
                ));
            }
            if !options.iter().all(|(key, _)| is_gemini_option(key)) {
                return Err(Error::InvalidConfig(
                    "Invalid Gemini Decisions option key.".to_string(),
                ));
            }
        }
    }
    Ok(())
}

fn is_gemini_option(key: &str) -> bool {
    let char_count = key.chars().count();
    (1..=128).contains(&char_count)
        && key.len() <= 512
        && key == key.trim()
        && key.chars().all(is_printable_char)
}

/// Match Python's `str.isprintable()`: disallow control chars, non-ASCII whitespace/separators,
/// and invisible Unicode format characters.
fn is_printable_char(c: char) -> bool {
    if c == ' ' {
        return true;
    }
    if c.is_control() || c.is_whitespace() {
        return false;
    }
    !matches!(
        c,
        '\u{00AD}'
            | '\u{200B}'..='\u{200F}'
            | '\u{202A}'..='\u{202E}'
            | '\u{2060}'..='\u{206F}'
            | '\u{FEFF}'
    )
}

fn named_questions(fields: &[(String, DecisionField)]) -> Map<String, Value> {
    let mut map = Map::new();
    for (name, field) in fields {
        let question = match field {
            DecisionField::Bool { instructions } => {
                json!({"type": "noul", "instructions": instructions})
            }
            DecisionField::Choice {
                instructions,
                options,
            } => {
                let mut criteria = Map::new();
                for (key, _) in options {
                    criteria.insert(key.clone(), Value::String(key.clone()));
                }
                json!({
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": Value::Object(criteria),
                })
            }
        };
        map.insert(name.clone(), question);
    }
    map
}

fn openai_question(name: &str, field: &DecisionField) -> Value {
    match field {
        DecisionField::Bool { instructions } => {
            json!({"name": name, "type": "predicate", "instructions": instructions})
        }
        DecisionField::Choice {
            instructions,
            options,
        } => {
            let choices: Vec<Value> = options
                .iter()
                .map(|(key, _)| json!({"value": key, "description": key}))
                .collect();
            json!({
                "name": name,
                "type": "choice",
                "instructions": instructions,
                "choices": choices,
            })
        }
    }
}

fn guidance(instructions: Option<&str>) -> Vec<String> {
    match instructions {
        Some(instr) if !instr.is_empty() => vec![PROMPT.to_string(), instr.to_string()],
        _ => vec![PROMPT.to_string()],
    }
}

fn require_text_or_image(content: &Content) -> Result<()> {
    if !(content.is_text() || content.is_image()) {
        return Err(Error::InvalidConfig(format!(
            "Decisions models read text and images, not {}.",
            content.media_type
        )));
    }
    Ok(())
}

fn extract_decision_fields(schema: &Value) -> Result<Vec<(String, DecisionField)>> {
    let obj = ensure_schema_object(schema)?;
    let props = obj
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    let mut result = Vec::with_capacity(props.len());
    for (name, prop) in &props {
        let field = parse_decision_field(schema, name, prop)?;
        result.push((name.clone(), field));
    }
    Ok(result)
}

fn parse_decision_field(root: &Value, name: &str, prop: &Value) -> Result<DecisionField> {
    let prop_desc = prop.get("description").and_then(Value::as_str);
    let target = resolve_property_target(root, prop)?;
    let target_desc = target.get("description").and_then(Value::as_str);
    let instructions = prop_desc
        .or(target_desc)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| name.replace('_', " "));

    let Some(target_obj) = target.as_object() else {
        return Err(unsupported_decision_field(name));
    };

    // Check enum / const / oneOf-const first.
    if let Some(options) = extract_choice_options(target_obj)? {
        return Ok(DecisionField::Choice {
            instructions,
            options,
        });
    }

    if target_obj.get("type").and_then(Value::as_str) == Some("boolean") {
        return Ok(DecisionField::Bool { instructions });
    }

    Err(unsupported_decision_field(name))
}

fn resolve_property_target<'a>(root: &'a Value, prop: &'a Value) -> Result<&'a Value> {
    let mut current = prop;
    let mut depth = 0;
    loop {
        if depth > 16 {
            return Err(Error::InvalidConfig(
                "Recursive JSON Schema reference in Decisions schema.".to_string(),
            ));
        }
        if let Some(r) = current.get("$ref").and_then(Value::as_str) {
            current = resolve_ref(root, r)?;
            depth += 1;
            continue;
        }
        if let Some(all_of) = current.get("allOf").and_then(Value::as_array)
            && all_of.len() == 1
            && let Some(r) = all_of[0].get("$ref").and_then(Value::as_str)
        {
            current = resolve_ref(root, r)?;
            depth += 1;
            continue;
        }
        return Ok(current);
    }
}

fn extract_choice_options(obj: &Map<String, Value>) -> Result<Option<Vec<(String, Value)>>> {
    if let Some(values) = obj.get("enum").and_then(Value::as_array) {
        let mut options = Vec::with_capacity(values.len());
        for val in values {
            let key = option_wire_key(val)?;
            options.push((key, val.clone()));
        }
        return Ok(Some(options));
    }
    if let Some(constant) = obj.get("const") {
        let key = option_wire_key(constant)?;
        return Ok(Some(vec![(key, constant.clone())]));
    }
    for union_key in ["oneOf", "anyOf"] {
        if let Some(branches) = obj.get(union_key).and_then(Value::as_array)
            && !branches.is_empty()
        {
            let mut options = Vec::new();
            let mut all_const_or_enum = true;
            for branch in branches {
                let Some(bobj) = branch.as_object() else {
                    all_const_or_enum = false;
                    break;
                };
                if let Some(c) = bobj.get("const") {
                    options.push((option_wire_key(c)?, c.clone()));
                } else if let Some(arr) = bobj.get("enum").and_then(Value::as_array) {
                    for val in arr {
                        options.push((option_wire_key(val)?, val.clone()));
                    }
                } else {
                    all_const_or_enum = false;
                    break;
                }
            }
            if all_const_or_enum && !options.is_empty() {
                return Ok(Some(options));
            }
        }
    }
    Ok(None)
}

fn option_wire_key(val: &Value) -> Result<String> {
    match val {
        Value::String(s) => Ok(s.clone()),
        Value::Number(n) => Ok(n.to_string()),
        Value::Bool(b) => Ok(if *b { "True" } else { "False" }.to_string()),
        _ => Err(Error::InvalidConfig(
            "Decisions enum options must be scalar values.".to_string(),
        )),
    }
}

fn unsupported_decision_field(name: &str) -> Error {
    Error::InvalidConfig(format!(
        "Decisions models only fill bool, Literal, and Enum fields; '{name}' is not supported."
    ))
}

fn answer_map(
    raw_answers: &Value,
    is_openai: bool,
) -> std::result::Result<Map<String, Value>, String> {
    if !is_openai {
        return raw_answers
            .as_object()
            .cloned()
            .ok_or_else(|| "Expected an object of decision answers.".to_string());
    }
    let items = raw_answers
        .as_array()
        .ok_or_else(|| "Expected a list of decision answers.".to_string())?;
    let mut map = Map::new();
    for item in items {
        let obj = item
            .as_object()
            .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
        let name = obj
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
        if map.insert(name.to_string(), item.clone()).is_some() {
            return Err("Expected at most one answer per schema field.".to_string());
        }
    }
    Ok(map)
}

fn parse_single_answer(
    name: &str,
    field: &DecisionField,
    answer: &Value,
    is_openai: bool,
) -> std::result::Result<Value, String> {
    let obj = answer
        .as_object()
        .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
    let answer_type = obj
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| "The provider returned an unexpected response structure.".to_string())?;
    if answer_type == "refusal" {
        return Err(format!("The model refused to answer '{name}'."));
    }

    match field {
        DecisionField::Bool { .. } => {
            let (yes_type, prob_key) = if is_openai {
                ("predicate", "probability")
            } else {
                ("noul", "noul")
            };
            let prob_val = obj.get(prob_key).ok_or_else(|| {
                "The provider returned an unexpected response structure.".to_string()
            })?;
            let Some(chance) = prob_val.as_f64() else {
                return Err(format!(
                    "Expected a yes/no answer with a numeric probability from 0 to 1 for '{name}'."
                ));
            };
            if answer_type != yes_type || !(0.0..=1.0).contains(&chance) {
                return Err(format!(
                    "Expected a yes/no answer with a numeric probability from 0 to 1 for '{name}'."
                ));
            }
            Ok(Value::Bool(chance >= YES_THRESHOLD))
        }
        DecisionField::Choice { options, .. } => {
            if answer_type != "choice" {
                return Err(format!("Expected a choice answer for '{name}'."));
            }
            let choice = obj.get("choice").and_then(Value::as_str);
            if let Some(choice_str) = choice
                && let Some((_, val)) = options.iter().find(|(k, _)| k == choice_str)
            {
                return Ok(val.clone());
            }
            let keys: Vec<String> = options.iter().map(|(k, _)| format!("'{k}'")).collect();
            Err(format!(
                "Expected one of [{}] for field '{name}'.",
                keys.join(", ")
            ))
        }
    }
}
