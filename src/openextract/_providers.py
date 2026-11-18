"""Request builders and response readers for general structured-output APIs."""

import json
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel

from ._content import DOCUMENT_EXTENSIONS, PROMPT, Content
from ._schemas import Schema, anthropic_json, openai_strict
from .models import ResolvedModel

type Request = tuple[str, dict[str, Any]]

_TOOL_DESCRIPTION = "Return the extracted information."
_INCOMPLETE = "The model response was not completed."
_INCOMPLETE_TOOL_CALL = "The model did not complete the extraction tool call."
_WEB_IMAGES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
_CHAT_AUDIO = {"audio/mpeg": "mp3", "audio/wav": "wav"}
# Each provider documents its own Office and OpenDocument allowlist.
_RESPONSES_DOCUMENTS = {
    media: extension
    for media, extension in DOCUMENT_EXTENSIONS.items()
    if extension not in (".dotx", ".odp", ".ods")
}
_GEMINI_DOCUMENTS = DOCUMENT_EXTENSIONS.keys() - {"application/rtf"}
_GEMINI_OUTPUT_SETTINGS = {
    "responseMimeType",
    "responseJsonSchema",
    "responseSchema",
    "responseFormat",
}
# Top-level fields that carry the request structure in at least one API.
_STRUCTURE_FIELDS = {
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
}


def request(
    schema: type[BaseModel], model: ResolvedModel, content: Content, instructions: str | None
) -> Request:
    """Build the provider request, then merge the caller's generation settings into it."""
    build, _ = _ADAPTERS[model.api]
    path, body = build(model, schema.model_json_schema(), content, instructions)
    settings = dict(model.config.settings)
    if model.api == "gemini":
        if "spreadsheetMetadata" in settings:
            if DOCUMENT_EXTENSIONS.get(content.media_type) not in (".xlsx", ".xls", ".ods"):
                raise ValueError("spreadsheetMetadata requires an XLSX, XLS, or ODS input.")
            body["contents"][0]["parts"][-1]["spreadsheetMetadata"] = settings.pop(
                "spreadsheetMetadata"
            )
        _configure(body["generationConfig"], settings, _GEMINI_OUTPUT_SETTINGS, "structured output")
    else:
        # Anthropic's max_tokens is a default, not a field callers are kept from configuring.
        managed = (body.keys() | _STRUCTURE_FIELDS) - {"max_tokens"}
        _configure(body, settings, managed, "request fields")
    if body.get("stream"):
        raise ValueError("Streaming is not supported by extract; omit settings['stream'].")
    return path, body


def parse[T: BaseModel](schema: type[T], model: ResolvedModel, payload: dict[str, Any]) -> T:
    """Validate the structured output, rejecting refusals and incomplete generations."""
    _, read = _ADAPTERS[model.api]
    if model.config.output_mode == "json_schema":
        read = _JSON_READERS[model.api]
    output = read(payload)
    if not isinstance(output, (str, dict)) or output == "":
        raise ValueError("The model returned no structured output.")
    # Tool inputs arrive decoded; re-encode them so strict types keep JSON validation semantics.
    return schema.model_validate_json(json.dumps(output) if isinstance(output, dict) else output)


def _configure(
    target: dict[str, Any], settings: dict[str, Any], managed: set[str], kind: str
) -> None:
    if overridden := managed & settings.keys():
        raise ValueError(f"settings cannot override {kind}: {sorted(overridden)}.")
    target.update(settings)


def _responses(
    model: ResolvedModel, schema: Schema, content: Content, instructions: str | None
) -> Request:
    if content.is_text:
        document = {"type": "input_text", "text": content.text}
    elif content.is_image:
        document = {"type": "input_image", "image_url": content.data_uri}
    elif extension := _RESPONSES_DOCUMENTS.get(content.media_type):
        filename = "document" + extension
        document = {"type": "input_file", "file_data": content.data_uri, "filename": filename}
    else:
        raise _unsupported("responses", content)
    strict_schema = openai_strict(schema)
    output = {"type": "json_schema", "name": "extract", "schema": strict_schema, "strict": True}
    body = {
        "model": model.model_name,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": PROMPT}, document]}],
        "text": {"format": output},
    }
    if instructions:
        body["instructions"] = instructions
    return "/responses", body


def _chat(
    model: ResolvedModel, schema: Schema, content: Content, instructions: str | None
) -> Request:
    messages = [*_system(instructions), _user(PROMPT, _chat_document(content, "chat"))]
    body: dict[str, Any] = {"model": model.model_name, "messages": messages}
    if model.config.output_mode == "json_schema":
        output = {"name": "extract", "strict": True, "schema": openai_strict(schema)}
        body["response_format"] = {"type": "json_schema", "json_schema": output}
    else:
        tool = {"name": "extract", "description": _TOOL_DESCRIPTION, "parameters": schema}
        body["tools"] = [{"type": "function", "function": tool}]
        body["tool_choice"] = {"type": "function", "function": {"name": "extract"}}
    return "/chat/completions", body


def _cohere(
    model: ResolvedModel, schema: Schema, content: Content, instructions: str | None
) -> Request:
    prompt = PROMPT + " Return the extracted information as JSON."
    messages = [*_system(instructions), _user(prompt, _chat_document(content, "cohere"))]
    output = {"type": "json_object", "schema": schema}
    return "/chat", {"model": model.model_name, "messages": messages, "response_format": output}


def _anthropic(
    model: ResolvedModel, schema: Schema, content: Content, instructions: str | None
) -> Request:
    if content.is_text:
        document = _text(content.text)
    elif content.is_pdf or content.media_type in _WEB_IMAGES:
        source = {"type": "base64", "media_type": content.media_type, "data": content.base64}
        document = {"type": "document" if content.is_pdf else "image", "source": source}
    else:
        raise _unsupported("anthropic", content)
    body = {"model": model.model_name, "max_tokens": 4096, "messages": [_user(PROMPT, document)]}
    if instructions:
        body["system"] = instructions
    if model.config.output_mode == "json_schema":
        output = {"type": "json_schema", "schema": anthropic_json(schema)}
        body["output_config"] = {"format": output}
    else:
        tool = {"name": "extract", "description": _TOOL_DESCRIPTION, "input_schema": schema}
        body["tools"] = [tool]
        body["tool_choice"] = {"type": "tool", "name": "extract"}
    return "/messages", body


def _gemini(
    model: ResolvedModel, schema: Schema, content: Content, instructions: str | None
) -> Request:
    media = content.media_type
    if content.is_text:
        document = {"text": content.text}
    elif media.startswith(("image/", "audio/", "video/")) or media in _GEMINI_DOCUMENTS:
        document = {"inlineData": {"mimeType": media, "data": content.base64}}
    else:
        raise _unsupported("gemini", content)
    body = {
        "contents": [{"role": "user", "parts": [{"text": PROMPT}, document]}],
        "generationConfig": {"responseMimeType": "application/json", "responseJsonSchema": schema},
    }
    if instructions:
        body["systemInstruction"] = {"parts": [{"text": instructions}]}
    return f"/models/{quote(model.model_name, safe='')}:generateContent", body


def _ollama(
    model: ResolvedModel, schema: Schema, content: Content, instructions: str | None
) -> Request:
    message: dict[str, Any] = {"role": "user", "content": PROMPT}
    if content.is_text:
        message["content"] += "\n\n" + content.text
    elif content.is_image:
        message["images"] = [content.base64]
    else:
        raise _unsupported("ollama", content)
    messages = [*_system(instructions), message]
    body = {"model": model.model_name, "messages": messages, "format": schema, "stream": False}
    return "/api/chat", body


def _mistral_ocr(
    model: ResolvedModel, schema: Schema, content: Content, instructions: str | None
) -> Request:
    if not content.is_pdf:
        raise ValueError("The mistral-ocr adapter supports only PDF inputs.")
    output = {"name": "extract", "schema": schema, "strict": True}
    return "/ocr", {
        "model": model.model_name,
        "document": {"type": "document_url", "document_url": content.data_uri},
        "document_annotation_format": {"type": "json_schema", "json_schema": output},
        "document_annotation_prompt": f"{PROMPT}\n{instructions or ''}".strip(),
        "bbox_annotation_format": None,
        "include_image_base64": False,
    }


def _bedrock(
    model: ResolvedModel, schema: Schema, content: Content, instructions: str | None
) -> Request:
    source = {"bytes": content.base64}
    if content.is_text:
        document = {"text": content.text}
    elif content.media_type in _WEB_IMAGES:
        document = {"image": {"format": content.media_type.split("/")[1], "source": source}}
    elif content.is_pdf:
        document = {"document": {"format": "pdf", "name": "document", "source": source}}
    else:
        raise _unsupported("bedrock", content)
    tool = {"name": "extract", "description": _TOOL_DESCRIPTION, "inputSchema": {"json": schema}}
    body = {
        "messages": [{"role": "user", "content": [{"text": PROMPT}, document]}],
        "toolConfig": {"tools": [{"toolSpec": tool}], "toolChoice": {"tool": {"name": "extract"}}},
    }
    if instructions:
        body["system"] = [{"text": instructions}]
    return f"/model/{quote(model.model_name, safe='')}/converse", body


def _chat_document(content: Content, api: str) -> dict[str, Any]:
    """Encode a source as Chat Completions content; Cohere accepts only text and images."""
    if content.is_text:
        return _text(content.text)
    if content.is_image:
        return {"type": "image_url", "image_url": {"url": content.data_uri}}
    if api == "chat" and content.is_pdf:
        file = {"filename": "document.pdf", "file_data": content.data_uri}
        return {"type": "file", "file": file}
    if api == "chat" and content.media_type in _CHAT_AUDIO:
        audio = {"data": content.base64, "format": _CHAT_AUDIO[content.media_type]}
        return {"type": "input_audio", "input_audio": audio}
    raise _unsupported(api, content)


def _text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _user(prompt: str, document: dict[str, Any]) -> dict[str, Any]:
    return {"role": "user", "content": [_text(prompt), document]}


def _system(instructions: str | None) -> list[dict[str, Any]]:
    return [{"role": "system", "content": instructions}] if instructions else []


def _unsupported(api: str, content: Content) -> ValueError:
    return ValueError(f"The {api} adapter does not support {content.media_type} inputs.")


def _read_responses(payload: dict[str, Any]) -> Any:
    if payload.get("status") != "completed":
        raise ValueError(_INCOMPLETE)
    parts = [
        part for item in payload["output"] if item["type"] == "message" for part in item["content"]
    ]
    if any(part["type"] == "refusal" for part in parts):
        raise ValueError("The model refused to extract the document.")
    return "".join(part["text"] for part in parts if part["type"] == "output_text")


def _read_chat(payload: dict[str, Any]) -> Any:
    choice = payload["choices"][0]
    message = choice["message"]
    # A forced function call can report a natural stop even with complete tool arguments.
    if choice["finish_reason"] not in ("tool_calls", "stop") or message.get("refusal"):
        raise ValueError(_INCOMPLETE_TOOL_CALL)
    return _tool_call([call["function"] for call in message.get("tool_calls", [])], "arguments")


def _read_chat_json(payload: dict[str, Any]) -> Any:
    choice = payload["choices"][0]
    message = choice["message"]
    if choice["finish_reason"] != "stop" or message.get("refusal") or message.get("tool_calls"):
        raise ValueError("The model did not complete a JSON Schema response.")
    if not isinstance(message["content"], str):
        raise ValueError("Expected JSON text in the model response.")
    return message["content"]


def _read_anthropic(payload: dict[str, Any]) -> Any:
    if payload["stop_reason"] != "tool_use":
        raise ValueError(_INCOMPLETE_TOOL_CALL)
    return _tool_call([part for part in payload["content"] if part["type"] == "tool_use"], "input")


def _read_anthropic_json(payload: dict[str, Any]) -> Any:
    if payload["stop_reason"] != "end_turn":
        raise ValueError("The model refused or did not complete a JSON Schema response.")
    parts = payload["content"]
    if any(part["type"] == "tool_use" for part in parts):
        raise ValueError("Unexpected tool call in a JSON Schema response.")
    texts = [part["text"] for part in parts if part["type"] == "text"]
    if len(texts) != 1 or not isinstance(texts[0], str):
        raise ValueError("Expected one JSON text block in the model response.")
    return texts[0]


def _read_gemini(payload: dict[str, Any]) -> Any:
    candidate = payload["candidates"][0]
    if candidate["finishReason"] != "STOP":
        raise ValueError("The model response was blocked or incomplete.")
    parts = candidate["content"]["parts"]
    return "".join(part["text"] for part in parts if "text" in part and not part.get("thought"))


def _read_ollama(payload: dict[str, Any]) -> Any:
    if payload.get("done") is not True or payload.get("done_reason", "stop") != "stop":
        raise ValueError(_INCOMPLETE)
    return payload["message"]["content"]


def _read_cohere(payload: dict[str, Any]) -> Any:
    if payload["finish_reason"].lower() != "complete":
        raise ValueError(_INCOMPLETE)
    return "".join(part["text"] for part in payload["message"]["content"] if part["type"] == "text")


def _read_bedrock(payload: dict[str, Any]) -> Any:
    if payload["stopReason"] != "tool_use":
        raise ValueError(_INCOMPLETE_TOOL_CALL)
    parts = payload["output"]["message"]["content"]
    return _tool_call([part["toolUse"] for part in parts if "toolUse" in part], "input")


def _read_mistral_ocr(payload: dict[str, Any]) -> Any:
    return payload["document_annotation"]


def _tool_call(calls: list[dict[str, Any]], arguments: str) -> Any:
    """Return the arguments of the one forced ``extract`` call."""
    if len(calls) != 1 or calls[0]["name"] != "extract":
        raise ValueError("Expected exactly one extraction tool call.")
    return calls[0][arguments]


# API: (request builder, default response reader).
_ADAPTERS = {
    "responses": (_responses, _read_responses),
    "chat": (_chat, _read_chat),
    "anthropic": (_anthropic, _read_anthropic),
    "gemini": (_gemini, _read_gemini),
    "ollama": (_ollama, _read_ollama),
    "cohere": (_cohere, _read_cohere),
    "bedrock": (_bedrock, _read_bedrock),
    "mistral-ocr": (_mistral_ocr, _read_mistral_ocr),
}
_JSON_READERS = {"chat": _read_chat_json, "anthropic": _read_anthropic_json}
