"""Request builders and response readers for general structured-output APIs."""

import json
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel

from ._content import DOCUMENT_EXTENSIONS, PROMPT, Content
from ._schemas import Schema, anthropic_json, openai_strict
from .models import ResolvedModel, merge_settings

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
_GEMINI_SPREADSHEETS = {
    media
    for media, extension in DOCUMENT_EXTENSIONS.items()
    if extension in (".xlsx", ".xls", ".ods")
}
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
    if model.api != "gemini":
        # Anthropic's max_tokens is a default, not a field callers are kept from configuring.
        managed = (body.keys() | _STRUCTURE_FIELDS) - {"max_tokens"}
        merge_settings(body, model.config.settings, managed)
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


def _build_responses(
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
        raise _unsupported_media_error("responses", content)
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


def _read_responses(payload: dict[str, Any]) -> Any:
    if payload.get("status") != "completed":
        raise ValueError(_INCOMPLETE)
    parts = [
        part
        for message in payload["output"]
        if message["type"] == "message"
        for part in message["content"]
    ]
    if any(part["type"] == "refusal" for part in parts):
        raise ValueError("The model refused to extract the document.")
    return "".join(part["text"] for part in parts if part["type"] == "output_text")


def _build_chat(
    model: ResolvedModel, schema: Schema, content: Content, instructions: str | None
) -> Request:
    document = _chat_document(content, "chat")
    messages = [*_system_messages(instructions), _user_message(PROMPT, document)]
    body: dict[str, Any] = {"model": model.model_name, "messages": messages}
    if model.config.output_mode == "json_schema":
        output = {"name": "extract", "strict": True, "schema": openai_strict(schema)}
        body["response_format"] = {"type": "json_schema", "json_schema": output}
    else:
        tool = {"name": "extract", "description": _TOOL_DESCRIPTION, "parameters": schema}
        body["tools"] = [{"type": "function", "function": tool}]
        body["tool_choice"] = {"type": "function", "function": {"name": "extract"}}
    return "/chat/completions", body


def _read_chat(payload: dict[str, Any]) -> Any:
    choice = payload["choices"][0]
    message = choice["message"]
    # A forced function call can report a natural stop even with complete tool arguments.
    if choice["finish_reason"] not in ("tool_calls", "stop") or message.get("refusal"):
        raise ValueError(_INCOMPLETE_TOOL_CALL)
    calls = [call["function"] for call in message.get("tool_calls", [])]
    return _extract_tool_arguments(calls, "arguments")


def _read_chat_json(payload: dict[str, Any]) -> Any:
    choice = payload["choices"][0]
    message = choice["message"]
    if choice["finish_reason"] != "stop" or message.get("refusal") or message.get("tool_calls"):
        raise ValueError("The model did not complete a JSON Schema response.")
    if not isinstance(message["content"], str):
        raise ValueError("Expected JSON text in the model response.")
    return message["content"]


def _build_cohere(
    model: ResolvedModel, schema: Schema, content: Content, instructions: str | None
) -> Request:
    prompt = PROMPT + " Return the extracted information as JSON."
    document = _chat_document(content, "cohere")
    messages = [*_system_messages(instructions), _user_message(prompt, document)]
    output = {"type": "json_object", "schema": schema}
    return "/chat", {"model": model.model_name, "messages": messages, "response_format": output}


def _read_cohere(payload: dict[str, Any]) -> Any:
    if payload["finish_reason"].lower() != "complete":
        raise ValueError(_INCOMPLETE)
    return "".join(part["text"] for part in payload["message"]["content"] if part["type"] == "text")


def _build_anthropic(
    model: ResolvedModel, schema: Schema, content: Content, instructions: str | None
) -> Request:
    if content.is_text:
        document = _text_block(content.text)
    elif content.is_pdf or content.media_type in _WEB_IMAGES:
        source = {"type": "base64", "media_type": content.media_type, "data": content.base64}
        document = {"type": "document" if content.is_pdf else "image", "source": source}
    else:
        raise _unsupported_media_error("anthropic", content)
    body = {
        "model": model.model_name,
        "max_tokens": 4096,
        "messages": [_user_message(PROMPT, document)],
    }
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


def _read_anthropic(payload: dict[str, Any]) -> Any:
    if payload["stop_reason"] != "tool_use":
        raise ValueError(_INCOMPLETE_TOOL_CALL)
    calls = [part for part in payload["content"] if part["type"] == "tool_use"]
    return _extract_tool_arguments(calls, "input")


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


def _build_gemini(
    model: ResolvedModel, schema: Schema, content: Content, instructions: str | None
) -> Request:
    media = content.media_type
    if content.is_text:
        document: dict[str, Any] = {"text": content.text}
    elif media.startswith(("image/", "audio/", "video/")) or media in _GEMINI_DOCUMENTS:
        document = {"inlineData": {"mimeType": media, "data": content.base64}}
    else:
        raise _unsupported_media_error("gemini", content)
    settings = dict(model.config.settings)
    if "documentMetadata" in settings:
        if media not in _GEMINI_DOCUMENTS - _GEMINI_SPREADSHEETS:
            raise ValueError("documentMetadata requires a PDF, document, or presentation input.")
        document["documentMetadata"] = settings.pop("documentMetadata")
    if "spreadsheetMetadata" in settings:
        if media not in _GEMINI_SPREADSHEETS:
            raise ValueError("spreadsheetMetadata requires an XLSX, XLS, or ODS input.")
        document["spreadsheetMetadata"] = settings.pop("spreadsheetMetadata")
    generation_config = {"responseMimeType": "application/json", "responseJsonSchema": schema}
    merge_settings(generation_config, settings, _GEMINI_OUTPUT_SETTINGS, "structured output")
    body: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": PROMPT}, document]}],
        "generationConfig": generation_config,
    }
    if instructions:
        body["systemInstruction"] = {"parts": [{"text": instructions}]}
    return f"/models/{quote(model.model_name, safe='')}:generateContent", body


def _read_gemini(payload: dict[str, Any]) -> Any:
    candidate = payload["candidates"][0]
    if candidate["finishReason"] != "STOP":
        raise ValueError("The model response was blocked or incomplete.")
    parts = candidate["content"]["parts"]
    return "".join(part["text"] for part in parts if "text" in part and not part.get("thought"))


def _build_ollama(
    model: ResolvedModel, schema: Schema, content: Content, instructions: str | None
) -> Request:
    message: dict[str, Any] = {"role": "user", "content": PROMPT}
    if content.is_text:
        message["content"] += "\n\n" + content.text
    elif content.is_image:
        message["images"] = [content.base64]
    else:
        raise _unsupported_media_error("ollama", content)
    messages = [*_system_messages(instructions), message]
    body = {"model": model.model_name, "messages": messages, "format": schema, "stream": False}
    return "/api/chat", body


def _read_ollama(payload: dict[str, Any]) -> Any:
    if payload.get("done") is not True or payload.get("done_reason", "stop") != "stop":
        raise ValueError(_INCOMPLETE)
    return payload["message"]["content"]


def _build_mistral_ocr(
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


def _read_mistral_ocr(payload: dict[str, Any]) -> Any:
    return payload["document_annotation"]


def _build_bedrock(
    model: ResolvedModel, schema: Schema, content: Content, instructions: str | None
) -> Request:
    if content.is_text:
        document: dict[str, Any] = {"text": content.text}
    elif content.media_type in _WEB_IMAGES:
        image_format = content.media_type.split("/")[1]
        document = {"image": {"format": image_format, "source": {"bytes": content.base64}}}
    elif content.is_pdf:
        document = {
            "document": {"format": "pdf", "name": "document", "source": {"bytes": content.base64}}
        }
    else:
        raise _unsupported_media_error("bedrock", content)
    tool = {"name": "extract", "description": _TOOL_DESCRIPTION, "inputSchema": {"json": schema}}
    body = {
        "messages": [{"role": "user", "content": [{"text": PROMPT}, document]}],
        "toolConfig": {"tools": [{"toolSpec": tool}], "toolChoice": {"tool": {"name": "extract"}}},
    }
    if instructions:
        body["system"] = [{"text": instructions}]
    return f"/model/{quote(model.model_name, safe='')}/converse", body


def _read_bedrock(payload: dict[str, Any]) -> Any:
    if payload["stopReason"] != "tool_use":
        raise ValueError(_INCOMPLETE_TOOL_CALL)
    parts = payload["output"]["message"]["content"]
    calls = [part["toolUse"] for part in parts if "toolUse" in part]
    return _extract_tool_arguments(calls, "input")


def _chat_document(content: Content, api: str) -> dict[str, Any]:
    """Encode a source as Chat Completions content; Cohere accepts only text and images."""
    if content.is_text:
        return _text_block(content.text)
    if content.is_image:
        return {"type": "image_url", "image_url": {"url": content.data_uri}}
    if api == "chat" and content.is_pdf:
        file = {"filename": "document.pdf", "file_data": content.data_uri}
        return {"type": "file", "file": file}
    if api == "chat" and content.media_type in _CHAT_AUDIO:
        audio = {"data": content.base64, "format": _CHAT_AUDIO[content.media_type]}
        return {"type": "input_audio", "input_audio": audio}
    raise _unsupported_media_error(api, content)


def _text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _user_message(prompt: str, document: dict[str, Any]) -> dict[str, Any]:
    return {"role": "user", "content": [_text_block(prompt), document]}


def _system_messages(instructions: str | None) -> list[dict[str, Any]]:
    return [{"role": "system", "content": instructions}] if instructions else []


def _unsupported_media_error(api: str, content: Content) -> ValueError:
    return ValueError(f"The {api} adapter does not support {content.media_type} inputs.")


def _extract_tool_arguments(calls: list[dict[str, Any]], arguments_key: str) -> Any:
    """Return the arguments of the one forced ``extract`` call."""
    if len(calls) != 1 or calls[0]["name"] != "extract":
        raise ValueError("Expected exactly one extraction tool call.")
    return calls[0][arguments_key]


# API: (request builder, default response reader).
_ADAPTERS = {
    "responses": (_build_responses, _read_responses),
    "chat": (_build_chat, _read_chat),
    "anthropic": (_build_anthropic, _read_anthropic),
    "gemini": (_build_gemini, _read_gemini),
    "ollama": (_build_ollama, _read_ollama),
    "cohere": (_build_cohere, _read_cohere),
    "bedrock": (_build_bedrock, _read_bedrock),
    "mistral-ocr": (_build_mistral_ocr, _read_mistral_ocr),
}
_JSON_READERS = {"chat": _read_chat_json, "anthropic": _read_anthropic_json}
