"""Pure HTTP request/response adapters for supported model APIs."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import quote

from ._content import Content
from .models import ResolvedModel

_DOCUMENT_TYPES = {
    "application/pdf": ".pdf",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/rtf": ".rtf",
    "application/vnd.oasis.opendocument.text": ".odt",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
}
_TOOL_DESCRIPTION = "Return the extracted information."
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
    "format",
}


def _strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Adapt fixed-object schemas for OpenAI strict output without mutating the original."""
    schema = copy.deepcopy(schema)

    def visit(node: dict[str, Any]) -> None:
        node.pop("default", None)
        node.pop("discriminator", None)
        if "oneOf" in node:
            # Tagged union branches remain disjoint through their discriminator literals.
            node["anyOf"] = node.pop("oneOf")
        if node.get("type") == "object":
            if node.get("additionalProperties") not in (None, False):
                raise ValueError(
                    "Strict JSON Schema output does not support open-ended dictionaries; "
                    "use Model(api='chat') for tool-based output."
                )
            node["additionalProperties"] = False
            node["required"] = list(node.get("properties", {}))
        for keyword in ("properties", "$defs", "definitions"):
            for child in node.get(keyword, {}).values():
                visit(child)
        for keyword in ("items", "additionalProperties"):
            if isinstance(node.get(keyword), dict):
                visit(node[keyword])
        for keyword in ("anyOf", "allOf", "oneOf", "prefixItems"):
            for child in node.get(keyword, []):
                visit(child)

    visit(schema)
    return schema


def _document(
    content: Content, api: Literal["responses", "chat", "cohere", "anthropic"]
) -> dict[str, Any]:
    if content.is_text:
        key = "input_text" if api == "responses" else "text"
        return {"type": key, "text": content.text}
    if api == "responses":
        if content.media_type.startswith("image/"):
            return {"type": "input_image", "image_url": content.data_uri}
        if extension := _DOCUMENT_TYPES.get(content.media_type):
            return {
                "type": "input_file",
                "file_data": content.data_uri,
                "filename": "document" + extension,
            }
    elif api in ("chat", "cohere"):
        if content.media_type.startswith("image/"):
            return {"type": "image_url", "image_url": {"url": content.data_uri}}
        if api == "chat":
            if content.media_type == "application/pdf":
                return {
                    "type": "file",
                    "file": {"filename": "document.pdf", "file_data": content.data_uri},
                }
            if content.media_type in ("audio/mpeg", "audio/wav"):
                return {
                    "type": "input_audio",
                    "input_audio": {
                        "data": content.base64,
                        "format": "mp3" if content.media_type == "audio/mpeg" else "wav",
                    },
                }
    else:
        if content.media_type in (
            "image/png",
            "image/jpeg",
            "image/gif",
            "image/webp",
            "application/pdf",
        ):
            return {
                "type": "document" if content.media_type == "application/pdf" else "image",
                "source": {
                    "type": "base64",
                    "media_type": content.media_type,
                    "data": content.base64,
                },
            }
    raise ValueError(f"The {api} adapter does not support {content.media_type} inputs.")


def _responses(
    model: ResolvedModel,
    schema: dict[str, Any],
    content: Content,
    prompt: str,
    instructions: str | None,
) -> tuple[str, dict[str, Any]]:
    body: dict[str, Any] = {
        "model": model.model_name,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    _document(content, "responses"),
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "extract",
                "schema": _strict_schema(schema),
                "strict": True,
            }
        },
    }
    if instructions:
        body["instructions"] = instructions
    return "/responses", body


def _messages(
    model: ResolvedModel,
    schema: dict[str, Any],
    content: Content,
    prompt: str,
    instructions: str | None,
) -> tuple[str, dict[str, Any]]:
    api = model.api
    assert api in ("chat", "anthropic", "cohere")
    path = "/messages" if api == "anthropic" else "/chat/completions" if api == "chat" else "/chat"
    body = {
        "model": model.model_name,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, _document(content, api)],
            }
        ],
    }
    if instructions:
        if api == "anthropic":
            body["system"] = instructions
        else:
            body["messages"].insert(0, {"role": "system", "content": instructions})
    if api == "chat" and model.config.output_mode == "json_schema":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "extract", "strict": True, "schema": _strict_schema(schema)},
        }
    elif api == "chat":
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "extract",
                    "description": _TOOL_DESCRIPTION,
                    "parameters": schema,
                },
            }
        ]
        body["tool_choice"] = {"type": "function", "function": {"name": "extract"}}
    elif api == "anthropic":
        body["max_tokens"] = 4096
        body["tools"] = [
            {
                "name": "extract",
                "description": _TOOL_DESCRIPTION,
                "input_schema": schema,
            }
        ]
        body["tool_choice"] = {"type": "tool", "name": "extract"}
    else:
        body["messages"][-1] = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt + " Return the extracted information as JSON."},
                _document(content, api),
            ],
        }
        body["response_format"] = {"type": "json_object", "schema": schema}
    return path, body


def _gemini(
    model: ResolvedModel,
    schema: dict[str, Any],
    content: Content,
    prompt: str,
    instructions: str | None,
) -> tuple[str, dict[str, Any]]:
    path = f"/models/{quote(model.model_name, safe='')}:generateContent"
    if content.is_text:
        document = {"text": content.text}
    elif (
        content.media_type.startswith(("image/", "audio/", "video/"))
        or content.media_type == "application/pdf"
    ):
        document = {"inlineData": {"mimeType": content.media_type, "data": content.base64}}
    else:
        raise ValueError(f"The gemini adapter does not support {content.media_type} inputs.")
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}, document]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": schema,
        },
    }
    if instructions:
        body["systemInstruction"] = {"parts": [{"text": instructions}]}
    return path, body


def _ollama(
    model: ResolvedModel,
    schema: dict[str, Any],
    content: Content,
    prompt: str,
    instructions: str | None,
) -> tuple[str, dict[str, Any]]:
    message: dict[str, Any] = {"role": "user", "content": prompt}
    if content.is_text:
        message["content"] += "\n\n" + content.text
    elif content.media_type.startswith("image/"):
        message["images"] = [content.base64]
    else:
        raise ValueError(f"The ollama adapter does not support {content.media_type} inputs.")
    messages = [{"role": "system", "content": instructions}] if instructions else []
    body = {
        "model": model.model_name,
        "messages": [*messages, message],
        "format": schema,
        "stream": False,
    }
    return "/api/chat", body


def _bedrock(
    model: ResolvedModel,
    schema: dict[str, Any],
    content: Content,
    prompt: str,
    instructions: str | None,
) -> tuple[str, dict[str, Any]]:
    path = f"/model/{quote(model.model_name, safe='')}/converse"
    if content.is_text:
        document = {"text": content.text}
    elif content.media_type in ("image/png", "image/jpeg", "image/gif", "image/webp"):
        document = {
            "image": {
                "format": content.media_type.split("/")[1],
                "source": {"bytes": content.base64},
            }
        }
    elif content.media_type == "application/pdf":
        document = {
            "document": {
                "format": "pdf",
                "name": "document",
                "source": {"bytes": content.base64},
            }
        }
    else:
        raise ValueError(f"The bedrock adapter does not support {content.media_type} inputs.")
    body = {
        "messages": [{"role": "user", "content": [{"text": prompt}, document]}],
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "name": "extract",
                        "description": _TOOL_DESCRIPTION,
                        "inputSchema": {"json": schema},
                    }
                }
            ],
            "toolChoice": {"tool": {"name": "extract"}},
        },
    }
    if instructions:
        body["system"] = [{"text": instructions}]
    return path, body


def _parse_responses(payload: dict[str, Any]) -> Any:
    if payload.get("status") != "completed":
        raise ValueError("The model response was not completed.")
    parts = [
        part for item in payload["output"] if item["type"] == "message" for part in item["content"]
    ]
    if any(part["type"] == "refusal" for part in parts):
        raise ValueError("The model refused to extract the document.")
    return "".join(part["text"] for part in parts if part["type"] == "output_text")


def _parse_chat(payload: dict[str, Any]) -> Any:
    choice = payload["choices"][0]
    # Forced function calls can report a natural stop even with complete tool arguments.
    if choice["finish_reason"] not in ("tool_calls", "stop") or choice["message"].get("refusal"):
        raise ValueError("The model did not complete the extraction tool call.")
    calls = choice["message"].get("tool_calls", [])
    if len(calls) != 1 or calls[0]["function"]["name"] != "extract":
        raise ValueError("Expected exactly one extraction tool call.")
    return calls[0]["function"]["arguments"]


def _parse_chat_json(payload: dict[str, Any]) -> Any:
    choice = payload["choices"][0]
    message = choice["message"]
    if choice["finish_reason"] != "stop" or message.get("refusal") or message.get("tool_calls"):
        raise ValueError("The model did not complete a JSON Schema response.")
    if not isinstance(message["content"], str):
        raise ValueError("Expected JSON text in the model response.")
    return message["content"]


def _parse_anthropic(payload: dict[str, Any]) -> Any:
    if payload["stop_reason"] != "tool_use":
        raise ValueError("The model did not complete the extraction tool call.")
    calls = [part for part in payload["content"] if part["type"] == "tool_use"]
    if len(calls) != 1 or calls[0]["name"] != "extract":
        raise ValueError("Expected exactly one extraction tool call.")
    return calls[0]["input"]


def _parse_gemini(payload: dict[str, Any]) -> Any:
    candidate = payload["candidates"][0]
    if candidate["finishReason"] != "STOP":
        raise ValueError("The model response was blocked or incomplete.")
    return "".join(
        part["text"]
        for part in candidate["content"]["parts"]
        if "text" in part and not part.get("thought")
    )


def _parse_ollama(payload: dict[str, Any]) -> Any:
    if payload.get("done") is not True or payload.get("done_reason", "stop") != "stop":
        raise ValueError("The model response was not completed.")
    return payload["message"]["content"]


def _parse_cohere(payload: dict[str, Any]) -> Any:
    if payload["finish_reason"].lower() != "complete":
        raise ValueError("The model response was not completed.")
    return "".join(part["text"] for part in payload["message"]["content"] if part["type"] == "text")


def _parse_bedrock(payload: dict[str, Any]) -> Any:
    if payload["stopReason"] != "tool_use":
        raise ValueError("The model did not complete the extraction tool call.")
    calls = [
        part["toolUse"] for part in payload["output"]["message"]["content"] if "toolUse" in part
    ]
    if len(calls) != 1 or calls[0]["name"] != "extract":
        raise ValueError("Expected exactly one extraction tool call.")
    return calls[0]["input"]


type _RequestBuilder = Callable[
    [ResolvedModel, dict[str, Any], Content, str, str | None], tuple[str, dict[str, Any]]
]


@dataclass(frozen=True)
class _Adapter:
    build: _RequestBuilder
    parse: Callable[[dict[str, Any]], Any]


_ADAPTERS = {
    "responses": _Adapter(_responses, _parse_responses),
    "chat": _Adapter(_messages, _parse_chat),
    "anthropic": _Adapter(_messages, _parse_anthropic),
    "gemini": _Adapter(_gemini, _parse_gemini),
    "ollama": _Adapter(_ollama, _parse_ollama),
    "cohere": _Adapter(_messages, _parse_cohere),
    "bedrock": _Adapter(_bedrock, _parse_bedrock),
}


def build(
    model: ResolvedModel,
    schema: dict[str, Any],
    content: Content,
    prompt: str,
    instructions: str | None,
) -> tuple[str, dict[str, Any]]:
    api = model.api
    path, body = _ADAPTERS[api].build(model, schema, content, prompt, instructions)
    settings = dict(model.config.settings)
    # These are defaults, not fields the caller is prevented from configuring.
    if api == "anthropic":
        body["max_tokens"] = settings.pop("max_tokens", body["max_tokens"])
    if api == "gemini":
        reserved = {
            "responseMimeType",
            "responseJsonSchema",
            "responseSchema",
            "responseFormat",
        } & settings.keys()
        if reserved:
            raise ValueError(f"settings cannot override structured output: {sorted(reserved)}.")
        body["generationConfig"].update(settings)
    else:
        reserved = (body.keys() | _STRUCTURE_FIELDS) & settings.keys()
        if reserved:
            raise ValueError(f"settings cannot override request fields: {sorted(reserved)}.")
        body.update(settings)
    if body.get("stream"):
        raise ValueError("Streaming is not supported by extract; omit settings['stream'].")
    return path, body


def parse(model: ResolvedModel, payload: dict[str, Any]) -> str | dict[str, Any]:
    """Return structured output, rejecting refusals and incomplete generations."""
    if model.api == "chat" and model.config.output_mode == "json_schema":
        result = _parse_chat_json(payload)
    else:
        result = _ADAPTERS[model.api].parse(payload)
    if not isinstance(result, (str, dict)) or result == "":
        raise ValueError("The model returned no structured output.")
    return result
