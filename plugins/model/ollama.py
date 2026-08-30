from __future__ import annotations

import json
from typing import Any, Iterator, Optional

import requests

from core.errors import ModelError
from core.messages import Message
from core.plugin import Plugin
from core.tool_call_extraction import extract_tool_calls_from_text


class OllamaModel(Plugin):
    name = "ollama_model"

    def __init__(
        self,
        model: str = "qwen2.5-coder:1.5b",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict[str, Any], stream: bool = False) -> Any:
        try:
            response = requests.post(
                f"{self.base_url}{path}",
                json=payload,
                timeout=self.timeout,
                stream=stream,
            )
            response.raise_for_status()
            return response
        except Exception as exc:
            raise ModelError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._post("/api/chat", payload)
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise ModelError(f"Invalid JSON from Ollama: {response.text[:1000]}") from exc

    def _stream_chunks(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        payload["stream"] = True
        response = self._post("/api/chat", payload, stream=True)
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> Message:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_dict() for message in messages],
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        try:
            data = self._request(payload)
        except ModelError as exc:
            if tools and "does not support tools" in str(exc):
                payload.pop("tools", None)
                data = self._request(payload)
            else:
                raise

        raw_message = data.get("message")
        if not isinstance(raw_message, dict):
            raise ModelError("Ollama response did not contain a message object.")

        content = raw_message.get("content", "")
        tool_calls = raw_message.get("tool_calls")
        if not tool_calls and content:
            tool_calls = extract_tool_calls_from_text(content, tools)

        return Message(
            role=raw_message.get("role", "assistant"),
            content=content,
            tool_calls=tool_calls or None,
        )

    def stream_chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> Iterator[Message]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_dict() for message in messages],
        }
        if tools:
            payload["tools"] = tools

        current_role: Optional[str] = None
        current_content = ""
        current_tool_calls: list[dict[str, Any]] = []

        for chunk in self._stream_chunks(payload):
            message = chunk.get("message", {})
            role = message.get("role")
            content_delta = message.get("content", "")
            tool_calls_delta = message.get("tool_calls")

            if role:
                current_role = role
            if content_delta:
                current_content += content_delta
            if tool_calls_delta:
                current_tool_calls.extend(tool_calls_delta)

            yield Message(
                role=current_role or "assistant",
                content=current_content,
                tool_calls=current_tool_calls if current_tool_calls else None,
            )

            if chunk.get("done"):
                break

    def list_models(self) -> list[str]:
        response = self._post("/api/tags", {})
        data = response.json()
        models = data.get("models", [])
        return [
            str(item["name"])
            for item in models
            if isinstance(item, dict) and "name" in item
        ]

    def health(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "healthy": True,
            "model": self.model,
            "base_url": self.base_url,
        }

    def embed(self, text: str) -> list[float]:
        payload = {"model": self.model, "prompt": text}
        response = self._post("/api/embeddings", payload)
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise ModelError(f"Invalid JSON from Ollama embeddings: {response.text[:1000]}") from exc
        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            raise ModelError("Ollama embeddings response did not contain an embedding list.")
        return embedding
