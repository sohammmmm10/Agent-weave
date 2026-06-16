"""OpenAI LLM backend for agent-weave."""

from __future__ import annotations

import json
from typing import Any

from ..errors import ConfigurationError, LLMBackendError
from ..models import LLMResponse, Message, ToolCall, ToolSchema
from .base import LLMBackend


class OpenAIBackend(LLMBackend):
    """LLM backend using the OpenAI Python SDK.

    Requires ``pip install agent-weave[openai]``.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str = "gpt-4o-mini",
        organization: str | None = None,
    ) -> None:
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise ConfigurationError(
                "OpenAI SDK not installed. Run: pip install agent-weave[openai]"
            ) from exc

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        if organization:
            kwargs["organization"] = organization

        self._client = openai.OpenAI(**kwargs)
        self._async_client = openai.AsyncOpenAI(**kwargs)
        self._default_model = default_model

    def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        request: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": self._format_messages(messages),
        }
        if tools:
            request["tools"] = [t.to_openai_tool() for t in tools]
        if temperature is not None:
            request["temperature"] = temperature
        if max_tokens is not None:
            request["max_tokens"] = max_tokens

        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            raise LLMBackendError(f"OpenAI API error: {exc}") from exc

        return self._parse_response(response)

    async def agenerate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        request: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": self._format_messages(messages),
        }
        if tools:
            request["tools"] = [t.to_openai_tool() for t in tools]
        if temperature is not None:
            request["temperature"] = temperature
        if max_tokens is not None:
            request["max_tokens"] = max_tokens

        try:
            response = await self._async_client.chat.completions.create(**request)
        except Exception as exc:
            raise LLMBackendError(f"OpenAI API error: {exc}") from exc

        return self._parse_response(response)

    @staticmethod
    def _format_messages(messages: list[Message]) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        for msg in messages:
            entry: dict[str, Any] = {"role": msg.role, "content": msg.content or ""}
            if msg.name:
                entry["name"] = msg.name
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            formatted.append(entry)
        return formatted

    @staticmethod
    def _parse_response(response: Any) -> LLMResponse:
        choice = response.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] | None = None
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        usage = response.usage
        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            tokens_input=usage.prompt_tokens if usage else None,
            tokens_output=usage.completion_tokens if usage else None,
            model=response.model,
            raw=response,
        )
