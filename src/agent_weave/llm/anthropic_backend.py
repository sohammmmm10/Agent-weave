"""Anthropic LLM backend for agent-weave."""

from __future__ import annotations

import json
import uuid
from typing import Any

from ..errors import ConfigurationError, LLMBackendError
from ..models import LLMResponse, Message, ToolCall, ToolSchema
from .base import LLMBackend


class AnthropicBackend(LLMBackend):
    """LLM backend using the Anthropic Python SDK.

    Requires ``pip install agent-weave[anthropic]``.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
    ) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise ConfigurationError(
                "Anthropic SDK not installed. Run: pip install agent-weave[anthropic]"
            ) from exc

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key

        self._client = anthropic.Anthropic(**kwargs)
        self._async_client = anthropic.AsyncAnthropic(**kwargs)
        self._default_model = default_model
        self._default_max_tokens = max_tokens

    def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        system_text, api_messages = self._split_system(messages)

        request: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": api_messages,
            "max_tokens": max_tokens or self._default_max_tokens,
        }
        if system_text:
            request["system"] = system_text
        if tools:
            request["tools"] = [t.to_anthropic_tool() for t in tools]
        if temperature is not None:
            request["temperature"] = temperature

        try:
            response = self._client.messages.create(**request)
        except Exception as exc:
            raise LLMBackendError(f"Anthropic API error: {exc}") from exc

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
        system_text, api_messages = self._split_system(messages)

        request: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": api_messages,
            "max_tokens": max_tokens or self._default_max_tokens,
        }
        if system_text:
            request["system"] = system_text
        if tools:
            request["tools"] = [t.to_anthropic_tool() for t in tools]
        if temperature is not None:
            request["temperature"] = temperature

        try:
            response = await self._async_client.messages.create(**request)
        except Exception as exc:
            raise LLMBackendError(f"Anthropic API error: {exc}") from exc

        return self._parse_response(response)

    @staticmethod
    def _split_system(
        messages: list[Message],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Separate system messages from conversation messages.

        Anthropic takes ``system`` as a top-level parameter.
        """
        system_parts: list[str] = []
        api_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            elif msg.role == "tool":
                # Anthropic expects tool results as user messages.
                api_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id or "",
                            "content": msg.content,
                        }
                    ],
                })
            elif msg.role == "assistant" and msg.tool_calls:
                content_blocks: list[dict[str, Any]] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                api_messages.append({"role": "assistant", "content": content_blocks})
            else:
                api_messages.append({
                    "role": msg.role,
                    "content": msg.content or "",
                })

        return "\n\n".join(system_parts), api_messages

    @staticmethod
    def _parse_response(response: Any) -> LLMResponse:
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id or str(uuid.uuid4()),
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        usage = response.usage
        return LLMResponse(
            content="\n".join(content_parts) if content_parts else None,
            tool_calls=tool_calls or None,
            tokens_input=usage.input_tokens if usage else None,
            tokens_output=usage.output_tokens if usage else None,
            model=response.model,
            raw=response,
        )
