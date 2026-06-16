"""Base LLM backend interface for agent-weave."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from ..models import LLMResponse, Message, ToolSchema


class LLMBackend(ABC):
    """Abstract LLM backend that agents use for reasoning."""

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send messages to the LLM and return a response."""

    async def agenerate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Async version of generate. Defaults to running sync in a thread."""
        return await asyncio.to_thread(
            self.generate,
            messages,
            tools=tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
