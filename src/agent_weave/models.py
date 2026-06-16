"""Data models for agent-weave."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class Message:
    """A single message in the conversation."""

    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return payload


@dataclass(frozen=True)
class ToolCall:
    """Represents a tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


@dataclass(frozen=True)
class ToolResult:
    """Result from executing a tool."""

    tool_call_id: str
    tool_name: str
    output: str
    is_error: bool = False


@dataclass(frozen=True)
class LLMResponse:
    """Response from the LLM backend."""

    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tokens_input: int | None = None
    tokens_output: int | None = None
    model: str | None = None
    raw: Any = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def total_tokens(self) -> int:
        return (self.tokens_input or 0) + (self.tokens_output or 0)


@dataclass(frozen=True)
class ToolSchema:
    """Schema describing a tool for the LLM."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {},
        "required": [],
    })

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_tool(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


@dataclass(frozen=True)
class AgentStep:
    """A single step in the agent's reasoning loop."""

    step_number: int
    thought: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_results: list[ToolResult] | None = None
    tokens_used: int = 0


@dataclass
class AgentResult:
    """Final result returned by an agent run."""

    output: str
    agent_name: str
    steps: list[AgentStep] = field(default_factory=list)
    total_tokens: int = 0
    total_iterations: int = 0
    model: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total_tool_calls(self) -> int:
        return sum(
            len(step.tool_calls or [])
            for step in self.steps
        )
