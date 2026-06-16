"""Base memory interface for agent-weave."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Message


class Memory(ABC):
    """Abstract base class for agent memory."""

    @abstractmethod
    def add(self, message: Message) -> None:
        """Store a message in memory."""

    @abstractmethod
    def get_messages(self) -> list[Message]:
        """Retrieve all stored messages."""

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored messages."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of messages currently stored."""

    def add_user(self, content: str) -> None:
        """Convenience: add a user message."""
        self.add(Message(role="user", content=content))

    def add_assistant(self, content: str) -> None:
        """Convenience: add an assistant message."""
        self.add(Message(role="assistant", content=content))

    def add_system(self, content: str) -> None:
        """Convenience: add a system message."""
        self.add(Message(role="system", content=content))

    def add_tool_result(
        self, content: str, *, tool_call_id: str, name: str
    ) -> None:
        """Convenience: add a tool result message."""
        self.add(
            Message(
                role="tool",
                content=content,
                tool_call_id=tool_call_id,
                name=name,
            )
        )
