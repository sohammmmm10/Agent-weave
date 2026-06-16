"""Conversation memory implementations for agent-weave."""

from __future__ import annotations

from collections import deque

from ..models import Message
from .base import Memory


class ConversationMemory(Memory):
    """Unbounded conversation memory that stores all messages.

    Suitable for short conversations or when you want complete history.
    """

    def __init__(self) -> None:
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)

    def get_messages(self) -> list[Message]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    @property
    def size(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"ConversationMemory(messages={self.size})"


class SlidingWindowMemory(Memory):
    """Fixed-size sliding window memory.

    Keeps the most recent ``max_messages`` messages.  Optionally
    preserves the system prompt at the start so the agent never loses
    its instructions.
    """

    def __init__(
        self,
        max_messages: int = 20,
        *,
        preserve_system: bool = True,
    ) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1.")
        self._max = max_messages
        self._preserve_system = preserve_system
        self._system_messages: list[Message] = []
        self._messages: deque[Message] = deque(maxlen=max_messages)

    def add(self, message: Message) -> None:
        if self._preserve_system and message.role == "system":
            # Replace existing system messages.
            self._system_messages = [message]
        else:
            self._messages.append(message)

    def get_messages(self) -> list[Message]:
        result: list[Message] = []
        if self._preserve_system:
            result.extend(self._system_messages)
        result.extend(self._messages)
        return result

    def clear(self) -> None:
        self._system_messages.clear()
        self._messages.clear()

    @property
    def size(self) -> int:
        return len(self._system_messages) + len(self._messages)

    def __repr__(self) -> str:
        return (
            f"SlidingWindowMemory(messages={self.size}, "
            f"max={self._max})"
        )
