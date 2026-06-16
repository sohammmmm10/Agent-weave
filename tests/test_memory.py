"""Tests for memory modules."""

from __future__ import annotations

import pytest

from agent_weave import Message
from agent_weave.memory import ConversationMemory, SlidingWindowMemory


class TestConversationMemory:
    def test_add_and_retrieve(self) -> None:
        mem = ConversationMemory()
        mem.add(Message(role="user", content="hello"))
        mem.add(Message(role="assistant", content="hi"))

        messages = mem.get_messages()
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].content == "hi"

    def test_clear(self) -> None:
        mem = ConversationMemory()
        mem.add_user("test")
        assert mem.size == 1
        mem.clear()
        assert mem.size == 0

    def test_convenience_methods(self) -> None:
        mem = ConversationMemory()
        mem.add_system("You are helpful.")
        mem.add_user("Hello")
        mem.add_assistant("Hi there!")
        mem.add_tool_result("result", tool_call_id="tc_1", name="search")

        messages = mem.get_messages()
        assert len(messages) == 4
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[2].role == "assistant"
        assert messages[3].role == "tool"
        assert messages[3].tool_call_id == "tc_1"

    def test_repr(self) -> None:
        mem = ConversationMemory()
        assert "0" in repr(mem)


class TestSlidingWindowMemory:
    def test_max_messages_enforced(self) -> None:
        mem = SlidingWindowMemory(max_messages=3, preserve_system=False)
        for i in range(5):
            mem.add(Message(role="user", content=f"msg {i}"))

        messages = mem.get_messages()
        assert len(messages) == 3
        assert messages[0].content == "msg 2"
        assert messages[-1].content == "msg 4"

    def test_preserves_system_prompt(self) -> None:
        mem = SlidingWindowMemory(max_messages=3, preserve_system=True)
        mem.add(Message(role="system", content="Be helpful."))

        for i in range(5):
            mem.add(Message(role="user", content=f"msg {i}"))

        messages = mem.get_messages()
        # System + 3 most recent.
        assert messages[0].role == "system"
        assert messages[0].content == "Be helpful."
        assert len(messages) == 4  # 1 system + 3 window

    def test_system_replace(self) -> None:
        mem = SlidingWindowMemory(max_messages=3, preserve_system=True)
        mem.add(Message(role="system", content="V1"))
        mem.add(Message(role="system", content="V2"))

        messages = mem.get_messages()
        assert len([m for m in messages if m.role == "system"]) == 1
        assert messages[0].content == "V2"

    def test_invalid_max(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            SlidingWindowMemory(max_messages=0)

    def test_clear(self) -> None:
        mem = SlidingWindowMemory(max_messages=5)
        mem.add(Message(role="system", content="sys"))
        mem.add(Message(role="user", content="usr"))
        assert mem.size == 2
        mem.clear()
        assert mem.size == 0

    def test_repr(self) -> None:
        mem = SlidingWindowMemory(max_messages=10)
        assert "max=10" in repr(mem)
