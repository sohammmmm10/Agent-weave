"""Tests for the Agent class."""

from __future__ import annotations

import pytest

from agent_weave import Agent, AgentResult, tool
from agent_weave.errors import ConfigurationError
from agent_weave.llm.base import LLMBackend
from agent_weave.models import LLMResponse, Message, ToolCall, ToolSchema


class MockLLM(LLMBackend):
    """Simple mock LLM for testing."""

    def __init__(self, responses: list[LLMResponse] | None = None) -> None:
        self._responses = list(responses or [
            LLMResponse(content="Mock answer.", tokens_input=10, tokens_output=5),
        ])
        self._call_count = 0
        self.last_messages: list[Message] = []

    def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.last_messages = messages
        if self._call_count >= len(self._responses):
            return LLMResponse(content="Fallback.", tokens_input=5, tokens_output=5)
        response = self._responses[self._call_count]
        self._call_count += 1
        return response


class TestAgent:
    def test_basic_run(self) -> None:
        llm = MockLLM()
        agent = Agent(name="test", llm=llm)
        result = agent.run("Hello")

        assert isinstance(result, AgentResult)
        assert result.output == "Mock answer."
        assert result.agent_name == "test"

    def test_system_prompt_in_messages(self) -> None:
        llm = MockLLM()
        agent = Agent(
            name="test",
            llm=llm,
            system_prompt="You are a pirate.",
        )
        agent.run("Ahoy!")

        # Verify system prompt was sent.
        assert llm.last_messages[0].role == "system"
        assert "pirate" in llm.last_messages[0].content

    def test_run_with_tools(self) -> None:
        @tool(description="Double a number")
        def double(n: int) -> int:
            return n * 2

        llm = MockLLM([
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="tc_1", name="double", arguments={"n": 5})],
                tokens_input=10,
                tokens_output=5,
            ),
            LLMResponse(content="10", tokens_input=10, tokens_output=5),
        ])
        agent = Agent(name="math", llm=llm, tools=[double])
        result = agent.run("Double 5")

        assert result.output == "10"
        assert result.total_tool_calls == 1

    def test_chat_preserves_history(self) -> None:
        llm = MockLLM([
            LLMResponse(content="Hi!", tokens_input=10, tokens_output=5),
            LLMResponse(content="Fine thanks!", tokens_input=20, tokens_output=5),
        ])
        agent = Agent(name="chatbot", llm=llm)

        agent.chat("Hello")
        result = agent.chat("How are you?")

        # Second call should have more messages (history preserved).
        assert len(llm.last_messages) >= 3  # system + user1 + assistant1 + user2

    def test_run_resets_memory(self) -> None:
        llm = MockLLM([
            LLMResponse(content="First", tokens_input=10, tokens_output=5),
            LLMResponse(content="Second", tokens_input=10, tokens_output=5),
        ])
        agent = Agent(name="test", llm=llm)

        agent.run("Task 1")
        agent.run("Task 2")

        # After second run, memory should only have system + user (fresh).
        assert len(llm.last_messages) == 2

    def test_empty_name_raises(self) -> None:
        llm = MockLLM()
        with pytest.raises(ConfigurationError, match="empty"):
            Agent(name="", llm=llm)

    def test_add_tool(self) -> None:
        llm = MockLLM()
        agent = Agent(name="test", llm=llm)
        assert len(agent.tools) == 0

        @tool(description="Test")
        def my_tool() -> str:
            return "ok"

        agent.add_tool(my_tool)
        assert len(agent.tools) == 1
        assert "my_tool" in agent.tool_names

    def test_reset_clears_memory(self) -> None:
        llm = MockLLM()
        agent = Agent(name="test", llm=llm)
        agent.chat("Hello")
        agent.reset()

        # Memory should be empty after reset.
        assert agent._memory.size == 0

    def test_repr(self) -> None:
        llm = MockLLM()
        agent = Agent(name="demo", llm=llm, model="gpt-4o")
        assert "demo" in repr(agent)
        assert "gpt-4o" in repr(agent)

    def test_token_budget_integration(self) -> None:
        llm = MockLLM([
            LLMResponse(content="Answer.", tokens_input=50, tokens_output=20),
        ])
        agent = Agent(name="test", llm=llm, token_budget=10_000)
        result = agent.run("Test task")

        assert result.total_tokens == 70
