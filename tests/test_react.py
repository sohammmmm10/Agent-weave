"""Tests for the ReAct engine."""

from __future__ import annotations

import pytest

from agent_weave.errors import MaxIterationsError
from agent_weave.guardrails import TokenBudgetGuardrail
from agent_weave.llm.base import LLMBackend
from agent_weave.memory import ConversationMemory
from agent_weave.models import LLMResponse, Message, ToolCall, ToolSchema
from agent_weave.react import ReActEngine
from agent_weave.tool import tool


# --- Mock LLM backend ---

class MockLLM(LLMBackend):
    """LLM that returns pre-configured responses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if self._call_count >= len(self._responses):
            return LLMResponse(content="Fallback answer.", tokens_input=10, tokens_output=5)
        response = self._responses[self._call_count]
        self._call_count += 1
        return response


# --- Tools for testing ---

@tool(description="Add two numbers")
def add(a: int, b: int) -> int:
    return a + b


@tool(description="Echo text back")
def echo(text: str) -> str:
    return text


class TestReActEngine:
    def test_direct_answer_no_tools(self) -> None:
        """LLM returns a final answer without tool calls."""
        llm = MockLLM([
            LLMResponse(content="The answer is 42.", tokens_input=10, tokens_output=5),
        ])
        memory = ConversationMemory()
        memory.add(Message(role="system", content="Be helpful."))
        memory.add(Message(role="user", content="What is 42?"))

        engine = ReActEngine(llm=llm, memory=memory, max_iterations=5)
        result = engine.run("test-agent")

        assert result.output == "The answer is 42."
        assert result.total_iterations == 1
        assert result.total_tokens == 15

    def test_tool_call_then_answer(self) -> None:
        """LLM calls a tool, then returns a final answer."""
        llm = MockLLM([
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="tc_1", name="add", arguments={"a": 5, "b": 3})],
                tokens_input=20,
                tokens_output=10,
            ),
            LLMResponse(
                content="5 + 3 = 8",
                tokens_input=30,
                tokens_output=5,
            ),
        ])
        memory = ConversationMemory()
        memory.add(Message(role="system", content="Use tools."))
        memory.add(Message(role="user", content="What is 5 + 3?"))

        engine = ReActEngine(llm=llm, tools=[add], memory=memory, max_iterations=5)
        result = engine.run("test-agent")

        assert result.output == "5 + 3 = 8"
        assert result.total_iterations == 2
        assert result.total_tool_calls == 1

    def test_multiple_tool_calls(self) -> None:
        """LLM calls multiple tools in one step."""
        llm = MockLLM([
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="tc_1", name="add", arguments={"a": 1, "b": 2}),
                    ToolCall(id="tc_2", name="echo", arguments={"text": "hello"}),
                ],
                tokens_input=20,
                tokens_output=10,
            ),
            LLMResponse(content="Done! 3 and hello", tokens_input=30, tokens_output=5),
        ])
        memory = ConversationMemory()
        memory.add(Message(role="user", content="test"))

        engine = ReActEngine(
            llm=llm, tools=[add, echo], memory=memory, max_iterations=5
        )
        result = engine.run("test-agent")

        assert result.total_tool_calls == 2
        assert result.output == "Done! 3 and hello"

    def test_unknown_tool_handled_gracefully(self) -> None:
        """LLM calls a tool that doesn't exist — engine returns error to LLM."""
        llm = MockLLM([
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="tc_1", name="nonexistent", arguments={})],
                tokens_input=10,
                tokens_output=5,
            ),
            LLMResponse(content="Sorry, tool not found.", tokens_input=10, tokens_output=5),
        ])
        memory = ConversationMemory()
        memory.add(Message(role="user", content="test"))

        engine = ReActEngine(llm=llm, tools=[add], memory=memory, max_iterations=5)
        result = engine.run("test-agent")

        # Check that the error was fed back and the agent recovered.
        assert result.output == "Sorry, tool not found."
        assert result.steps[0].tool_results is not None
        assert result.steps[0].tool_results[0].is_error

    def test_max_iterations_exceeded(self) -> None:
        """Engine raises MaxIterationsError if loop doesn't converge."""
        # LLM always asks for tool calls, never gives a final answer.
        infinite_tool_calls = [
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id=f"tc_{i}", name="echo", arguments={"text": "loop"})],
                tokens_input=10,
                tokens_output=5,
            )
            for i in range(5)
        ]
        llm = MockLLM(infinite_tool_calls)
        memory = ConversationMemory()
        memory.add(Message(role="user", content="test"))

        engine = ReActEngine(llm=llm, tools=[echo], memory=memory, max_iterations=3)

        with pytest.raises(MaxIterationsError, match="3"):
            engine.run("test-agent")

    def test_token_budget_tracking(self) -> None:
        """Token budget guardrail tracks usage."""
        llm = MockLLM([
            LLMResponse(content="Answer.", tokens_input=50, tokens_output=20),
        ])
        memory = ConversationMemory()
        memory.add(Message(role="user", content="test"))

        budget = TokenBudgetGuardrail(budget=10_000)
        engine = ReActEngine(
            llm=llm, memory=memory, max_iterations=5, token_budget=budget
        )
        result = engine.run("test-agent")

        assert budget.used == 70
        assert budget.remaining == 9_930

    def test_memory_populated_after_run(self) -> None:
        """Memory should contain the full conversation after a run."""
        llm = MockLLM([
            LLMResponse(content="Hello!", tokens_input=10, tokens_output=5),
        ])
        memory = ConversationMemory()
        memory.add(Message(role="system", content="Be nice."))
        memory.add(Message(role="user", content="Hi"))

        engine = ReActEngine(llm=llm, memory=memory, max_iterations=5)
        engine.run("test-agent")

        messages = memory.get_messages()
        assert len(messages) == 3  # system + user + assistant
        assert messages[-1].role == "assistant"
        assert messages[-1].content == "Hello!"
