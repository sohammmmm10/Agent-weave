"""Tests for multi-agent Team."""

from __future__ import annotations

import pytest

from agent_weave import Agent, Strategy, Team
from agent_weave.errors import ConfigurationError
from agent_weave.llm.base import LLMBackend
from agent_weave.models import LLMResponse, Message, ToolSchema


class MockLLM(LLMBackend):
    """Mock LLM that returns a configurable response."""

    def __init__(self, answer: str = "Mock output.") -> None:
        self._answer = answer

    def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content=self._answer,
            tokens_input=10,
            tokens_output=5,
        )


class TestTeam:
    def test_sequential_strategy(self) -> None:
        agent1 = Agent(name="researcher", llm=MockLLM("Research findings."))
        agent2 = Agent(name="writer", llm=MockLLM("Blog post."))

        team = Team(agents=[agent1, agent2], strategy=Strategy.SEQUENTIAL)
        result = team.run("Write about AI")

        assert result.final_output == "Blog post."
        assert len(result.agent_results) == 2
        assert result.agent_names == ["researcher", "writer"]
        assert result.total_tokens > 0

    def test_round_robin_strategy(self) -> None:
        agent1 = Agent(name="a1", llm=MockLLM("Output A"))
        agent2 = Agent(name="a2", llm=MockLLM("Output B"))

        team = Team(agents=[agent1, agent2], strategy=Strategy.ROUND_ROBIN)
        result = team.run("Task")

        assert result.final_output == "Output B"
        assert len(result.agent_results) == 2

    def test_router_strategy(self) -> None:
        specialist = Agent(name="coder", llm=MockLLM("Here's the code."))
        writer = Agent(name="writer", llm=MockLLM("Here's the article."))
        router = Agent(name="router", llm=MockLLM("coder"))

        team = Team(
            agents=[specialist, writer],
            strategy=Strategy.ROUTER,
            router=router,
        )
        result = team.run("Write a Python script")

        # Router should select 'coder'.
        assert result.final_output == "Here's the code."

    def test_router_fallback(self) -> None:
        agent1 = Agent(name="first", llm=MockLLM("First output."))
        agent2 = Agent(name="second", llm=MockLLM("Second output."))
        router = Agent(name="router", llm=MockLLM("unknown_agent"))

        team = Team(
            agents=[agent1, agent2],
            strategy=Strategy.ROUTER,
            router=router,
        )
        result = team.run("Task")

        # Should fall back to first agent.
        assert result.final_output == "First output."

    def test_empty_agents_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="at least one"):
            Team(agents=[])

    def test_router_without_router_agent_raises(self) -> None:
        agent = Agent(name="test", llm=MockLLM())
        with pytest.raises(ConfigurationError, match="router"):
            Team(agents=[agent], strategy=Strategy.ROUTER)

    def test_strategy_string(self) -> None:
        agent = Agent(name="test", llm=MockLLM())
        team = Team(agents=[agent], strategy=Strategy.SEQUENTIAL)
        result = team.run("Task")
        assert result.strategy == "sequential"
