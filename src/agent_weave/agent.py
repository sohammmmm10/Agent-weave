"""Core Agent class for agent-weave."""

from __future__ import annotations

from typing import Any, Sequence

from .config import AgentSettings
from .errors import ConfigurationError
from .guardrails import (
    Guardrail,
    GuardrailPipeline,
    TokenBudgetGuardrail,
)
from .llm.base import LLMBackend
from .memory import ConversationMemory, Memory
from .models import AgentResult, Message
from .react import ReActEngine
from .tool import Tool


class Agent:
    """A self-contained AI agent with tools, memory, and guardrails.

    Usage::

        from agent_weave import Agent, tool
        from agent_weave.llm.openai_backend import OpenAIBackend

        @tool(description="Search the web")
        def web_search(query: str) -> str:
            return f"Results for: {query}"

        agent = Agent(
            name="researcher",
            llm=OpenAIBackend(api_key="sk-..."),
            tools=[web_search],
            system_prompt="You are a research assistant.",
        )

        result = agent.run("Find the latest AI trends in 2025")
        print(result.output)
    """

    def __init__(
        self,
        name: str,
        *,
        llm: LLMBackend,
        tools: Sequence[Tool] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        memory: Memory | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_iterations: int | None = None,
        token_budget: int | None = None,
        output_guardrails: list[Guardrail] | None = None,
        settings: AgentSettings | None = None,
        verbose: bool | None = None,
    ) -> None:
        if not name.strip():
            raise ConfigurationError("Agent name cannot be empty.")

        self.name = name.strip()
        self._llm = llm
        self._tools = list(tools or [])
        self._system_prompt = system_prompt
        self._memory = memory or ConversationMemory()
        self._settings = settings or AgentSettings()

        self._model = model or self._settings.default_model
        self._temperature = temperature
        self._max_iterations = max_iterations or self._settings.max_iterations
        self._verbose = verbose if verbose is not None else self._settings.verbose

        # Guardrails.
        self._token_budget: TokenBudgetGuardrail | None = None
        if token_budget or self._settings.token_budget:
            self._token_budget = TokenBudgetGuardrail(
                budget=token_budget or self._settings.token_budget or 0
            )

        self._output_guardrails: GuardrailPipeline | None = None
        if output_guardrails:
            self._output_guardrails = GuardrailPipeline(output_guardrails)

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools)

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self._tools]

    def add_tool(self, tool_instance: Tool) -> None:
        """Register an additional tool."""
        self._tools.append(tool_instance)

    def run(self, task: str) -> AgentResult:
        """Run the agent on a task (synchronous).

        Sets up memory with system prompt + user task, then
        runs the ReAct loop until completion.
        """
        self._prepare_memory(task)

        engine = self._build_engine()
        return engine.run(self.name)

    async def arun(self, task: str) -> AgentResult:
        """Run the agent on a task (asynchronous)."""
        self._prepare_memory(task)

        engine = self._build_engine()
        return await engine.arun(self.name)

    def chat(self, message: str) -> AgentResult:
        """Continue a conversation (preserves history).

        Unlike ``run()``, this does NOT reset memory. It appends
        the new user message and continues the conversation.
        """
        # Ensure system prompt exists on first call.
        if self._memory.size == 0:
            self._memory.add(
                Message(role="system", content=self._system_prompt)
            )

        self._memory.add(Message(role="user", content=message))

        engine = self._build_engine()
        return engine.run(self.name)

    async def achat(self, message: str) -> AgentResult:
        """Continue a conversation (async, preserves history)."""
        if self._memory.size == 0:
            self._memory.add(
                Message(role="system", content=self._system_prompt)
            )

        self._memory.add(Message(role="user", content=message))

        engine = self._build_engine()
        return await engine.arun(self.name)

    def reset(self) -> None:
        """Clear memory and token budget."""
        self._memory.clear()
        if self._token_budget:
            self._token_budget.reset()

    def _prepare_memory(self, task: str) -> None:
        """Reset memory and load system prompt + user task."""
        self._memory.clear()
        if self._token_budget:
            self._token_budget.reset()

        self._memory.add(Message(role="system", content=self._system_prompt))
        self._memory.add(Message(role="user", content=task))

    def _build_engine(self) -> ReActEngine:
        return ReActEngine(
            llm=self._llm,
            tools=self._tools,
            memory=self._memory,
            max_iterations=self._max_iterations,
            model=self._model,
            temperature=self._temperature,
            token_budget=self._token_budget,
            output_guardrails=self._output_guardrails,
            verbose=self._verbose,
        )

    def __repr__(self) -> str:
        return (
            f"Agent(name={self.name!r}, model={self._model!r}, "
            f"tools={self.tool_names})"
        )
