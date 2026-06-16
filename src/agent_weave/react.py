"""ReAct (Reasoning + Acting) loop engine for agent-weave."""

from __future__ import annotations

import logging
from typing import Any

from .errors import (
    MaxIterationsError,
    ToolExecutionError,
    ToolNotFoundError,
)
from .guardrails import GuardrailPipeline, TokenBudgetGuardrail
from .llm.base import LLMBackend
from .memory.base import Memory
from .models import (
    AgentResult,
    AgentStep,
    LLMResponse,
    Message,
    ToolCall,
    ToolResult,
    ToolSchema,
)
from .tool import Tool

logger = logging.getLogger("agent_weave.react")


class ReActEngine:
    """Execute the ReAct loop: Think -> Act -> Observe -> Repeat.

    The engine sends messages to the LLM, checks if the LLM requested
    any tool calls, executes them, feeds results back, and continues
    until the LLM produces a final text answer or the iteration limit
    is reached.
    """

    def __init__(
        self,
        *,
        llm: LLMBackend,
        tools: list[Tool] | None = None,
        memory: Memory,
        max_iterations: int = 10,
        model: str | None = None,
        temperature: float | None = None,
        token_budget: TokenBudgetGuardrail | None = None,
        output_guardrails: GuardrailPipeline | None = None,
        verbose: bool = False,
    ) -> None:
        self._llm = llm
        self._tools: dict[str, Tool] = {t.name: t for t in (tools or [])}
        self._memory = memory
        self._max_iterations = max_iterations
        self._model = model
        self._temperature = temperature
        self._token_budget = token_budget
        self._output_guardrails = output_guardrails
        self._verbose = verbose

    @property
    def tool_schemas(self) -> list[ToolSchema]:
        return [t.schema for t in self._tools.values()]

    def run(self, agent_name: str) -> AgentResult:
        """Execute the synchronous ReAct loop."""
        steps: list[AgentStep] = []
        total_tokens = 0

        for iteration in range(1, self._max_iterations + 1):
            if self._verbose:
                logger.info("Iteration %d/%d", iteration, self._max_iterations)

            # --- THINK: ask the LLM ---
            messages = self._memory.get_messages()
            schemas = self.tool_schemas if self._tools else None

            response: LLMResponse = self._llm.generate(
                messages,
                tools=schemas,
                model=self._model,
                temperature=self._temperature,
            )

            step_tokens = response.total_tokens
            total_tokens += step_tokens

            if self._token_budget:
                self._token_budget.consume(step_tokens)

            # --- No tool calls → final answer ---
            if not response.has_tool_calls:
                content = response.content or ""

                # Apply output guardrails.
                if self._output_guardrails:
                    content = self._output_guardrails.run(content)

                self._memory.add(Message(role="assistant", content=content))

                step = AgentStep(
                    step_number=iteration,
                    thought=content,
                    tokens_used=step_tokens,
                )
                steps.append(step)

                return AgentResult(
                    output=content,
                    agent_name=agent_name,
                    steps=steps,
                    total_tokens=total_tokens,
                    total_iterations=iteration,
                    model=response.model,
                )

            # --- ACT: execute tool calls ---
            assert response.tool_calls is not None

            # Store the assistant message with tool calls.
            self._memory.add(
                Message(
                    role="assistant",
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                )
            )

            tool_results = self._execute_tools(response.tool_calls)

            # --- OBSERVE: feed results back ---
            for result in tool_results:
                self._memory.add_tool_result(
                    result.output,
                    tool_call_id=result.tool_call_id,
                    name=result.tool_name,
                )

            step = AgentStep(
                step_number=iteration,
                thought=response.content,
                tool_calls=response.tool_calls,
                tool_results=tool_results,
                tokens_used=step_tokens,
            )
            steps.append(step)

            if self._verbose:
                for r in tool_results:
                    status = "ERROR" if r.is_error else "OK"
                    logger.info(
                        "  Tool %s [%s]: %s",
                        r.tool_name,
                        status,
                        r.output[:200],
                    )

        # Exhausted iterations.
        raise MaxIterationsError(self._max_iterations)

    async def arun(self, agent_name: str) -> AgentResult:
        """Execute the async ReAct loop."""
        steps: list[AgentStep] = []
        total_tokens = 0

        for iteration in range(1, self._max_iterations + 1):
            if self._verbose:
                logger.info("Iteration %d/%d", iteration, self._max_iterations)

            messages = self._memory.get_messages()
            schemas = self.tool_schemas if self._tools else None

            response: LLMResponse = await self._llm.agenerate(
                messages,
                tools=schemas,
                model=self._model,
                temperature=self._temperature,
            )

            step_tokens = response.total_tokens
            total_tokens += step_tokens

            if self._token_budget:
                self._token_budget.consume(step_tokens)

            if not response.has_tool_calls:
                content = response.content or ""
                if self._output_guardrails:
                    content = self._output_guardrails.run(content)

                self._memory.add(Message(role="assistant", content=content))

                step = AgentStep(
                    step_number=iteration,
                    thought=content,
                    tokens_used=step_tokens,
                )
                steps.append(step)

                return AgentResult(
                    output=content,
                    agent_name=agent_name,
                    steps=steps,
                    total_tokens=total_tokens,
                    total_iterations=iteration,
                    model=response.model,
                )

            assert response.tool_calls is not None

            self._memory.add(
                Message(
                    role="assistant",
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                )
            )

            tool_results = await self._aexecute_tools(response.tool_calls)

            for result in tool_results:
                self._memory.add_tool_result(
                    result.output,
                    tool_call_id=result.tool_call_id,
                    name=result.tool_name,
                )

            step = AgentStep(
                step_number=iteration,
                thought=response.content,
                tool_calls=response.tool_calls,
                tool_results=tool_results,
                tokens_used=step_tokens,
            )
            steps.append(step)

        raise MaxIterationsError(self._max_iterations)

    def _execute_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        results: list[ToolResult] = []
        for tc in tool_calls:
            tool_obj = self._tools.get(tc.name)
            if tool_obj is None:
                available = list(self._tools.keys())
                results.append(
                    ToolResult(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        output=f"Error: Tool '{tc.name}' not found. Available: {available}",
                        is_error=True,
                    )
                )
                continue

            try:
                output = tool_obj.execute(tc.arguments)
                results.append(
                    ToolResult(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        output=output,
                    )
                )
            except ToolExecutionError as exc:
                results.append(
                    ToolResult(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        output=f"Error: {exc}",
                        is_error=True,
                    )
                )
        return results

    async def _aexecute_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        results: list[ToolResult] = []
        for tc in tool_calls:
            tool_obj = self._tools.get(tc.name)
            if tool_obj is None:
                results.append(
                    ToolResult(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        output=f"Error: Tool '{tc.name}' not found.",
                        is_error=True,
                    )
                )
                continue

            try:
                output = await tool_obj.aexecute(tc.arguments)
                results.append(
                    ToolResult(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        output=output,
                    )
                )
            except ToolExecutionError as exc:
                results.append(
                    ToolResult(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        output=f"Error: {exc}",
                        is_error=True,
                    )
                )
        return results
