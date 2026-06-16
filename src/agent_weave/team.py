"""Multi-agent team orchestration for agent-weave."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Sequence

from .agent import Agent
from .errors import AgentWeaveError, ConfigurationError
from .models import AgentResult

logger = logging.getLogger("agent_weave.team")


class Strategy(str, Enum):
    """Execution strategy for the team."""

    SEQUENTIAL = "sequential"
    ROUND_ROBIN = "round_robin"
    ROUTER = "router"


@dataclass
class TeamResult:
    """Result of a multi-agent team run."""

    final_output: str
    agent_results: list[AgentResult] = field(default_factory=list)
    total_tokens: int = 0
    strategy: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def agent_names(self) -> list[str]:
        return [r.agent_name for r in self.agent_results]


class Team:
    """Orchestrate multiple agents to solve a task.

    Strategies:

    - **sequential**: Each agent runs in order. The output of one
      agent becomes the input (context) for the next.
    - **round_robin**: Each agent takes a turn processing the task.
      After all agents go, the last output is the final answer.
    - **router**: A router agent decides which specialist agent
      should handle the task.

    Usage::

        researcher = Agent(name="researcher", llm=backend, ...)
        writer = Agent(name="writer", llm=backend, ...)

        team = Team(
            agents=[researcher, writer],
            strategy=Strategy.SEQUENTIAL,
        )
        result = team.run("Write a blog post about AI agents")
    """

    def __init__(
        self,
        agents: Sequence[Agent],
        *,
        strategy: Strategy = Strategy.SEQUENTIAL,
        router: Agent | None = None,
        verbose: bool = False,
    ) -> None:
        if not agents:
            raise ConfigurationError("Team must have at least one agent.")
        self._agents = list(agents)
        self._strategy = strategy
        self._router = router
        self._verbose = verbose

        if strategy == Strategy.ROUTER and router is None:
            raise ConfigurationError(
                "Router strategy requires a 'router' agent."
            )

    @property
    def agent_names(self) -> list[str]:
        return [a.name for a in self._agents]

    def run(self, task: str) -> TeamResult:
        """Run the team on a task (synchronous)."""
        if self._strategy == Strategy.SEQUENTIAL:
            return self._run_sequential(task)
        elif self._strategy == Strategy.ROUND_ROBIN:
            return self._run_round_robin(task)
        elif self._strategy == Strategy.ROUTER:
            return self._run_router(task)
        else:
            raise ConfigurationError(f"Unknown strategy: {self._strategy}")

    async def arun(self, task: str) -> TeamResult:
        """Run the team on a task (async)."""
        if self._strategy == Strategy.SEQUENTIAL:
            return await self._arun_sequential(task)
        elif self._strategy == Strategy.ROUND_ROBIN:
            return await self._arun_round_robin(task)
        elif self._strategy == Strategy.ROUTER:
            return await self._arun_router(task)
        else:
            raise ConfigurationError(f"Unknown strategy: {self._strategy}")

    # ── Sequential ──────────────────────────────────────────────

    def _run_sequential(self, task: str) -> TeamResult:
        """Each agent runs in order. Output of agent N is input for agent N+1."""
        results: list[AgentResult] = []
        current_input = task

        for agent in self._agents:
            if self._verbose:
                logger.info("Sequential: running agent '%s'", agent.name)
            result = agent.run(current_input)
            results.append(result)
            current_input = (
                f"Previous agent ({agent.name}) produced:\n\n"
                f"{result.output}\n\n"
                f"Original task: {task}\n\n"
                f"Continue and improve upon this."
            )

        return self._build_team_result(results, Strategy.SEQUENTIAL)

    async def _arun_sequential(self, task: str) -> TeamResult:
        results: list[AgentResult] = []
        current_input = task

        for agent in self._agents:
            if self._verbose:
                logger.info("Sequential: running agent '%s'", agent.name)
            result = await agent.arun(current_input)
            results.append(result)
            current_input = (
                f"Previous agent ({agent.name}) produced:\n\n"
                f"{result.output}\n\n"
                f"Original task: {task}\n\n"
                f"Continue and improve upon this."
            )

        return self._build_team_result(results, Strategy.SEQUENTIAL)

    # ── Round Robin ─────────────────────────────────────────────

    def _run_round_robin(self, task: str) -> TeamResult:
        """Each agent processes the task independently."""
        results: list[AgentResult] = []

        for agent in self._agents:
            if self._verbose:
                logger.info("Round-robin: running agent '%s'", agent.name)
            result = agent.run(task)
            results.append(result)

        return self._build_team_result(results, Strategy.ROUND_ROBIN)

    async def _arun_round_robin(self, task: str) -> TeamResult:
        results: list[AgentResult] = []

        for agent in self._agents:
            if self._verbose:
                logger.info("Round-robin: running agent '%s'", agent.name)
            result = await agent.arun(task)
            results.append(result)

        return self._build_team_result(results, Strategy.ROUND_ROBIN)

    # ── Router ──────────────────────────────────────────────────

    def _run_router(self, task: str) -> TeamResult:
        """Router agent decides which specialist handles the task."""
        assert self._router is not None

        routing_prompt = self._build_routing_prompt(task)
        routing_result = self._router.run(routing_prompt)

        selected = self._find_agent(routing_result.output)
        if self._verbose:
            logger.info("Router selected agent: '%s'", selected.name)

        agent_result = selected.run(task)
        return self._build_team_result(
            [routing_result, agent_result], Strategy.ROUTER
        )

    async def _arun_router(self, task: str) -> TeamResult:
        assert self._router is not None

        routing_prompt = self._build_routing_prompt(task)
        routing_result = await self._router.arun(routing_prompt)

        selected = self._find_agent(routing_result.output)
        if self._verbose:
            logger.info("Router selected agent: '%s'", selected.name)

        agent_result = await selected.arun(task)
        return self._build_team_result(
            [routing_result, agent_result], Strategy.ROUTER
        )

    # ── Helpers ─────────────────────────────────────────────────

    def _build_routing_prompt(self, task: str) -> str:
        agents_desc = "\n".join(
            f"- {a.name}" for a in self._agents
        )
        return (
            f"You are a routing agent. Given the task below, decide which "
            f"specialist agent should handle it.\n\n"
            f"Available agents:\n{agents_desc}\n\n"
            f"Task: {task}\n\n"
            f"Reply with ONLY the agent name, nothing else."
        )

    def _find_agent(self, name_output: str) -> Agent:
        """Match router output to an agent by name (fuzzy)."""
        cleaned = name_output.strip().lower()
        for agent in self._agents:
            if agent.name.lower() in cleaned or cleaned in agent.name.lower():
                return agent
        # Fallback to first agent.
        logger.warning(
            "Router output '%s' did not match any agent. Falling back to '%s'.",
            name_output.strip(),
            self._agents[0].name,
        )
        return self._agents[0]

    @staticmethod
    def _build_team_result(
        results: list[AgentResult], strategy: Strategy
    ) -> TeamResult:
        final = results[-1].output if results else ""
        total_tokens = sum(r.total_tokens for r in results)
        return TeamResult(
            final_output=final,
            agent_results=results,
            total_tokens=total_tokens,
            strategy=strategy.value,
        )
