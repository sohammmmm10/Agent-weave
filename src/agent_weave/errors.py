"""Custom exceptions for agent-weave."""

from __future__ import annotations


class AgentWeaveError(Exception):
    """Base exception for agent-weave."""


class ToolExecutionError(AgentWeaveError):
    """Raised when a tool fails during execution."""

    def __init__(self, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' failed: {message}")


class ToolNotFoundError(AgentWeaveError):
    """Raised when a requested tool is not registered."""

    def __init__(self, tool_name: str, available: list[str] | None = None) -> None:
        self.tool_name = tool_name
        avail = ", ".join(available) if available else "none"
        super().__init__(
            f"Tool '{tool_name}' not found. Available tools: {avail}"
        )


class MaxIterationsError(AgentWeaveError):
    """Raised when the ReAct loop exceeds the maximum iterations."""

    def __init__(self, max_iterations: int) -> None:
        self.max_iterations = max_iterations
        super().__init__(
            f"Agent exceeded maximum iterations ({max_iterations}). "
            "Increase max_iterations or simplify the task."
        )


class TokenBudgetExceededError(AgentWeaveError):
    """Raised when the token budget is exhausted."""

    def __init__(self, budget: int, used: int) -> None:
        self.budget = budget
        self.used = used
        super().__init__(
            f"Token budget exceeded: used {used:,} of {budget:,} allowed tokens."
        )


class LLMBackendError(AgentWeaveError):
    """Raised when the LLM backend fails."""


class GuardrailViolation(AgentWeaveError):
    """Raised when a guardrail check fails."""

    def __init__(self, guardrail_name: str, message: str) -> None:
        self.guardrail_name = guardrail_name
        super().__init__(f"Guardrail '{guardrail_name}': {message}")


class ConfigurationError(AgentWeaveError):
    """Raised for invalid configuration."""
