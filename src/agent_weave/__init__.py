"""agent-weave: Lightweight AI agent framework."""

from .agent import Agent
from .config import AgentSettings
from .errors import (
    AgentWeaveError,
    ConfigurationError,
    GuardrailViolation,
    LLMBackendError,
    MaxIterationsError,
    TokenBudgetExceededError,
    ToolExecutionError,
    ToolNotFoundError,
)
from .guardrails import (
    BlockedWordsGuardrail,
    Guardrail,
    GuardrailPipeline,
    MaxLengthGuardrail,
    PIIGuardrail,
    RegexGuardrail,
    TokenBudgetGuardrail,
)
from .models import AgentResult, AgentStep, Message, ToolCall, ToolResult
from .team import Strategy, Team, TeamResult
from .tool import Tool, tool

__version__ = "0.1.0"

__all__ = [
    # Core
    "Agent",
    "Tool",
    "tool",
    "Team",
    "Strategy",
    # Settings
    "AgentSettings",
    # Models
    "AgentResult",
    "AgentStep",
    "Message",
    "TeamResult",
    "ToolCall",
    "ToolResult",
    # Guardrails
    "Guardrail",
    "GuardrailPipeline",
    "BlockedWordsGuardrail",
    "MaxLengthGuardrail",
    "PIIGuardrail",
    "RegexGuardrail",
    "TokenBudgetGuardrail",
    # Errors
    "AgentWeaveError",
    "ConfigurationError",
    "GuardrailViolation",
    "LLMBackendError",
    "MaxIterationsError",
    "TokenBudgetExceededError",
    "ToolExecutionError",
    "ToolNotFoundError",
]
