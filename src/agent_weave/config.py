"""Configuration for agent-weave."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _as_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_float(value: str | None, *, default: float | None) -> float | None:
    if value is None or value.strip() == "":
        return default
    return float(value)


def _as_int(value: str | None, *, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    return int(value)


@dataclass
class AgentSettings:
    """Global settings for agent-weave."""

    default_model: str = "gpt-4o-mini"
    max_iterations: int = 10
    request_timeout: float | None = 60.0
    token_budget: int | None = None
    verbose: bool = False

    @classmethod
    def from_env(cls) -> "AgentSettings":
        return cls(
            default_model=os.getenv("AGENTWEAVE_DEFAULT_MODEL", "gpt-4o-mini"),
            max_iterations=_as_int(
                os.getenv("AGENTWEAVE_MAX_ITERATIONS"), default=10
            ),
            request_timeout=_as_float(
                os.getenv("AGENTWEAVE_REQUEST_TIMEOUT"), default=60.0
            ),
            token_budget=_as_int(
                os.getenv("AGENTWEAVE_TOKEN_BUDGET"), default=0
            ) or None,
            verbose=_as_bool(os.getenv("AGENTWEAVE_VERBOSE"), default=False),
        )
