"""Guardrails for agent-weave — input/output validators and budget controls."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .errors import GuardrailViolation, TokenBudgetExceededError


class Guardrail(ABC):
    """Base class for all guardrails."""

    name: str = "base"

    @abstractmethod
    def check(self, value: str, *, context: dict[str, Any] | None = None) -> str:
        """Validate and optionally transform the value.

        Returns the (possibly modified) value.
        Raises ``GuardrailViolation`` if the check fails.
        """


class MaxLengthGuardrail(Guardrail):
    """Reject output that exceeds a character limit."""

    name = "max_length"

    def __init__(self, max_chars: int = 10_000) -> None:
        self._max = max_chars

    def check(self, value: str, *, context: dict[str, Any] | None = None) -> str:
        if len(value) > self._max:
            raise GuardrailViolation(
                self.name,
                f"Output length {len(value):,} exceeds limit of {self._max:,} chars.",
            )
        return value


class BlockedWordsGuardrail(Guardrail):
    """Reject output that contains any blocked words."""

    name = "blocked_words"

    def __init__(self, words: list[str]) -> None:
        self._words = [w.lower() for w in words]

    def check(self, value: str, *, context: dict[str, Any] | None = None) -> str:
        lower_val = value.lower()
        for word in self._words:
            if word in lower_val:
                raise GuardrailViolation(
                    self.name, f"Blocked word detected: '{word}'"
                )
        return value


class RegexGuardrail(Guardrail):
    """Reject output that matches a forbidden regex pattern."""

    name = "regex_filter"

    def __init__(self, pattern: str, *, message: str = "Forbidden pattern detected.") -> None:
        self._pattern = re.compile(pattern, re.IGNORECASE)
        self._message = message

    def check(self, value: str, *, context: dict[str, Any] | None = None) -> str:
        if self._pattern.search(value):
            raise GuardrailViolation(self.name, self._message)
        return value


class PIIGuardrail(Guardrail):
    """Detect common PII patterns (emails, phone numbers, SSNs)."""

    name = "pii_filter"

    _PATTERNS = {
        "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
        "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    }

    def __init__(self, *, redact: bool = False) -> None:
        self._redact = redact

    def check(self, value: str, *, context: dict[str, Any] | None = None) -> str:
        for pii_type, pattern in self._PATTERNS.items():
            if pattern.search(value):
                if self._redact:
                    value = pattern.sub(f"[REDACTED_{pii_type.upper()}]", value)
                else:
                    raise GuardrailViolation(
                        self.name,
                        f"PII detected: {pii_type}. Set redact=True to auto-redact.",
                    )
        return value


@dataclass
class TokenBudgetGuardrail:
    """Track and enforce a token budget across the agent run."""

    budget: int
    _used: int = field(default=0, init=False)

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self._used)

    def consume(self, tokens: int) -> None:
        """Record token usage. Raises if budget exceeded."""
        self._used += tokens
        if self._used > self.budget:
            raise TokenBudgetExceededError(self.budget, self._used)

    def reset(self) -> None:
        self._used = 0


class GuardrailPipeline:
    """Run a sequence of guardrails on a value."""

    def __init__(self, guardrails: list[Guardrail] | None = None) -> None:
        self._guardrails = list(guardrails or [])

    def add(self, guardrail: Guardrail) -> "GuardrailPipeline":
        self._guardrails.append(guardrail)
        return self

    def run(self, value: str, *, context: dict[str, Any] | None = None) -> str:
        """Run all guardrails in sequence. Returns the (possibly modified) value."""
        result = value
        for rail in self._guardrails:
            result = rail.check(result, context=context)
        return result
