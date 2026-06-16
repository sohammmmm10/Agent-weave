"""Tests for guardrails."""

from __future__ import annotations

import pytest

from agent_weave.errors import GuardrailViolation, TokenBudgetExceededError
from agent_weave.guardrails import (
    BlockedWordsGuardrail,
    GuardrailPipeline,
    MaxLengthGuardrail,
    PIIGuardrail,
    RegexGuardrail,
    TokenBudgetGuardrail,
)


class TestMaxLengthGuardrail:
    def test_passes_under_limit(self) -> None:
        rail = MaxLengthGuardrail(max_chars=100)
        assert rail.check("hello") == "hello"

    def test_fails_over_limit(self) -> None:
        rail = MaxLengthGuardrail(max_chars=5)
        with pytest.raises(GuardrailViolation, match="exceeds limit"):
            rail.check("this is too long")


class TestBlockedWordsGuardrail:
    def test_passes_clean_text(self) -> None:
        rail = BlockedWordsGuardrail(words=["forbidden"])
        assert rail.check("This is fine.") == "This is fine."

    def test_blocks_bad_word(self) -> None:
        rail = BlockedWordsGuardrail(words=["secret"])
        with pytest.raises(GuardrailViolation, match="secret"):
            rail.check("This is a secret message.")

    def test_case_insensitive(self) -> None:
        rail = BlockedWordsGuardrail(words=["password"])
        with pytest.raises(GuardrailViolation):
            rail.check("Your PASSWORD is here.")


class TestRegexGuardrail:
    def test_passes_clean_text(self) -> None:
        rail = RegexGuardrail(r"\d{16}", message="Credit card detected")
        assert rail.check("No numbers here.") == "No numbers here."

    def test_blocks_pattern(self) -> None:
        rail = RegexGuardrail(r"\d{16}", message="Credit card detected")
        with pytest.raises(GuardrailViolation, match="Credit card"):
            rail.check("Card: 1234567890123456")


class TestPIIGuardrail:
    def test_detects_email(self) -> None:
        rail = PIIGuardrail()
        with pytest.raises(GuardrailViolation, match="email"):
            rail.check("Contact me at test@example.com")

    def test_detects_phone(self) -> None:
        rail = PIIGuardrail()
        with pytest.raises(GuardrailViolation, match="phone"):
            rail.check("Call me at 555-123-4567")

    def test_detects_ssn(self) -> None:
        rail = PIIGuardrail()
        with pytest.raises(GuardrailViolation, match="ssn"):
            rail.check("SSN: 123-45-6789")

    def test_redact_mode(self) -> None:
        rail = PIIGuardrail(redact=True)
        result = rail.check("Email: john@test.com, Phone: 555-123-4567")
        assert "[REDACTED_EMAIL]" in result
        assert "[REDACTED_PHONE]" in result
        assert "john@test.com" not in result

    def test_passes_clean_text(self) -> None:
        rail = PIIGuardrail()
        assert rail.check("Nothing personal here.") == "Nothing personal here."


class TestTokenBudgetGuardrail:
    def test_consume_within_budget(self) -> None:
        budget = TokenBudgetGuardrail(budget=1000)
        budget.consume(500)
        assert budget.used == 500
        assert budget.remaining == 500

    def test_exceed_budget(self) -> None:
        budget = TokenBudgetGuardrail(budget=100)
        with pytest.raises(TokenBudgetExceededError):
            budget.consume(150)

    def test_reset(self) -> None:
        budget = TokenBudgetGuardrail(budget=100)
        budget.consume(50)
        budget.reset()
        assert budget.used == 0
        assert budget.remaining == 100


class TestGuardrailPipeline:
    def test_runs_all_guardrails(self) -> None:
        pipeline = GuardrailPipeline([
            MaxLengthGuardrail(max_chars=1000),
            BlockedWordsGuardrail(words=["bomb"]),
        ])
        assert pipeline.run("This is safe.") == "This is safe."

    def test_fails_on_first_violation(self) -> None:
        pipeline = GuardrailPipeline([
            MaxLengthGuardrail(max_chars=5),
            BlockedWordsGuardrail(words=["bomb"]),
        ])
        with pytest.raises(GuardrailViolation, match="exceeds limit"):
            pipeline.run("This is way too long")

    def test_chained_add(self) -> None:
        pipeline = (
            GuardrailPipeline()
            .add(MaxLengthGuardrail(max_chars=100))
            .add(BlockedWordsGuardrail(words=["bad"]))
        )
        assert pipeline.run("good text") == "good text"

    def test_redaction_pipeline(self) -> None:
        pipeline = GuardrailPipeline([
            PIIGuardrail(redact=True),
        ])
        result = pipeline.run("My email is test@foo.com")
        assert "[REDACTED_EMAIL]" in result
