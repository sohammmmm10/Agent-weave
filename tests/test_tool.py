"""Tests for the tool module."""

from __future__ import annotations

import pytest

from agent_weave import Tool, tool
from agent_weave.errors import ToolExecutionError


class TestToolDecorator:
    def test_basic_decorator(self) -> None:
        @tool(description="Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b

        assert isinstance(add, Tool)
        assert add.name == "add"
        assert add.description == "Add two numbers"

    def test_custom_name(self) -> None:
        @tool(name="my_adder", description="Custom add")
        def add(a: int, b: int) -> int:
            return a + b

        assert add.name == "my_adder"

    def test_docstring_fallback(self) -> None:
        @tool(name="greet")
        def greet(name: str) -> str:
            """Say hello to someone."""
            return f"Hello, {name}!"

        assert greet.description == "Say hello to someone."


class TestToolExecution:
    def test_execute_returns_string(self) -> None:
        @tool(description="Echo")
        def echo(text: str) -> str:
            return text

        assert echo.execute({"text": "hi"}) == "hi"

    def test_execute_serializes_non_string(self) -> None:
        @tool(description="Sum")
        def add(a: int, b: int) -> int:
            return a + b

        result = echo = add.execute({"a": 2, "b": 3})
        assert result == "5"

    def test_execute_handles_dict_output(self) -> None:
        @tool(description="Info")
        def info() -> dict:
            return {"name": "agent-weave", "version": "0.1.0"}

        result = info.execute({})
        assert "agent-weave" in result

    def test_execute_raises_tool_execution_error(self) -> None:
        @tool(description="Fail")
        def fail() -> str:
            raise ValueError("boom")

        with pytest.raises(ToolExecutionError, match="boom"):
            fail.execute({})

    def test_execute_empty_args(self) -> None:
        @tool(description="No args")
        def hello() -> str:
            return "world"

        assert hello.execute() == "world"
        assert hello.execute(None) == "world"


class TestToolSchema:
    def test_auto_schema_generation(self) -> None:
        @tool(description="Search")
        def search(query: str, limit: int = 10) -> str:
            return query

        schema = search.schema
        assert schema.name == "search"
        params = schema.parameters
        assert "query" in params["properties"]
        assert "limit" in params["properties"]
        assert "query" in params["required"]
        assert "limit" not in params["required"]

    def test_openai_tool_format(self) -> None:
        @tool(description="Ping")
        def ping() -> str:
            return "pong"

        oai = ping.schema.to_openai_tool()
        assert oai["type"] == "function"
        assert oai["function"]["name"] == "ping"
        assert oai["function"]["description"] == "Ping"

    def test_anthropic_tool_format(self) -> None:
        @tool(description="Ping")
        def ping() -> str:
            return "pong"

        anth = ping.schema.to_anthropic_tool()
        assert anth["name"] == "ping"
        assert anth["description"] == "Ping"
        assert "input_schema" in anth

    def test_type_mapping(self) -> None:
        @tool(description="Types")
        def typed(
            name: str,
            age: int,
            score: float,
            active: bool,
            tags: list[str],
        ) -> str:
            return "ok"

        props = typed.schema.parameters["properties"]
        assert props["name"]["type"] == "string"
        assert props["age"]["type"] == "integer"
        assert props["score"]["type"] == "number"
        assert props["active"]["type"] == "boolean"
        assert props["tags"]["type"] == "array"
