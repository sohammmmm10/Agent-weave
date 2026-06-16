"""Tool definition and @tool decorator for agent-weave."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints

from .errors import ToolExecutionError
from .models import ToolSchema

# Python type -> JSON Schema type mapping.
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(annotation: Any) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema fragment."""
    origin = getattr(annotation, "__origin__", None)

    if origin is list:
        args = getattr(annotation, "__args__", ())
        items = _python_type_to_json_schema(args[0]) if args else {}
        return {"type": "array", "items": items}

    if origin is dict:
        return {"type": "object"}

    if annotation in _TYPE_MAP:
        return {"type": _TYPE_MAP[annotation]}

    return {"type": "string"}


def _build_parameters_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Auto-generate a JSON Schema for a function's parameters from type hints."""
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        prop: dict[str, Any] = {}
        if name in hints:
            prop = _python_type_to_json_schema(hints[name])

        # Use default as description hint.
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            prop["default"] = param.default

        properties[name] = prop

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


@dataclass
class Tool:
    """A callable tool that an agent can use."""

    name: str
    description: str
    func: Callable[..., Any]
    parameters_schema: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.parameters_schema:
            self.parameters_schema = _build_parameters_schema(self.func)

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters_schema,
        )

    def execute(self, arguments: dict[str, Any] | None = None) -> str:
        """Execute the tool with the given arguments and return string output."""
        args = arguments or {}
        try:
            result = self.func(**args)
            if isinstance(result, str):
                return result
            return json.dumps(result, default=str, ensure_ascii=False)
        except Exception as exc:
            raise ToolExecutionError(self.name, str(exc)) from exc

    async def aexecute(self, arguments: dict[str, Any] | None = None) -> str:
        """Execute the tool asynchronously."""
        args = arguments or {}
        try:
            result = self.func(**args)
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, str):
                return result
            return json.dumps(result, default=str, ensure_ascii=False)
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(self.name, str(exc)) from exc


def tool(
    name: str | None = None,
    description: str | None = None,
    parameters_schema: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Tool]:
    """Decorator to register a function as an agent tool.

    Usage::

        @tool(description="Search the web for a query")
        def web_search(query: str) -> str:
            return f"Results for: {query}"

        @tool(name="calculator", description="Evaluate math expressions")
        def calc(expression: str) -> str:
            return str(eval(expression))
    """

    def decorator(func: Callable[..., Any]) -> Tool:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or f"Tool: {tool_name}"
        return Tool(
            name=tool_name,
            description=tool_desc.strip(),
            func=func,
            parameters_schema=parameters_schema or {},
        )

    return decorator
