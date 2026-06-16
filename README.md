# agent-weave

`agent-weave` is a lightweight Python framework for building AI agents with tool use, ReAct reasoning, multi-agent teams, memory, and guardrails.

No bloat. No magic. Just clean, composable building blocks.

## Features

- **`@tool` decorator** — Turn any Python function into an agent tool with auto-generated schemas.
- **ReAct loop** — Built-in Reasoning + Acting engine with configurable iteration limits.
- **Multi-agent teams** — Sequential pipelines, round-robin, and router-based orchestration.
- **Memory** — Conversation memory and sliding-window memory with system prompt preservation.
- **Guardrails** — PII detection, blocked words, regex filters, token budgets, and max-length checks.
- **Provider backends** — OpenAI, Anthropic, and any OpenAI-compatible API.
- **Async-first** — Full `async/await` support for every operation.
- **CLI** — Run agents and chat from the terminal.

## Install

```bash
pip install -e .
```

With OpenAI support:

```bash
pip install -e ".[openai]"
```

With Anthropic support:

```bash
pip install -e ".[anthropic]"
```

Install everything:

```bash
pip install -e ".[all]"
```

For development:

```bash
pip install -e ".[dev,all]"
```

## Quick Start

```python
import os
from agent_weave import Agent, tool
from agent_weave.llm.openai_backend import OpenAIBackend

@tool(description="Get the weather for a city")
def get_weather(city: str) -> str:
    return f"72°F and sunny in {city}"

@tool(description="Calculate a math expression")
def calculator(expression: str) -> str:
    return str(eval(expression))

agent = Agent(
    name="assistant",
    llm=OpenAIBackend(api_key=os.environ["OPENAI_API_KEY"]),
    tools=[get_weather, calculator],
    system_prompt="You are a helpful assistant. Use tools when needed.",
)

result = agent.run("What's the weather in NYC and what is 42 * 17?")
print(result.output)
print(f"Steps: {result.total_iterations}, Tokens: {result.total_tokens:,}")
```

## The `@tool` Decorator

Turn any function into a tool. Schemas are auto-generated from type hints:

```python
from agent_weave import tool

@tool(description="Search the web for a query")
def web_search(query: str, max_results: int = 5) -> str:
    return f"Results for: {query} (limit {max_results})"

# Access the generated schema
print(web_search.schema.to_openai_tool())
```

## Multi-Agent Teams

Chain agents in a pipeline, round-robin, or route to specialists:

```python
from agent_weave import Agent, Team, Strategy
from agent_weave.llm.openai_backend import OpenAIBackend

backend = OpenAIBackend(api_key=os.environ["OPENAI_API_KEY"])

researcher = Agent(name="researcher", llm=backend,
    system_prompt="Research the topic. Provide key facts.")

writer = Agent(name="writer", llm=backend,
    system_prompt="Write a blog post from the research provided.")

editor = Agent(name="editor", llm=backend,
    system_prompt="Polish and improve the writing.")

# Sequential: researcher -> writer -> editor
team = Team(
    agents=[researcher, writer, editor],
    strategy=Strategy.SEQUENTIAL,
)
result = team.run("AI agents in 2025")
print(result.final_output)
```

### Router Strategy

```python
router = Agent(name="router", llm=backend,
    system_prompt="You route tasks to the right specialist.")

team = Team(
    agents=[researcher, writer],
    strategy=Strategy.ROUTER,
    router=router,
)
result = team.run("Write a poem about AI")
```

## Memory

```python
from agent_weave.memory import ConversationMemory, SlidingWindowMemory

# Unlimited memory
agent = Agent(name="bot", llm=backend, memory=ConversationMemory())

# Fixed window (keeps last 20 messages + system prompt)
agent = Agent(name="bot", llm=backend,
    memory=SlidingWindowMemory(max_messages=20))
```

## Guardrails

```python
from agent_weave import (
    Agent, MaxLengthGuardrail, PIIGuardrail, BlockedWordsGuardrail,
)

agent = Agent(
    name="safe-bot",
    llm=backend,
    token_budget=10_000,  # Max 10k tokens per run
    output_guardrails=[
        MaxLengthGuardrail(max_chars=5_000),
        PIIGuardrail(redact=True),
        BlockedWordsGuardrail(words=["confidential", "password"]),
    ],
)
```

## Conversational Chat

```python
agent = Agent(name="chatbot", llm=backend,
    system_prompt="You are a friendly chatbot.")

# chat() preserves history across calls
agent.chat("Hello!")
agent.chat("What did I just say?")  # Agent remembers
agent.reset()  # Clear conversation
```

## Async Support

```python
import asyncio

async def main():
    result = await agent.arun("Summarize AI trends")
    print(result.output)

asyncio.run(main())
```

## Anthropic Backend

```python
from agent_weave.llm.anthropic_backend import AnthropicBackend

agent = Agent(
    name="claude-agent",
    llm=AnthropicBackend(api_key=os.environ["ANTHROPIC_API_KEY"]),
    system_prompt="You are helpful.",
)
result = agent.run("Explain quantum computing simply.")
```

## CLI

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Run a single task
agent-weave run "What are the top 3 AI trends in 2025?"

# Interactive chat
agent-weave chat

# Library info
agent-weave info
```

## Run Tests

```bash
pip install -e ".[dev]"
python -m pytest
```

## Project Structure

```
agent-weave/
├── src/agent_weave/
│   ├── __init__.py          # Public API
│   ├── agent.py             # Core Agent class
│   ├── tool.py              # @tool decorator & Tool class
│   ├── react.py             # ReAct reasoning engine
│   ├── team.py              # Multi-agent orchestration
│   ├── guardrails.py        # Safety & validation
│   ├── config.py            # Settings
│   ├── models.py            # Data models
│   ├── errors.py            # Custom exceptions
│   ├── cli.py               # CLI interface
│   ├── memory/
│   │   ├── base.py          # Memory interface
│   │   └── conversation.py  # Memory implementations
│   └── llm/
│       ├── base.py          # LLM backend interface
│       ├── openai_backend.py
│       └── anthropic_backend.py
├── tests/
├── examples/
├── pyproject.toml
└── README.md
```

## License

MIT
