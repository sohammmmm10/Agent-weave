"""Advanced tools and guardrails example for agent-weave."""

import json
import os

from agent_weave import (
    Agent,
    BlockedWordsGuardrail,
    MaxLengthGuardrail,
    PIIGuardrail,
    tool,
)
from agent_weave.llm.openai_backend import OpenAIBackend

# --- Custom tools ---


@tool(description="Search a knowledge base for information about a topic")
def knowledge_search(query: str) -> str:
    """Simulated knowledge base search."""
    kb = {
        "python": "Python is a high-level programming language known for simplicity.",
        "rust": "Rust is a systems language focused on safety and performance.",
        "agents": "AI agents are autonomous systems that can reason and use tools.",
        "rag": "RAG combines retrieval with generation for grounded AI responses.",
    }
    results = []
    for key, value in kb.items():
        if key in query.lower():
            results.append(value)
    return "\n".join(results) if results else "No results found."


@tool(
    name="create_summary",
    description="Create a structured JSON summary from text",
)
def create_summary(text: str, max_points: int = 3) -> str:
    """Create a bullet-point summary."""
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    points = sentences[:max_points]
    return json.dumps({"summary_points": points, "total_sentences": len(sentences)})


@tool(description="Get the current date and time")
def get_current_time() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# --- Agent with guardrails ---


def main() -> None:
    backend = OpenAIBackend(
        api_key=os.environ["OPENAI_API_KEY"],
        default_model="gpt-4o-mini",
    )

    agent = Agent(
        name="safe-researcher",
        llm=backend,
        tools=[knowledge_search, create_summary, get_current_time],
        system_prompt=(
            "You are a research assistant with access to a knowledge base. "
            "Always search before answering. Provide factual, concise answers."
        ),
        max_iterations=8,
        token_budget=10_000,
        output_guardrails=[
            MaxLengthGuardrail(max_chars=5_000),
            PIIGuardrail(redact=True),
            BlockedWordsGuardrail(words=["confidential", "secret"]),
        ],
        verbose=True,
    )

    result = agent.run("What can you tell me about Python and AI agents?")
    print(f"\nAnswer: {result.output}")
    print(f"\nSteps: {result.total_iterations}")
    print(f"Tokens: {result.total_tokens:,}")
    print(f"Tool calls: {result.total_tool_calls}")


if __name__ == "__main__":
    main()
