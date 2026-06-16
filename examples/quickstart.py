"""Quickstart example for agent-weave."""

import os

from agent_weave import Agent, tool
from agent_weave.llm.openai_backend import OpenAIBackend

# --- Define tools ---


@tool(description="Get the current weather for a city")
def get_weather(city: str) -> str:
    """Simulated weather lookup."""
    weather_data = {
        "new york": "72F, Sunny",
        "london": "58F, Cloudy",
        "tokyo": "80F, Humid",
        "mumbai": "90F, Partly Cloudy",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool(description="Calculate a math expression")
def calculator(expression: str) -> str:
    """Evaluate a simple math expression safely."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: Only basic math operations are allowed."
    try:
        result = eval(expression)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# --- Create and run the agent ---

def main() -> None:
    backend = OpenAIBackend(
        api_key=os.environ["OPENAI_API_KEY"],
        default_model="gpt-4o-mini",
    )

    agent = Agent(
        name="assistant",
        llm=backend,
        tools=[get_weather, calculator],
        system_prompt=(
            "You are a helpful assistant with access to weather data "
            "and a calculator. Use your tools when needed."
        ),
        max_iterations=5,
        verbose=True,
    )

    # Single task run.
    result = agent.run("What's the weather in Mumbai and what is 42 * 17?")
    print(f"\nAnswer: {result.output}")
    print(f"Steps: {result.total_iterations}, Tokens: {result.total_tokens:,}")
    print(f"Tool calls: {result.total_tool_calls}")


if __name__ == "__main__":
    main()
