"""Multi-agent team example for agent-weave."""

import os

from agent_weave import Agent, Strategy, Team
from agent_weave.llm.openai_backend import OpenAIBackend


def main() -> None:
    backend = OpenAIBackend(
        api_key=os.environ["OPENAI_API_KEY"],
        default_model="gpt-4o-mini",
    )

    # --- Sequential pipeline: Researcher -> Writer -> Editor ---

    researcher = Agent(
        name="researcher",
        llm=backend,
        system_prompt=(
            "You are a research specialist. Given a topic, provide "
            "3-5 key facts with brief explanations. Be factual and concise."
        ),
    )

    writer = Agent(
        name="writer",
        llm=backend,
        system_prompt=(
            "You are a blog writer. Take the research provided and "
            "write a compelling 200-word blog post section. "
            "Make it engaging and readable."
        ),
    )

    editor = Agent(
        name="editor",
        llm=backend,
        system_prompt=(
            "You are an editor. Review and polish the text. "
            "Fix grammar, improve clarity, and ensure a professional tone. "
            "Return the final polished version."
        ),
    )

    team = Team(
        agents=[researcher, writer, editor],
        strategy=Strategy.SEQUENTIAL,
        verbose=True,
    )

    result = team.run("AI agents and their impact on software development in 2025")

    print("=" * 60)
    print("FINAL OUTPUT")
    print("=" * 60)
    print(result.final_output)
    print(f"\nAgents used: {result.agent_names}")
    print(f"Total tokens: {result.total_tokens:,}")


if __name__ == "__main__":
    main()
