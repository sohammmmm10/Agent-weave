"""CLI interface for agent-weave."""

from __future__ import annotations

import argparse
import json
import os
import sys


def _get_agent():
    """Build a default agent from environment variables."""
    from .agent import Agent
    from .llm.openai_backend import OpenAIBackend

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is required.", file=sys.stderr)
        sys.exit(1)

    model = os.getenv("AGENTWEAVE_DEFAULT_MODEL", "gpt-4o-mini")

    backend = OpenAIBackend(api_key=api_key, default_model=model)
    return Agent(
        name="cli-agent",
        llm=backend,
        system_prompt="You are a helpful AI assistant. Be concise.",
        model=model,
    )


def cmd_run(args: argparse.Namespace) -> None:
    """Run a task with the agent."""
    agent = _get_agent()
    result = agent.run(args.task)
    print(result.output)

    if args.verbose:
        print(f"\n--- Stats ---")
        print(f"Iterations: {result.total_iterations}")
        print(f"Tokens: {result.total_tokens:,}")
        print(f"Tool calls: {result.total_tool_calls}")


def cmd_chat(args: argparse.Namespace) -> None:
    """Interactive chat with the agent."""
    agent = _get_agent()
    print("Agent Weave Chat (type 'quit' to exit)")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        result = agent.chat(user_input)
        print(f"\nAgent: {result.output}")


def cmd_info(args: argparse.Namespace) -> None:
    """Show library info."""
    from . import __version__
    info = {
        "name": "agent-weave",
        "version": __version__,
        "python": sys.version,
    }
    print(json.dumps(info, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agent-weave",
        description="Lightweight AI agent framework.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Run command.
    run_parser = subparsers.add_parser("run", help="Run a task")
    run_parser.add_argument("task", help="The task to execute")
    run_parser.add_argument("-v", "--verbose", action="store_true")
    run_parser.set_defaults(func=cmd_run)

    # Chat command.
    chat_parser = subparsers.add_parser("chat", help="Interactive chat")
    chat_parser.set_defaults(func=cmd_chat)

    # Info command.
    info_parser = subparsers.add_parser("info", help="Show library info")
    info_parser.set_defaults(func=cmd_info)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
