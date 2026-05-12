"""Interactive agent session for helix-rag.

Starts a conversation loop with the Research Assistant agent.
Type your question and press Enter. The agent decides whether to
search the papers or list available documents.

Type 'quit' or 'exit' to end the session.

Usage:
  python scripts/run_agent.py
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import Runner  # noqa: E402
from loguru import logger  # noqa: E402

from src.agent.agent import build_agent  # noqa: E402


def main() -> None:
    agent = build_agent()
    history = []

    print("\n" + "=" * 60)
    print("Research Assistant — helix-rag")
    print("Ask a question about the indexed papers.")
    print("Type 'quit' to exit.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        try:
            if history:
                input_data = history + [{"role": "user", "content": user_input}]
            else:
                input_data = user_input
            result = Runner.run_sync(agent, input_data)
            history = result.to_input_list()
            print(f"\nAssistant: {result.final_output}\n")
        except Exception as e:
            logger.error(f"Agent error: {e}")
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
