"""MVP: Agent A -> Communication Layer -> Agent B -> reply -> Agent A.

Runs fully offline (MockLLMAdapter). Set ``AGENT_B_MODEL`` (e.g.
``anthropic:claude-3-5-haiku-latest``) and the matching API key in ``.env`` to
make Agent B a real LLM.

    python examples/roundtrip.py
"""

from __future__ import annotations

import asyncio
import os

from agentcomm import (
    AgentInfo,
    CommunicationLayer,
    EchoAgent,
    LLMAgent,
    configure_logging,
    load_dotenv,
)
from agentcomm.llm import create_adapter


async def main() -> None:
    load_dotenv()
    configure_logging()

    layer = CommunicationLayer(default_timeout=60)

    # Agent A - a plain agent that asks questions.
    agent_a = EchoAgent(layer, "agent_a", "Agent A")

    # Agent B - an LLM agent (mock by default).
    adapter = create_adapter(os.environ.get("AGENT_B_MODEL", "mock"))
    agent_b = LLMAgent(
        layer,
        AgentInfo(id="agent_b", name="Agent B", role="assistant",
                  capabilities=["chat", "analysis"]),
        adapter,
    )

    await agent_a.start()
    await agent_b.start()

    print("\n=== Agent A -> Agent B -> Agent A ===")
    reply = await agent_a.ask("agent_b", "Hello Agent B, what are you capable of?")
    print(f"[{reply.sender} -> {reply.receiver}] ({reply.message_type.value}) {reply.content}")

    print("\n=== Conversation history ===")
    for m in layer.conversation(reply.conversation_id):
        print(f"  {m.timestamp}  {m.sender:>8} -> {m.receiver:<8} {m.message_type.value:<10} "
              f"{m.status.value:<10} {m.content[:60]!r}")

    await agent_a.stop()
    await agent_b.stop()
    print("\nOK: roundtrip succeeded")


if __name__ == "__main__":
    asyncio.run(main())
