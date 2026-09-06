"""Four agents (A, B, C, D) online at once: broadcast, concurrent 1:1 requests,
and graceful removal of one agent while the others keep working.

    python examples/four_agents.py

Each agent's model can be chosen via env, e.g. ``AGENT_C_MODEL=google:gemini-2.0-flash``.
Defaults to ``mock`` so the demo runs offline.
"""

from __future__ import annotations

import asyncio
import os

from agentcomm import (
    AgentInfo,
    CommunicationLayer,
    LLMAgent,
    MessageType,
    configure_logging,
    load_dotenv,
)
from agentcomm.errors import AgentNotFoundError
from agentcomm.llm import create_adapter

AGENTS = [
    ("agent_a", "Agent A (ChatGPT)", "generalist", ["chat"], "AGENT_A_MODEL"),
    ("agent_b", "Agent B (Claude)", "researcher", ["research", "analysis"], "AGENT_B_MODEL"),
    ("agent_c", "Agent C (Gemini)", "coder", ["coding"], "AGENT_C_MODEL"),
    ("agent_d", "Agent D (Grok)", "reviewer", ["review", "testing"], "AGENT_D_MODEL"),
]


async def main() -> None:
    load_dotenv()
    configure_logging()
    layer = CommunicationLayer(default_timeout=60)

    agents: list[LLMAgent] = []
    for agent_id, name, role, caps, env in AGENTS:
        adapter = create_adapter(os.environ.get(env, "mock"))
        agents.append(LLMAgent(layer, AgentInfo(id=agent_id, name=name, role=role, capabilities=caps), adapter))
    for ag in agents:
        await ag.start()

    a, b, c, d = agents
    print("\n=== Online agents ===")
    for info in layer.agents(online_only=True):
        print(f"  {info.id:8} {info.name:20} role={info.role:<11} model={info.model}")

    print("\n=== Broadcast from A ===")
    targets = await a.broadcast("Kick-off: we are building a multi-agent system.")
    print("  delivered to:", targets)

    print("\n=== Concurrent 1:1 requests (A asks B, C, D at once) ===")
    conv = "conv_demo_001"
    replies = await asyncio.gather(
        a.ask("agent_b", "Research: what is A2A communication?", conversation_id=conv),
        a.ask("agent_c", "Coding: sketch a message router in pseudocode.", conversation_id=conv),
        a.ask("agent_d", "Review: what could go wrong with in-memory queues?", conversation_id=conv,
              message_type=MessageType.REVIEW_REQUEST),
    )
    for r in replies:
        print(f"  {r.sender} ({r.message_type.value}): {r.content[:80]}")

    print("\n=== Remove Agent C; others keep working ===")
    await c.stop()
    try:
        await a.ask("agent_c", "still there?")
    except AgentNotFoundError as exc:
        print("  expected error:", exc)
    r = await b.ask("agent_d", "Can you review my findings?", conversation_id=conv)
    print(f"  {r.sender}: {r.content[:80]}")

    print(f"\n=== History: {len(layer.conversation(conv))} messages in {conv} ===")
    for ag in (a, b, d):
        await ag.stop()
    print("\nOK")


if __name__ == "__main__":
    asyncio.run(main())
