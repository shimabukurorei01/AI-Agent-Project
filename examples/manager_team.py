"""Hierarchical team:

    Manager Agent
     |- Research Agent
     |- Coding Agent
     |- Reviewer Agent
     `- Testing Agent

The user sends one task to the manager, which delegates sub-tasks by *role*
concurrently, waits for the results (with timeouts) and returns a combined
report. Works offline with ``mock``; set ``<ROLE>_MODEL`` env vars to use real
LLMs, e.g. ``RESEARCH_MODEL=anthropic:claude-3-5-sonnet-latest``.

    python examples/manager_team.py
"""

from __future__ import annotations

import asyncio
import os

from agentcomm import (
    AgentInfo,
    CommunicationLayer,
    EchoAgent,
    JsonlHistory,
    LLMAgent,
    ManagerAgent,
    MessageType,
    configure_logging,
    load_dotenv,
)
from agentcomm.llm import create_adapter

WORKERS = [
    ("research_agent", "Research Agent", "researcher", ["research", "analysis"], "RESEARCH_MODEL"),
    ("coding_agent", "Coding Agent", "coder", ["coding", "refactoring"], "CODING_MODEL"),
    ("reviewer_agent", "Reviewer Agent", "reviewer", ["review", "security"], "REVIEWER_MODEL"),
    ("testing_agent", "Testing Agent", "tester", ["testing", "qa"], "TESTING_MODEL"),
]


async def main() -> None:
    load_dotenv()
    configure_logging()

    # Persist history to disk so a restart keeps the memory.
    history = JsonlHistory(os.environ.get("AGENTCOMM_HISTORY_PATH", "data/history.jsonl"))
    layer = CommunicationLayer(history=history, default_timeout=120)

    manager_adapter = None
    if os.environ.get("MANAGER_MODEL"):
        manager_adapter = create_adapter(os.environ["MANAGER_MODEL"])
    manager = ManagerAgent(
        layer,
        AgentInfo(id="manager_agent", name="Manager Agent", role="manager",
                  capabilities=["planning", "delegation"]),
        adapter=manager_adapter,
        task_timeout=120,
    )

    workers = [
        LLMAgent(layer, AgentInfo(id=i, name=n, role=r, capabilities=c),
                 create_adapter(os.environ.get(env, "mock")))
        for i, n, r, c, env in WORKERS
    ]
    user = EchoAgent(layer, "user", "User")  # the human is just another participant

    for ag in (manager, *workers, user):
        await ag.start()

    goal = "Build a rate limiter for our public API."
    plan = {
        "researcher": f"Research best-practice algorithms for: {goal}",
        "coder": f"Write a Python implementation for: {goal}",
        "reviewer": f"List review criteria for: {goal}",
        "tester": f"Propose a test plan for: {goal}",
    }

    print(f"\n=== User -> Manager: {goal} ===")
    result = await user.ask(
        "manager_agent", goal, message_type=MessageType.TASK_REQUEST,
        task_id="task_demo_001", metadata={"plan": plan},
    )
    print(result.content)

    print("\n=== Task history (task_demo_001*) ===")
    for m in layer.history.all():
        if m.task_id and m.task_id.startswith("task_demo_001"):
            print(f"  {m.sender:>15} -> {m.receiver:<15} {m.message_type.value:<13} {m.status.value}")

    for ag in (manager, *workers, user):
        await ag.stop()
    print(f"\nOK - history persisted to {history._path}")


if __name__ == "__main__":
    asyncio.run(main())
