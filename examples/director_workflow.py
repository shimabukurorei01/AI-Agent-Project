"""Director Agent MVP workflow:

    User -> Director Agent -> Research Agent -> Coding Agent -> Director Agent -> Final Result

* Default (offline): SequentialPlanner (rule based) + mock workers.
* Set ``DIRECTOR_MODEL=anthropic:claude-3-5-sonnet-latest`` (any provider) to let an
  LLM analyse the goal, decompose it dynamically, add follow-up tasks and write the
  final synthesis.  Workers: ``RESEARCH_MODEL`` / ``CODING_MODEL`` / ``REVIEWER_MODEL``.

    python examples/director_workflow.py
    DIRECTOR_MODEL=openai:gpt-4o RESEARCH_MODEL=google:gemini-2.0-flash python examples/director_workflow.py
"""

from __future__ import annotations

import asyncio
import json
import os

from agentcomm import (
    AgentInfo,
    CommunicationLayer,
    EchoAgent,
    LLMAgent,
    MessageType,
    configure_logging,
    load_dotenv,
)
from agentcomm.llm import MockLLMAdapter, create_adapter
from agentcomm.orchestration import DirectorAgent, LLMPlanner, Planner, SequentialPlanner

GOAL = "Build a rate limiter for our public API."


def _mock_worker(role: str) -> MockLLMAdapter:
    """Offline stand-ins that make the data flow visible."""
    def respond(msgs: list, system: str | None) -> str:  # type: ignore[type-arg]
        last = msgs[-1].content
        if role == "researcher":
            return "Findings: token-bucket is the standard choice; sliding-window is an alternative."
        if role == "coder":
            used = "token-bucket" if "token-bucket" in last else "generic"
            return f"```python\nclass RateLimiter:  # {used} implementation\n    ...\n```"
        return f"[{role}] processed: {last[:40]}..."
    return MockLLMAdapter(model=f"mock-{role}", responder=respond)


def _save_plan(plan: dict) -> None:  # type: ignore[type-arg]
    os.makedirs("data", exist_ok=True)
    with open("data/last_plan.json", "w", encoding="utf-8") as fh:
        json.dump(plan, fh, ensure_ascii=False, indent=2)


async def main() -> None:
    load_dotenv()
    configure_logging()
    layer = CommunicationLayer(default_timeout=180)

    workers = []
    for agent_id, name, role, caps, env in [
        ("research_agent", "Research Agent", "researcher", ["research", "analysis"], "RESEARCH_MODEL"),
        ("coding_agent", "Coding Agent", "coder", ["coding"], "CODING_MODEL"),
        ("reviewer_agent", "Reviewer Agent", "reviewer", ["review"], "REVIEWER_MODEL"),
    ]:
        spec = os.environ.get(env)
        adapter = create_adapter(spec) if spec else _mock_worker(role)
        workers.append(LLMAgent(layer, AgentInfo(id=agent_id, name=name, role=role, capabilities=caps), adapter))

    planner: Planner
    if os.environ.get("DIRECTOR_MODEL"):
        planner = LLMPlanner(create_adapter(os.environ["DIRECTOR_MODEL"]), max_replans=1)
        mode = f"LLMPlanner ({os.environ['DIRECTOR_MODEL']})"
    else:
        planner = SequentialPlanner(["researcher", "coder"])
        mode = "SequentialPlanner (researcher -> coder)"

    director = DirectorAgent(
        layer,
        AgentInfo(id="director_agent", name="Director Agent", role="director",
                  capabilities=["planning", "delegation", "synthesis"]),
        planner,
        task_timeout=150,
        max_retries=1,
    )
    user = EchoAgent(layer, "user", "User")

    for ag in (*workers, director, user):
        await ag.start()

    print(f"\n=== Planner: {mode} ===")
    print(f"=== User -> Director: {GOAL} ===\n")
    result = await user.ask("director_agent", GOAL, message_type=MessageType.TASK_REQUEST,
                            task_id="task_director_001", conversation_id="conv_director_001")

    plan = result.metadata["plan"]
    print("=== Final result ===")
    print(result.content)

    print(f"\n=== Plan {plan['plan_id']}  status={plan['status']}  rounds={plan['rounds']} ===")
    print("Analysis:", plan["analysis"])
    for st in plan["subtasks"]:
        deps = ",".join(st["depends_on"]) or "-"
        print(f"  [{st['id']:<14}] role={st['role']:<10} -> {st['assigned_to'] or '-':<15} "
              f"{st['status']:<8} attempts={st['attempts']} deps={deps} task_id={st['task_id']}")

    print("\n=== Event trail ===")
    for e in plan["events"]:
        print(f"  {e['timestamp'][11:23]}  {e['event']:<11} {e['subtask_id'] or '-':<14} {e['detail'][:70]}")

    print("\n=== Messages in conversation (all share conversation_id) ===")
    for m in layer.conversation("conv_director_001"):
        print(f"  {m.sender:>15} -> {m.receiver:<15} {m.message_type.value:<13} {m.status.value:<9} "
              f"task_id={m.task_id}")

    for ag in (*workers, director, user):
        await ag.stop()

    await asyncio.to_thread(_save_plan, plan)
    print("\nOK - plan saved to data/last_plan.json")


if __name__ == "__main__":
    asyncio.run(main())
