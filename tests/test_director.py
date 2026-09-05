"""DirectorAgent end-to-end tests (offline; MockLLMAdapter everywhere)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

from agentcomm import (
    AgentInfo,
    BaseAgent,
    CommunicationLayer,
    EchoAgent,
    LLMAgent,
    Message,
    MessageType,
)
from agentcomm.llm import ChatMessage, MockLLMAdapter
from agentcomm.orchestration import (
    DirectorAgent,
    LLMPlanner,
    Plan,
    PlanStatus,
    SequentialPlanner,
    StaticPlanner,
    SubTask,
    SubTaskStatus,
)


def _worker(layer: CommunicationLayer, agent_id: str, role: str, adapter: MockLLMAdapter) -> LLMAgent:
    return LLMAgent(layer, AgentInfo(id=agent_id, name=agent_id, role=role, capabilities=[role]), adapter)


def _director(layer: CommunicationLayer, planner: Any, **kw: Any) -> DirectorAgent:
    kw.setdefault("task_timeout", 1.0)
    return DirectorAgent(layer, AgentInfo(id="director", name="Director", role="director"), planner, **kw)


async def _start(*agents: BaseAgent) -> None:
    for a in agents:
        await a.start()


async def _stop(*agents: BaseAgent) -> None:
    for a in agents:
        await a.stop()


# --------------------------------------------------------------------- MVP workflow
async def test_mvp_user_director_research_coding_director(layer: CommunicationLayer) -> None:
    """User -> Director -> Research -> Coding -> Director -> Final Result."""
    research = _worker(layer, "research_agent", "researcher",
                       MockLLMAdapter(scripted=["FINDING: use token bucket"]))

    def code(msgs: list[ChatMessage], system: str | None) -> str:
        # the coder must receive the researcher's output as input
        assert "FINDING: use token bucket" in msgs[-1].content
        assert "[s1_researcher]" in msgs[-1].content
        return "CODE: class TokenBucket"

    coding = _worker(layer, "coding_agent", "coder", MockLLMAdapter(responder=code))
    director = _director(layer, SequentialPlanner(["researcher", "coder"]))
    user = EchoAgent(layer, "user")
    await _start(research, coding, director, user)
    try:
        result = await user.ask("director", "Build a rate limiter", message_type=MessageType.TASK_REQUEST,
                                task_id="task_1", conversation_id="conv_1", timeout=5)
        assert result.message_type is MessageType.TASK_RESULT
        assert "FINDING: use token bucket" in result.content
        assert "CODE: class TokenBucket" in result.content

        plan = result.metadata["plan"]
        assert plan["status"] == "completed"
        assert plan["task_id"] == "task_1" and plan["conversation_id"] == "conv_1"
        ids = [s["id"] for s in plan["subtasks"]]
        assert ids == ["s1_researcher", "s2_coder"]
        assert plan["subtasks"][1]["depends_on"] == ["s1_researcher"]
        for st in plan["subtasks"]:
            assert st["status"] == "done" and st["attempts"] == 1
            assert st["task_id"].startswith("task_1:")
            assert st["request_message_id"] and st["result_message_id"]
        assert plan["subtasks"][0]["assigned_to"] == "research_agent"
        assert plan["subtasks"][1]["assigned_to"] == "coding_agent"
        assert [e["event"] for e in plan["events"]] == [
            "planned", "dispatched", "completed", "dispatched", "completed", "finished"]

        # every message in the workflow shares the conversation id and the task_id hierarchy
        conv = layer.conversation("conv_1")
        assert [(m.sender, m.receiver) for m in conv] == [
            ("user", "director"), ("director", "research_agent"), ("research_agent", "director"),
            ("director", "coding_agent"), ("coding_agent", "director"), ("director", "user")]
        assert all(m.task_id and m.task_id.startswith("task_1") for m in conv)
        # the result message ids in the plan point to real history entries
        assert layer.history.get(plan["subtasks"][0]["result_message_id"]) is not None
        json.dumps(plan)  # fully serialisable
    finally:
        await _stop(research, coding, director, user)


# ----------------------------------------------------------------- DAG execution
async def test_parallel_branches_and_join(layer: CommunicationLayer) -> None:
    order: list[str] = []

    def mk(role: str, latency: float = 0.0) -> MockLLMAdapter:
        def r(msgs: list[ChatMessage], system: str | None) -> str:
            order.append(role)
            return f"{role}-out"
        return MockLLMAdapter(responder=r, latency=latency)

    a = _worker(layer, "r", "researcher", mk("researcher", 0.05))
    b = _worker(layer, "c", "coder", mk("coder", 0.05))
    c = _worker(layer, "t", "tester", mk("tester"))
    plan_src = [
        SubTask(id="research", role="researcher", instruction="x"),
        SubTask(id="code", role="coder", instruction="y"),
        SubTask(id="test", role="tester", instruction="z", depends_on=["research", "code"]),
    ]
    director = _director(layer, StaticPlanner(plan_src))
    await _start(a, b, c, director)
    try:
        plan = await director.run("goal", task_id="t", conversation_id="cv")
        assert plan.status is PlanStatus.COMPLETED
        assert order[-1] == "tester" and set(order[:2]) == {"researcher", "coder"}
        assert plan.get("test").result == "tester-out"
        # the join task received both inputs
        tester_prompt = c.adapter.calls[0][-1].content  # type: ignore[attr-defined]
        assert "researcher-out" in tester_prompt and "coder-out" in tester_prompt
        assert plan.results() == {"research": "researcher-out", "code": "coder-out", "test": "tester-out"}
    finally:
        await _stop(a, b, c, director)


# ------------------------------------------------------------------ error handling
async def test_missing_role_fails_subtask_and_skips_dependents(layer: CommunicationLayer) -> None:
    coder = _worker(layer, "c", "coder", MockLLMAdapter(scripted=["code"]))
    director = _director(layer, SequentialPlanner(["researcher", "coder"], strict=True))
    await _start(coder, director)
    try:
        plan = await director.run("goal")
        assert plan.status is PlanStatus.FAILED  # nothing completed
        s1, s2 = plan.subtasks
        assert s1.status is SubTaskStatus.FAILED and "no online agent" in (s1.error or "")
        assert s2.status is SubTaskStatus.SKIPPED and s2.error == "dependency failed"
        assert coder.adapter.calls == []  # type: ignore[attr-defined]
        assert plan.final_result and "ERROR" in plan.final_result
    finally:
        await _stop(coder, director)


async def test_worker_error_is_retried_on_another_agent(layer: CommunicationLayer) -> None:
    broken = _worker(layer, "coder_1", "coder", MockLLMAdapter(fail_with=RuntimeError("gpu on fire")))
    healthy = _worker(layer, "coder_2", "coder", MockLLMAdapter(scripted=["works"]))
    director = _director(layer, SequentialPlanner(["coder"]), max_retries=1)
    await _start(broken, healthy, director)
    try:
        plan = await director.run("goal")
        st = plan.subtasks[0]
        assert plan.status is PlanStatus.COMPLETED
        assert st.attempts == 2 and st.assigned_to == "coder_2" and st.result == "works"
        events = [(e.event, e.detail) for e in plan.events if e.subtask_id == st.id]
        assert events[0][0] == "dispatched" and "coder_1" in events[0][1]
        assert events[1][0] == "retry" and "gpu on fire" in events[1][1]
        assert events[2][0] == "dispatched" and "coder_2" in events[2][1]
        assert events[3][0] == "completed"
    finally:
        await _stop(broken, healthy, director)


async def test_worker_timeout_exhausts_retries_partial_result(layer: CommunicationLayer) -> None:
    research = _worker(layer, "r", "researcher", MockLLMAdapter(scripted=["found"]))
    slow = _worker(layer, "c", "coder", MockLLMAdapter(latency=5))
    reviewer = _worker(layer, "v", "reviewer", MockLLMAdapter(scripted=["lgtm"]))
    src = [
        SubTask(id="a", role="researcher", instruction="x"),
        SubTask(id="b", role="coder", instruction="y", depends_on=["a"]),
        SubTask(id="c", role="reviewer", instruction="z", depends_on=["a"]),
    ]
    director = _director(layer, StaticPlanner(src), task_timeout=0.2, max_retries=1)
    await _start(research, slow, reviewer, director)
    try:
        plan = await director.run("goal")
        assert plan.status is PlanStatus.PARTIAL
        b = plan.get("b")
        assert b.status is SubTaskStatus.FAILED and b.attempts == 2 and "timeout" in (b.error or "")
        assert plan.get("a").status is SubTaskStatus.DONE and plan.get("c").status is SubTaskStatus.DONE
        assert "ERROR: timeout" in (plan.final_result or "")
    finally:
        await _stop(research, slow, reviewer, director)


async def test_task_rejected_by_worker_counts_as_failure(layer: CommunicationLayer) -> None:
    class Refuser(BaseAgent):
        async def handle(self, message: Message) -> None:
            if message.reply_required:
                await self.reply(message, "not my job", message_type=MessageType.TASK_REJECTED)

    refuser = Refuser(layer, AgentInfo(id="ref", name="ref", role="coder"))
    director = _director(layer, SequentialPlanner(["coder"]), max_retries=0)
    await _start(refuser, director)
    try:
        plan = await director.run("goal")
        st = plan.subtasks[0]
        assert st.status is SubTaskStatus.FAILED and "rejected: not my job" == st.error
    finally:
        await _stop(refuser, director)


async def test_planner_exception_produces_failed_plan_not_crash(layer: CommunicationLayer) -> None:
    class Boom:
        async def plan(self, goal: str, agents: Sequence[AgentInfo], *, context: dict[str, Any]) -> Plan:
            raise RuntimeError("llm quota exceeded")

        async def replan(self, plan: Plan, agents: Sequence[AgentInfo]) -> list[SubTask]:
            return []

        async def synthesize(self, plan: Plan) -> str:
            return ""

    director = _director(layer, Boom())
    user = EchoAgent(layer, "user")
    await _start(director, user)
    try:
        res = await user.ask("director", "goal", message_type=MessageType.TASK_REQUEST, timeout=5)
        assert res.message_type is MessageType.TASK_REJECTED
        assert "llm quota exceeded" in res.content
        assert res.metadata["plan"]["status"] == "failed"
        # director still alive
        res2 = await user.ask("director", "again", message_type=MessageType.TASK_REQUEST, timeout=5)
        assert res2.message_type is MessageType.TASK_REJECTED
    finally:
        await _stop(director, user)


async def test_invalid_plan_from_planner_is_rejected(layer: CommunicationLayer) -> None:
    cyclic = [SubTask(id="a", role="coder", instruction="i", depends_on=["b"]),
              SubTask(id="b", role="coder", instruction="i", depends_on=["a"])]
    coder = _worker(layer, "c", "coder", MockLLMAdapter())
    director = _director(layer, StaticPlanner(cyclic))
    await _start(coder, director)
    try:
        plan = await director.run("goal")
        assert plan.status is PlanStatus.FAILED and "cycle" in (plan.final_result or "")
        assert coder.adapter.calls == []  # type: ignore[attr-defined]
    finally:
        await _stop(coder, director)


async def test_director_rejects_non_task_messages(layer: CommunicationLayer) -> None:
    director = _director(layer, SequentialPlanner([]))
    user = EchoAgent(layer, "user")
    await _start(director, user)
    try:
        res = await user.ask("director", "review this", message_type=MessageType.REVIEW_REQUEST, timeout=2)
        assert res.message_type is MessageType.TASK_REJECTED
    finally:
        await _stop(director, user)


# ------------------------------------------------------------------- LLM planner
def _llm_planner_adapter(plan_json: dict[str, Any], replan_json: dict[str, Any] | None = None,
                         synthesis: str = "FINAL") -> MockLLMAdapter:
    """Mock LLM that answers plan / replan / synthesize calls based on the system prompt."""
    replans = [replan_json or {"subtasks": []}]

    def respond(msgs: list[ChatMessage], system: str | None) -> str:
        assert system is not None
        if "Decompose the user's goal" in system:
            return "```json\n" + json.dumps(plan_json) + "\n```"
        if "ADDITIONAL sub-tasks" in system:
            return json.dumps(replans.pop(0) if replans else {"subtasks": []})
        if "final deliverable" in system:
            return synthesis
        raise AssertionError(f"unexpected system prompt: {system[:40]}")

    return MockLLMAdapter(model="planner", responder=respond)


async def test_llm_planner_dynamic_decomposition_and_synthesis(layer: CommunicationLayer) -> None:
    plan_json = {
        "analysis": "Needs research then code.",
        "subtasks": [
            {"id": "s1", "role": "researcher", "instruction": "Research algorithms", "depends_on": []},
            {"id": "s2", "role": "coder", "instruction": "Implement it", "depends_on": ["s1"]},
            {"id": "s3", "role": "designer", "instruction": "not available", "depends_on": []},  # dropped
        ],
    }
    brain = _llm_planner_adapter(plan_json, synthesis="FINAL: rate limiter delivered")
    research = _worker(layer, "r", "researcher", MockLLMAdapter(scripted=["token bucket"]))
    coder = _worker(layer, "c", "coder", MockLLMAdapter(scripted=["class TB"]))
    director = _director(layer, LLMPlanner(brain, max_replans=0))
    user = EchoAgent(layer, "user")
    await _start(research, coder, director, user)
    try:
        res = await user.ask("director", "Build a rate limiter", message_type=MessageType.TASK_REQUEST, timeout=5)
        assert res.content == "FINAL: rate limiter delivered"
        plan = res.metadata["plan"]
        assert plan["analysis"] == "Needs research then code."
        assert [s["id"] for s in plan["subtasks"]] == ["s1", "s2"]  # unknown role dropped
        assert plan["status"] == "completed"
        # the planner was told which roles exist
        first_prompt = brain.calls[0][-1].content
        assert "role=researcher" in first_prompt and "role=coder" in first_prompt
        assert "GOAL:\nBuild a rate limiter" in first_prompt
        # the LLM-based planner never touched the communication layer directly
        assert all(m.sender in {"user", "director", "r", "c"} for m in layer.history.all())
    finally:
        await _stop(research, coder, director, user)


async def test_llm_planner_replan_adds_follow_up_tasks(layer: CommunicationLayer) -> None:
    plan_json = {"analysis": "a", "subtasks": [
        {"id": "s1", "role": "coder", "instruction": "write code", "depends_on": []}]}
    replan_json = {"subtasks": [
        {"id": "s2", "role": "reviewer", "instruction": "review the code", "depends_on": ["s1"]},
        {"id": "s1", "role": "coder", "instruction": "duplicate id must be ignored", "depends_on": []}]}
    brain = _llm_planner_adapter(plan_json, replan_json)
    coder = _worker(layer, "c", "coder", MockLLMAdapter(scripted=["the code"]))
    reviewer = _worker(layer, "v", "reviewer", MockLLMAdapter(scripted=["approved"]))
    director = _director(layer, LLMPlanner(brain, max_replans=1), max_rounds=3)
    await _start(coder, reviewer, director)
    try:
        plan = await director.run("goal")
        assert plan.status is PlanStatus.COMPLETED and plan.rounds == 2
        assert [s.id for s in plan.subtasks] == ["s1", "s2"]
        assert plan.get("s2").result == "approved"
        assert any(e.event == "replanned" for e in plan.events)
        review_prompt = reviewer.adapter.calls[0][-1].content  # type: ignore[attr-defined]
        assert "the code" in review_prompt
    finally:
        await _stop(coder, reviewer, director)


async def test_llm_planner_bad_json_yields_failed_plan(layer: CommunicationLayer) -> None:
    brain = MockLLMAdapter(scripted=["I cannot plan this, sorry."])
    coder = _worker(layer, "c", "coder", MockLLMAdapter())
    director = _director(layer, LLMPlanner(brain))
    await _start(coder, director)
    try:
        plan = await director.run("goal")
        assert plan.status is PlanStatus.FAILED
        assert "no JSON object" in (plan.final_result or "")
    finally:
        await _stop(coder, director)


async def test_synthesis_failure_falls_back_to_report(layer: CommunicationLayer) -> None:
    calls = {"n": 0}

    def respond(msgs: list[ChatMessage], system: str | None) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"analysis": "x", "subtasks": [
                {"id": "s1", "role": "coder", "instruction": "code", "depends_on": []}]})
        raise RuntimeError("synth down")

    coder = _worker(layer, "c", "coder", MockLLMAdapter(scripted=["CODE"]))
    director = _director(layer, LLMPlanner(MockLLMAdapter(responder=respond), max_replans=0))
    await _start(coder, director)
    try:
        plan = await director.run("goal")
        assert plan.status is PlanStatus.COMPLETED
        assert "CODE" in (plan.final_result or "")
        assert any(e.event == "synthesis_failed" for e in plan.events)
    finally:
        await _stop(coder, director)


async def test_multiple_concurrent_goals_are_tracked_separately(layer: CommunicationLayer) -> None:
    coder = _worker(layer, "c", "coder", MockLLMAdapter(latency=0.05))
    director = _director(layer, SequentialPlanner(["coder"]))
    await _start(coder, director)
    try:
        p1, p2 = await asyncio.gather(director.run("goal one", task_id="t1"), director.run("goal two", task_id="t2"))
        assert p1.plan_id != p2.plan_id and set(director.plans) == {p1.plan_id, p2.plan_id}
        assert p1.status is PlanStatus.COMPLETED and p2.status is PlanStatus.COMPLETED
        assert layer.task_history("t1:s1_coder") and layer.task_history("t2:s1_coder")
    finally:
        await _stop(coder, director)
