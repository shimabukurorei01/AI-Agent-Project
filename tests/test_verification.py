"""Question -> Plan -> Execution -> Critique -> Refinement cycle (offline, MockLLMAdapter only)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

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
    AgentVerifier,
    CompositeVerifier,
    Critique,
    DefaultRefinementPolicy,
    DirectorAgent,
    ExecutionRecord,
    Hypothesis,
    Issue,
    IssueCategory,
    LLMVerifier,
    Plan,
    PlanStatus,
    Question,
    ReasoningTrace,
    Refinement,
    RefinementAction,
    RuleBasedVerifier,
    SequentialPlanner,
    Severity,
    StaticPlanner,
    SubTask,
    SubTaskStatus,
    feedback_text,
    render_results,
)

# ----------------------------------------------------------------------- helpers


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


def _review_json(passed: bool, *issues: tuple[str, str, str]) -> str:
    return json.dumps({
        "passed": passed, "summary": "ok" if passed else "problems found", "confidence": 0.9,
        "issues": [{"category": c, "severity": s, "summary": t, "evidence": "quoted", "recommendation": "fix it"}
                   for c, s, t in issues],
    })


def _reviewer_agent(layer: CommunicationLayer, responder: Any, agent_id: str = "reviewer") -> LLMAgent:
    return _worker(layer, agent_id, "reviewer", MockLLMAdapter(model="review-model", responder=responder))


def _result_under_review(msgs: list[ChatMessage]) -> str:
    return msgs[-1].content.split("RESULT TO REVIEW", 1)[-1]


class ScriptedVerifier:
    """Test double: returns pre-scripted critiques in order (defaults to pass)."""

    verifier_id = "scripted"

    def __init__(self, *verdicts: Critique | Exception) -> None:
        self.verdicts = list(verdicts)
        self.calls: list[tuple[str, int, str]] = []

    async def verify(self, plan: Plan, subtask: SubTask, result: str) -> Critique:
        self.calls.append((subtask.id, subtask.attempts, result))
        v = self.verdicts.pop(0) if self.verdicts else Critique("scripted", True, subtask.attempts)
        if isinstance(v, Exception):
            raise v
        v.attempt = subtask.attempts
        return v


def _fail(*cats: IssueCategory, sev: Severity = Severity.HIGH) -> Critique:
    return Critique("scripted", False, 0, summary="rejected",
                    issues=[Issue(c, sev, f"{c.value} found", evidence="e", recommendation="r") for c in cats])


def _pass() -> Critique:
    return Critique("scripted", True, 0, summary="looks good")


# ============================================================ happy path
async def test_happy_path_question_plan_execution_critique_pass(layer: CommunicationLayer) -> None:
    coder = _worker(layer, "coder", "coder", MockLLMAdapter(scripted=["class RateLimiter: ..."]))
    reviewer = _reviewer_agent(layer, lambda m, s: _review_json(True))
    director = _director(layer, SequentialPlanner(["coder"]), verifier=AgentVerifier(layer, "director"))
    await _start(coder, reviewer, director)
    try:
        plan = await director.run("Build a rate limiter", task_id="t1", conversation_id="c1")
        assert plan.status is PlanStatus.COMPLETED
        st = plan.subtasks[0]
        r = st.reasoning
        # 1. Question
        assert isinstance(r.question, Question) and "rate limiter" in r.question.summary.lower()
        assert r.question.context["plan_id"] == plan.plan_id
        # 2. Hypothesis / Plan
        assert isinstance(r.hypothesis, Hypothesis) and r.hypothesis.role == "coder"
        assert "independent critique" in r.hypothesis.steps
        # 3. Execution
        assert len(r.executions) == 1
        ex = r.executions[0]
        assert ex.agent_id == "coder" and ex.outcome == "success" and ex.attempt == 1
        assert ex.request_message_id == st.request_message_id and ex.result_message_id == st.result_message_id
        assert "RateLimiter" in ex.result_summary
        # 4. Critique (pass) by someone other than the worker
        assert len(r.critiques) == 1 and r.critiques[0].passed and r.critiques[0].verifier == "reviewer"
        assert r.critiques[0].independent and r.critiques[0].attempt == 1
        assert st.verified_by == "reviewer" and st.refinements == 0
        # 5. No refinement needed -> final result
        assert r.refinements == [] and st.status is SubTaskStatus.DONE
        assert [e.event for e in plan.events] == [
            "planned", "dispatched", "completed", "critique", "verified", "finished"]
        assert "RateLimiter" in (plan.final_result or "")
    finally:
        await _stop(coder, reviewer, director)


async def test_no_verifier_keeps_legacy_behaviour(layer: CommunicationLayer) -> None:
    coder = _worker(layer, "coder", "coder", MockLLMAdapter(scripted=["x"]))
    director = _director(layer, SequentialPlanner(["coder"]))
    await _start(coder, director)
    try:
        plan = await director.run("g")
        st = plan.subtasks[0]
        assert plan.status is PlanStatus.COMPLETED and st.verified_by is None
        assert st.reasoning.critiques == [] and st.reasoning.refinements == []
        assert len(st.reasoning.executions) == 1
        assert st.reasoning.hypothesis is not None
        assert "independent critique" not in st.reasoning.hypothesis.steps
        assert [e.event for e in plan.events] == ["planned", "dispatched", "completed", "finished"]
    finally:
        await _stop(coder, director)


# ============================================================ refinement paths
async def test_refinement_cycle_fail_then_retry_same_then_pass(layer: CommunicationLayer) -> None:
    """Critique(fail) -> Refinement(retry_same with feedback) -> re-execution -> Critique(pass)."""
    prompts: list[str] = []

    def code(msgs: list[ChatMessage], system: str | None) -> str:
        prompts.append(msgs[-1].content)
        return "code + tests" if "Reviewer feedback" in msgs[-1].content else "code only"

    def review(msgs: list[ChatMessage], system: str | None) -> str:
        ok = "tests" in _result_under_review(msgs)
        return _review_json(True) if ok else _review_json(False, ("test_failure", "high", "no tests"))

    coder = _worker(layer, "coder", "coder", MockLLMAdapter(responder=code))
    reviewer = _reviewer_agent(layer, review)
    director = _director(layer, SequentialPlanner(["coder"]), verifier=AgentVerifier(layer, "director"),
                         max_refinements=2)
    await _start(coder, reviewer, director)
    try:
        plan = await director.run("Build it", task_id="t1", conversation_id="c1")
        st = plan.subtasks[0]
        assert plan.status is PlanStatus.COMPLETED and st.result == "code + tests"
        assert st.attempts == 2 and st.refinements == 1 and st.verified_by == "reviewer"
        r = st.reasoning
        assert [(c.passed, c.attempt) for c in r.critiques] == [(False, 1), (True, 2)]
        issue = r.critiques[0].issues[0]
        assert issue.category is IssueCategory.TEST_FAILURE and issue.severity is Severity.HIGH
        assert issue.evidence == "quoted" and issue.recommendation == "fix it"
        assert [(x.action, x.after_attempt) for x in r.refinements] == [(RefinementAction.RETRY_SAME, 1)]
        assert [e.outcome for e in r.executions] == ["success", "success"]
        # the worker received structured feedback (issue + recommendation), nothing else
        assert "Reviewer feedback" not in prompts[0]
        assert "no tests" in prompts[1] and "fix it" in prompts[1]
        assert [e.event for e in plan.events] == [
            "planned", "dispatched", "completed", "critique", "refinement",
            "dispatched", "completed", "critique", "verified", "finished"]
    finally:
        await _stop(coder, reviewer, director)


async def test_refinement_reassigns_to_another_agent(layer: CommunicationLayer) -> None:
    bad = _worker(layer, "coder_a", "coder", MockLLMAdapter(responder=lambda m, s: "garbage"))
    good = _worker(layer, "coder_b", "coder", MockLLMAdapter(responder=lambda m, s: "solid"))
    verifier = ScriptedVerifier(_fail(IssueCategory.HALLUCINATION), _fail(IssueCategory.HALLUCINATION), _pass())
    director = _director(layer, SequentialPlanner(["coder"]), verifier=verifier, max_refinements=3,
                         refinement_policy=DefaultRefinementPolicy(replan_after=0))
    await _start(bad, good, director)
    try:
        plan = await director.run("g")
        st = plan.subtasks[0]
        assert plan.status is PlanStatus.COMPLETED and st.result == "solid"
        assert [x.action for x in st.reasoning.refinements] == [
            RefinementAction.RETRY_SAME, RefinementAction.REASSIGN]
        assert st.reasoning.refinements[1].exclude_agents == ["coder_a"]
        assert st.reasoning.executors() == ["coder_a", "coder_b"]
        assert [e.agent_id for e in st.reasoning.executions] == ["coder_a", "coder_a", "coder_b"]
        assert st.assigned_to == "coder_b" and st.attempts == 3 and st.verified_by == "scripted"
    finally:
        await _stop(bad, good, director)


async def test_refinement_gather_info_from_other_role(layer: CommunicationLayer) -> None:
    seen: list[str] = []

    def code(msgs: list[ChatMessage], system: str | None) -> str:
        seen.append(msgs[-1].content)
        return "complete" if "Additional information" in msgs[-1].content else "incomplete"

    coder = _worker(layer, "coder", "coder", MockLLMAdapter(responder=code))
    researcher = _worker(layer, "researcher", "researcher", MockLLMAdapter(scripted=["THE MISSING SPEC"]))
    verifier = ScriptedVerifier(_fail(IssueCategory.MISSING_PRECONDITION), _pass())
    director = _director(layer, SequentialPlanner(["coder"]), verifier=verifier, max_refinements=2,
                         refinement_policy=DefaultRefinementPolicy(info_role="researcher"))
    await _start(coder, researcher, director)
    try:
        plan = await director.run("g", task_id="t1", conversation_id="c1")
        st = plan.subtasks[0]
        assert plan.status is PlanStatus.COMPLETED and st.result == "complete"
        assert st.reasoning.refinements[0].action is RefinementAction.GATHER_INFO
        assert st.reasoning.refinements[0].target_role == "researcher"
        assert "THE MISSING SPEC" in seen[1]
        assert any(e.event == "gathered_info" and "researcher" in e.detail for e in plan.events)
        info_msgs = layer.task_history("t1:s1_coder:info")
        assert [m.message_type for m in info_msgs] == [MessageType.QUESTION, MessageType.ANSWER]
        assert all(m.conversation_id == "c1" for m in info_msgs)
    finally:
        await _stop(coder, researcher, director)


async def test_gather_info_without_available_role_falls_back_to_feedback_only(
    layer: CommunicationLayer,
) -> None:
    coder = _worker(layer, "coder", "coder", MockLLMAdapter(responder=lambda m, s: "v"))
    verifier = ScriptedVerifier(_fail(IssueCategory.OMISSION), _pass())
    director = _director(layer, SequentialPlanner(["coder"]), verifier=verifier, max_refinements=2,
                         refinement_policy=DefaultRefinementPolicy(info_role="archivist"))  # nobody online
    await _start(coder, director)
    try:
        plan = await director.run("g")
        assert plan.status is PlanStatus.COMPLETED
        assert any(e.event == "gather_info_failed" for e in plan.events)
        assert plan.subtasks[0].attempts == 2  # still re-executed with the critique feedback
    finally:
        await _stop(coder, director)


async def test_minor_issues_are_accepted(layer: CommunicationLayer) -> None:
    coder = _worker(layer, "coder", "coder", MockLLMAdapter(scripted=["ok-ish"]))
    verifier = ScriptedVerifier(_fail(IssueCategory.OMISSION, sev=Severity.LOW))
    director = _director(layer, SequentialPlanner(["coder"]), verifier=verifier)
    await _start(coder, director)
    try:
        plan = await director.run("g")
        st = plan.subtasks[0]
        assert st.status is SubTaskStatus.DONE and st.attempts == 1
        assert st.reasoning.refinements[0].action is RefinementAction.ACCEPT
        assert st.verified_by == "scripted"
        assert any(e.event == "verified" and "minor" in e.detail for e in plan.events)
    finally:
        await _stop(coder, director)


# ============================================================ REPLAN path (focus)
class _ReplanOnce(SequentialPlanner):
    """Planner that, on the first replan() call, proposes a fallback sub-task and records what it saw."""

    def __init__(self, roles: list[str], fallback: SubTask | None = None) -> None:
        super().__init__(roles)
        self.fallback = fallback or SubTask(id="fallback", role="researcher", instruction="explain manually")
        self.replan_calls: list[str] = []

    async def replan(self, plan: Plan, agents: Sequence[AgentInfo]) -> list[SubTask]:
        self.replan_calls.append(render_results(plan))
        if len(self.replan_calls) > 1:
            return []
        return [SubTask(id=self.fallback.id, role=self.fallback.role, instruction=self.fallback.instruction,
                        depends_on=list(self.fallback.depends_on))]


async def test_replan_action_reaches_planner_replan_with_critique_context(layer: CommunicationLayer) -> None:
    """REPLAN: sub-task FAILED -> DAG round ends -> planner.replan() is called (existing path) and
    receives the structured critique so it can choose a different approach."""
    planner = _ReplanOnce(["coder"])
    coder = _worker(layer, "coder", "coder", MockLLMAdapter(responder=lambda m, s: "wrong"))
    researcher = _worker(layer, "researcher", "researcher", MockLLMAdapter(scripted=["explanation"]))
    verifier = ScriptedVerifier(_fail(IssueCategory.SPEC_MISMATCH), _pass())
    director = _director(layer, planner, verifier=verifier, max_refinements=2,
                         refinement_policy=DefaultRefinementPolicy(replan_after=1))
    await _start(coder, researcher, director)
    try:
        plan = await director.run("g", task_id="t1", conversation_id="c1")
        s1, fb = plan.subtasks
        # the rejected sub-task is FAILED with a traceable reason
        assert s1.status is SubTaskStatus.FAILED and "[replan]" in (s1.error or "")
        assert s1.reasoning.refinements[0].action is RefinementAction.REPLAN
        assert s1.attempts == 1 and s1.refinements == 1
        # the existing planner.replan() path was used exactly as before
        assert len(planner.replan_calls) == 2 and plan.rounds == 2
        assert any(e.event == "replanned" and "fallback" in e.detail for e in plan.events)
        # ... and the planner could see WHY (structured critique in the report)
        assert "Review by scripted (attempt 1): FAILED" in planner.replan_calls[0]
        assert "[high/spec_mismatch] spec_mismatch found -> r" in planner.replan_calls[0]
        # the follow-up task went through the full cycle too (incl. verification)
        assert fb.status is SubTaskStatus.DONE and fb.verified_by == "scripted"
        assert fb.task_id == "t1:fallback" and fb.reasoning.hypothesis is not None
        assert plan.status is PlanStatus.PARTIAL  # s1 failed, fallback done
        # event order: critique -> refinement(replan) -> failed -> replanned -> dispatched...
        events = [e.event for e in plan.events]
        i = events.index("refinement")
        assert events[i:i + 4] == ["refinement", "failed", "replanned", "dispatched"]
    finally:
        await _stop(coder, researcher, director)


async def test_replan_skips_dependants_before_replanning(layer: CommunicationLayer) -> None:
    """A REPLAN'd sub-task fails; its dependants are SKIPPED by the existing blocked() logic
    and the planner can add a replacement that depends on nothing broken."""
    planner = _ReplanOnce(["coder", "tester"],
                          fallback=SubTask(id="manual_check", role="researcher", instruction="check by hand"))
    coder = _worker(layer, "coder", "coder", MockLLMAdapter(responder=lambda m, s: "bad code"))
    tester = _worker(layer, "tester", "tester", MockLLMAdapter(responder=lambda m, s: "tests"))
    researcher = _worker(layer, "researcher", "researcher", MockLLMAdapter(scripted=["checked"]))
    verifier = ScriptedVerifier(_fail(IssueCategory.TEST_FAILURE), _pass())
    director = _director(layer, planner, verifier=verifier, max_refinements=1,
                         refinement_policy=DefaultRefinementPolicy(replan_after=1))
    await _start(coder, tester, researcher, director)
    try:
        plan = await director.run("g")
        by_id = {s.id: s for s in plan.subtasks}
        assert by_id["s1_coder"].status is SubTaskStatus.FAILED
        assert by_id["s2_tester"].status is SubTaskStatus.SKIPPED and by_id["s2_tester"].error == "dependency failed"
        assert by_id["s2_tester"].attempts == 0  # never dispatched
        assert by_id["manual_check"].status is SubTaskStatus.DONE
        assert tester.adapter.calls == []  # type: ignore[attr-defined]
        assert plan.status is PlanStatus.PARTIAL and plan.rounds == 2
    finally:
        await _stop(coder, tester, researcher, director)


async def test_replan_respects_max_rounds(layer: CommunicationLayer) -> None:
    planner = _ReplanOnce(["coder"])
    coder = _worker(layer, "coder", "coder", MockLLMAdapter(responder=lambda m, s: "wrong"))
    researcher = _worker(layer, "researcher", "researcher", MockLLMAdapter(scripted=["x"]))
    verifier = ScriptedVerifier(_fail(IssueCategory.SPEC_MISMATCH))
    director = _director(layer, planner, verifier=verifier, max_refinements=1, max_rounds=1,
                         refinement_policy=DefaultRefinementPolicy(replan_after=1))
    await _start(coder, researcher, director)
    try:
        plan = await director.run("g")
        assert plan.subtasks[0].reasoning.refinements[0].action is RefinementAction.REPLAN
        assert planner.replan_calls == []  # budget: no replanning round allowed
        assert plan.rounds == 1 and plan.status is PlanStatus.FAILED
        assert len(plan.subtasks) == 1
    finally:
        await _stop(coder, researcher, director)


async def test_replan_when_planner_adds_nothing_ends_as_failed(layer: CommunicationLayer) -> None:
    coder = _worker(layer, "coder", "coder", MockLLMAdapter(responder=lambda m, s: "wrong"))
    verifier = ScriptedVerifier(_fail(IssueCategory.SPEC_MISMATCH))
    director = _director(layer, SequentialPlanner(["coder"]), verifier=verifier, max_refinements=1,
                         refinement_policy=DefaultRefinementPolicy(replan_after=1))  # SequentialPlanner.replan -> []
    await _start(coder, director)
    try:
        plan = await director.run("g")
        st = plan.subtasks[0]
        assert st.status is SubTaskStatus.FAILED and st.reasoning.refinements[0].action is RefinementAction.REPLAN
        assert plan.status is PlanStatus.FAILED and plan.rounds == 1
        assert "FAILED" in (plan.final_result or "") and "spec_mismatch" in (plan.final_result or "")
    finally:
        await _stop(coder, director)


async def test_replan_rejects_invalid_followup_from_planner(layer: CommunicationLayer) -> None:
    planner = _ReplanOnce(["coder"], fallback=SubTask(id="bad", role="coder", instruction="x", depends_on=["ghost"]))
    coder = _worker(layer, "coder", "coder", MockLLMAdapter(responder=lambda m, s: "wrong"))
    verifier = ScriptedVerifier(_fail(IssueCategory.SPEC_MISMATCH))
    director = _director(layer, planner, verifier=verifier, max_refinements=1,
                         refinement_policy=DefaultRefinementPolicy(replan_after=1))
    await _start(coder, director)
    try:
        plan = await director.run("g")
        assert any(e.event == "replan_rejected" and "unknown dependency" in e.detail for e in plan.events)
        assert [s.id for s in plan.subtasks] == ["s1_coder"] and plan.rounds == 1
    finally:
        await _stop(coder, director)


# ============================================================ failure paths
async def test_critique_keeps_failing_until_budget_exhausted(layer: CommunicationLayer) -> None:
    coder = _worker(layer, "coder", "coder", MockLLMAdapter(responder=lambda m, s: "still wrong"))
    verifier = ScriptedVerifier(*[_fail(IssueCategory.SPEC_MISMATCH) for _ in range(10)])
    director = _director(layer, SequentialPlanner(["coder"]), verifier=verifier, max_refinements=2,
                         refinement_policy=DefaultRefinementPolicy(replan_after=0))
    await _start(coder, director)
    try:
        plan = await director.run("g")
        st = plan.subtasks[0]
        assert plan.status is PlanStatus.FAILED and st.status is SubTaskStatus.FAILED
        assert st.attempts == 3 and st.refinements == 3  # 2 retries + the final give_up decision
        assert [x.action for x in st.reasoning.refinements] == [
            RefinementAction.RETRY_SAME, RefinementAction.RETRY_SAME, RefinementAction.GIVE_UP]
        assert len(st.reasoning.critiques) == 3 and not any(c.passed for c in st.reasoning.critiques)
        assert "rejected by scripted" in (st.error or "") and "[give_up]" in (st.error or "")
        assert len(verifier.calls) == 3
    finally:
        await _stop(coder, director)


async def test_max_refinements_zero_means_single_shot_verification(layer: CommunicationLayer) -> None:
    coder = _worker(layer, "coder", "coder", MockLLMAdapter(responder=lambda m, s: "x"))
    verifier = ScriptedVerifier(_fail(IssueCategory.SPEC_MISMATCH))
    director = _director(layer, SequentialPlanner(["coder"]), verifier=verifier, max_refinements=0)
    await _start(coder, director)
    try:
        plan = await director.run("g")
        st = plan.subtasks[0]
        assert st.status is SubTaskStatus.FAILED and st.attempts == 1
        assert [x.action for x in st.reasoning.refinements] == [RefinementAction.GIVE_UP]
    finally:
        await _stop(coder, director)


async def test_worker_failure_before_critique_uses_existing_retry(layer: CommunicationLayer) -> None:
    coder = _worker(layer, "coder", "coder", MockLLMAdapter(fail_with=RuntimeError("boom")))
    verifier = ScriptedVerifier()
    director = _director(layer, SequentialPlanner(["coder"]), verifier=verifier, max_retries=1)
    await _start(coder, director)
    try:
        plan = await director.run("g")
        st = plan.subtasks[0]
        assert st.status is SubTaskStatus.FAILED and st.attempts == 2
        assert [e.outcome for e in st.reasoning.executions] == ["error", "error"]
        assert all("boom" in (e.error or "") for e in st.reasoning.executions)
        assert verifier.calls == []  # nothing to verify
        assert st.reasoning.critiques == [] and st.reasoning.refinements == []
    finally:
        await _stop(coder, director)


async def test_reexecution_failure_after_refinement(layer: CommunicationLayer) -> None:
    calls = {"n": 0}

    def flaky(msgs: list[ChatMessage], system: str | None) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return "first"
        raise RuntimeError("worker crashed on retry")

    coder = _worker(layer, "coder", "coder", MockLLMAdapter(responder=flaky))
    verifier = ScriptedVerifier(_fail(IssueCategory.SPEC_MISMATCH), _pass())
    director = _director(layer, SequentialPlanner(["coder"]), verifier=verifier, max_retries=0)
    await _start(coder, director)
    try:
        plan = await director.run("g")
        st = plan.subtasks[0]
        assert st.status is SubTaskStatus.FAILED
        assert [e.outcome for e in st.reasoning.executions] == ["success", "error"]
        assert "worker crashed" in (st.error or "")
        assert len(st.reasoning.critiques) == 1
        assert st.reasoning.refinements[0].action is RefinementAction.RETRY_SAME
        assert len(verifier.verdicts) == 1  # second verdict never consumed
    finally:
        await _stop(coder, director)


async def test_verifier_infrastructure_failure_is_recorded_not_passed(layer: CommunicationLayer) -> None:
    coder = _worker(layer, "coder", "coder", MockLLMAdapter(responder=lambda m, s: "result"))
    verifier = ScriptedVerifier(RuntimeError("review service down"), _pass())
    director = _director(layer, SequentialPlanner(["coder"]), verifier=verifier)
    await _start(coder, director)
    try:
        plan = await director.run("g")
        st = plan.subtasks[0]
        c0 = st.reasoning.critiques[0]
        assert not c0.passed and c0.confidence == 0.0 and "review service down" in c0.summary
        assert any(e.event == "verification_failed" for e in plan.events)
        # a verifier outage never silently passes; the policy retried and the 2nd review passed
        assert st.status is SubTaskStatus.DONE and st.verified_by == "scripted" and st.attempts == 2
    finally:
        await _stop(coder, director)


async def test_agent_verifier_no_reviewer_online(layer: CommunicationLayer) -> None:
    coder = _worker(layer, "coder", "coder", MockLLMAdapter(responder=lambda m, s: "r"))
    director = _director(layer, SequentialPlanner(["coder"]), verifier=AgentVerifier(layer, "director"),
                         max_refinements=1, refinement_policy=DefaultRefinementPolicy(replan_after=0))
    await _start(coder, director)
    try:
        plan = await director.run("g")
        st = plan.subtasks[0]
        assert st.status is SubTaskStatus.FAILED
        assert all("no independent reviewer" in c.summary for c in st.reasoning.critiques)
        assert st.reasoning.refinements[-1].action is RefinementAction.GIVE_UP
    finally:
        await _stop(coder, director)


async def test_agent_verifier_never_lets_worker_review_itself(layer: CommunicationLayer) -> None:
    # the only agent with the "reviewer" capability is the worker itself
    both = LLMAgent(layer, AgentInfo(id="solo", name="solo", role="coder", capabilities=["coder", "reviewer"]),
                    MockLLMAdapter(responder=lambda m, s: _review_json(True)))
    director = _director(layer, SequentialPlanner(["coder"]), verifier=AgentVerifier(layer, "director"),
                         max_refinements=0)
    await _start(both, director)
    try:
        plan = await director.run("g")
        st = plan.subtasks[0]
        assert st.status is SubTaskStatus.FAILED
        assert "no independent reviewer" in st.reasoning.critiques[0].summary
        assert st.reasoning.critiques[0].verifier == "agent:reviewer"
        # the worker was asked exactly once (its task), never for a review of itself
        assert len(both.adapter.calls) == 1  # type: ignore[attr-defined]
    finally:
        await _stop(both, director)


async def test_agent_verifier_prefers_reviewer_other_than_worker(layer: CommunicationLayer) -> None:
    worker_and_reviewer = LLMAgent(layer, AgentInfo(id="dual", name="d", role="coder", capabilities=["reviewer"]),
                                   MockLLMAdapter(responder=lambda m, s: "code"))
    pure_reviewer = _reviewer_agent(layer, lambda m, s: _review_json(True), agent_id="pure")
    director = _director(layer, SequentialPlanner(["coder"]), verifier=AgentVerifier(layer, "director"))
    await _start(worker_and_reviewer, pure_reviewer, director)
    try:
        plan = await director.run("g")
        st = plan.subtasks[0]
        assert st.assigned_to == "dual" and st.verified_by == "pure"
        assert st.reasoning.critiques[0].independent
    finally:
        await _stop(worker_and_reviewer, pure_reviewer, director)


async def test_reviewer_agent_returns_invalid_json(layer: CommunicationLayer) -> None:
    coder = _worker(layer, "coder", "coder", MockLLMAdapter(responder=lambda m, s: "r"))
    reviewer = _reviewer_agent(layer, lambda m, s: "LGTM!! (not json)")
    director = _director(layer, SequentialPlanner(["coder"]), verifier=AgentVerifier(layer, "director"),
                         max_refinements=0)
    await _start(coder, reviewer, director)
    try:
        plan = await director.run("g")
        c = plan.subtasks[0].reasoning.critiques[0]
        assert not c.passed and c.confidence == 0.0 and c.verifier == "reviewer"
        assert c.issues[0].summary == "reviewer returned invalid JSON" and "LGTM" in c.issues[0].evidence
        assert len(c.verifier_message_ids) == 2
    finally:
        await _stop(coder, reviewer, director)


async def test_reviewer_agent_failure_propagates_as_verification_failure(layer: CommunicationLayer) -> None:
    coder = _worker(layer, "coder", "coder", MockLLMAdapter(responder=lambda m, s: "r"))
    reviewer = _reviewer_agent(layer, None)
    reviewer.adapter = MockLLMAdapter(fail_with=RuntimeError("reviewer LLM down"))
    director = _director(layer, SequentialPlanner(["coder"]),
                         verifier=AgentVerifier(layer, "director", timeout=1), max_refinements=0)
    await _start(coder, reviewer, director)
    try:
        plan = await director.run("g")
        c = plan.subtasks[0].reasoning.critiques[0]
        assert not c.passed and "reviewer LLM down" in c.summary
        assert any(e.event == "verification_failed" for e in plan.events)
    finally:
        await _stop(coder, reviewer, director)


async def test_planner_failure_yields_failed_plan_without_execution(layer: CommunicationLayer) -> None:
    class Boom:
        async def plan(self, goal: str, agents: Sequence[AgentInfo], *, context: dict[str, Any]) -> Plan:
            raise RuntimeError("planner down")

        async def replan(self, plan: Plan, agents: Sequence[AgentInfo]) -> list[SubTask]:
            return []

        async def synthesize(self, plan: Plan) -> str:
            return ""

    verifier = ScriptedVerifier()
    director = _director(layer, Boom(), verifier=verifier)
    await _start(director)
    try:
        plan = await director.run("g")
        assert plan.status is PlanStatus.FAILED and "planner down" in (plan.final_result or "")
        assert plan.subtasks == [] and verifier.calls == []
    finally:
        await _stop(director)


@pytest.mark.parametrize(
    ("subtasks", "expected"),
    [
        ([SubTask(id="a", role="coder", instruction="i", depends_on=["nope"])], "unknown dependency"),
        ([SubTask(id="a", role="coder", instruction="i", depends_on=["b"]),
          SubTask(id="b", role="coder", instruction="i", depends_on=["a"])], "cycle"),
        ([SubTask(id="a", role="coder", instruction="i", depends_on=["a"])], "depends on itself"),
    ],
)
async def test_invalid_plans_are_rejected_before_execution(layer: CommunicationLayer, subtasks: list[SubTask],
                                                            expected: str) -> None:
    coder = _worker(layer, "coder", "coder", MockLLMAdapter())
    verifier = ScriptedVerifier()
    director = _director(layer, StaticPlanner(subtasks), verifier=verifier)
    await _start(coder, director)
    try:
        plan = await director.run("g")
        assert plan.status is PlanStatus.FAILED and expected in (plan.final_result or "")
        assert verifier.calls == [] and coder.adapter.calls == []  # type: ignore[attr-defined]
    finally:
        await _stop(coder, director)


async def test_empty_plan_is_invalid(layer: CommunicationLayer) -> None:
    director = _director(layer, StaticPlanner([]), verifier=ScriptedVerifier())
    await _start(director)
    try:
        plan = await director.run("g")
        assert plan.status is PlanStatus.FAILED and "no sub-tasks" in (plan.final_result or "")
    finally:
        await _stop(director)


# ============================================================ tracking
async def test_tracking_ids_worker_verifier_relationship_and_counts(layer: CommunicationLayer) -> None:
    def code(msgs: list[ChatMessage], system: str | None) -> str:
        return "v2" if "Reviewer feedback" in msgs[-1].content else "v1"

    def strict_review(msgs: list[ChatMessage], system: str | None) -> str:
        body = _result_under_review(msgs)
        return _review_json(True) if "v2" in body else _review_json(False, ("spec_mismatch", "high", "v1 is wrong"))

    coder = _worker(layer, "coder", "coder", MockLLMAdapter(responder=code))
    reviewer = _reviewer_agent(layer, strict_review, agent_id="qa_agent")
    director = _director(layer, SequentialPlanner(["coder"]), verifier=AgentVerifier(layer, "director"),
                         max_refinements=2)
    await _start(coder, reviewer, director)
    try:
        plan = await director.run("g", task_id="task_9", conversation_id="conv_9")
        st = plan.subtasks[0]
        # conversation_id / task_id maintained for worker AND reviewer traffic
        conv = layer.conversation("conv_9")
        assert conv and all(m.conversation_id == "conv_9" for m in conv)
        assert {m.task_id for m in conv} == {"task_9:s1_coder", "task_9:s1_coder:review"}
        assert [(m.sender, m.receiver, m.message_type.value) for m in conv] == [
            ("director", "coder", "task_request"), ("coder", "director", "task_result"),
            ("director", "qa_agent", "review_request"), ("qa_agent", "director", "review_result"),
            ("director", "coder", "task_request"), ("coder", "director", "task_result"),
            ("director", "qa_agent", "review_request"), ("qa_agent", "director", "review_result"),
        ]
        # message ids in the trace resolve to real history entries
        for ex in st.reasoning.executions:
            assert layer.history.get(ex.request_message_id or "") is not None
            assert layer.history.get(ex.result_message_id or "") is not None
        for c in st.reasoning.critiques:
            assert len(c.verifier_message_ids) == 2
            req, res = (layer.history.get(i) for i in c.verifier_message_ids)
            assert req is not None and req.message_type is MessageType.REVIEW_REQUEST
            assert res is not None and res.message_type is MessageType.REVIEW_RESULT
            assert req.metadata["worker"] == "coder" and req.metadata["reviewed_attempt"] == c.attempt
            assert req.metadata["subtask_id"] == "s1_coder" and req.metadata["plan_id"] == plan.plan_id
        # who produced / who verified / how many times
        assert st.reasoning.executors() == ["coder"] and st.reasoning.verifiers() == ["qa_agent"]
        assert st.assigned_to == "coder" and st.verified_by == "qa_agent"
        assert st.attempts == 2 and st.refinements == 1
        assert [c.attempt for c in st.reasoning.critiques] == [1, 2]
        assert st.to_dict()["refinements"] == 1
        # request metadata carries the refinement counter for downstream tracing
        second_req = layer.history.get(st.reasoning.executions[1].request_message_id or "")
        assert second_req is not None and second_req.metadata["refinement"] == 1
        first_req = layer.history.get(st.reasoning.executions[0].request_message_id or "")
        assert first_req is not None and first_req.metadata["refinement"] == 0
    finally:
        await _stop(coder, reviewer, director)


async def test_director_reply_carries_reasoning_in_metadata(layer: CommunicationLayer) -> None:
    coder = _worker(layer, "coder", "coder", MockLLMAdapter(scripted=["done"]))
    reviewer = _reviewer_agent(layer, lambda m, s: _review_json(True))
    director = _director(layer, SequentialPlanner(["coder"]), verifier=AgentVerifier(layer, "director"))
    user = EchoAgent(layer, "user")
    await _start(coder, reviewer, director, user)
    try:
        res = await user.ask("director", "g", message_type=MessageType.TASK_REQUEST, timeout=5)
        sub = res.metadata["plan"]["subtasks"][0]
        assert sub["verified_by"] == "reviewer" and sub["refinements"] == 0
        assert sub["reasoning"]["critiques"][0]["passed"] is True
        assert sub["reasoning"]["question"]["summary"] and sub["reasoning"]["hypothesis"]["role"] == "coder"
        json.dumps(res.metadata)  # fully serialisable over the wire
        assert Plan.from_dict(res.metadata["plan"]).subtasks[0].verified_by == "reviewer"
    finally:
        await _stop(coder, reviewer, director, user)


async def test_worker_prompt_receives_only_structured_feedback(layer: CommunicationLayer) -> None:
    """Feedback passed to workers is the structured critique, never raw reviewer output."""
    worker_prompts: list[str] = []

    def code(msgs: list[ChatMessage], system: str | None) -> str:
        worker_prompts.append(msgs[-1].content)
        return "attempt"

    coder = _worker(layer, "coder", "coder", MockLLMAdapter(responder=code))
    raw_reviewer_output = json.dumps({
        "passed": False, "summary": "bad", "confidence": 0.5,
        "issues": [{"category": "omission", "severity": "high", "summary": "missing X",
                    "evidence": "no X", "recommendation": "add X"}],
        "scratchpad": "PRIVATE-NOTES-MUST-NOT-LEAK",
    })
    reviewer = _reviewer_agent(layer, lambda m, s: raw_reviewer_output)
    director = _director(layer, SequentialPlanner(["coder"]), verifier=AgentVerifier(layer, "director"),
                         max_refinements=1, refinement_policy=DefaultRefinementPolicy(replan_after=0))
    await _start(coder, reviewer, director)
    try:
        plan = await director.run("g")
        assert len(worker_prompts) == 2
        assert "missing X" in worker_prompts[1] and "add X" in worker_prompts[1]
        assert "PRIVATE-NOTES" not in worker_prompts[1] and "scratchpad" not in worker_prompts[1]
        # ...and the trace does not keep unknown fields either
        assert "scratchpad" not in json.dumps(plan.to_dict())
    finally:
        await _stop(coder, reviewer, director)


# ============================================================ serialisation
def test_reasoning_trace_json_roundtrip() -> None:
    trace = ReasoningTrace(
        question=Question(summary="q", inputs=["a"], requirements=["r1"], context={"k": 1}),
        hypothesis=Hypothesis(approach="ap", role="coder", steps=["s1"], expected_outcome="eo"),
        executions=[ExecutionRecord(attempt=1, agent_id="w", request_message_id="m1", result_message_id="m2",
                                    inputs_used=["a"], outcome="success", result_summary="rs")],
        critiques=[Critique("rev", False, 1, summary="bad", confidence=0.4, verifier_message_ids=["m3", "m4"],
                            issues=[Issue(IssueCategory.LOGICAL_LEAP, Severity.CRITICAL, "s", "e", "r")])],
        refinements=[Refinement(RefinementAction.REASSIGN, "why", 1, instruction_delta="d",
                                target_role="coder", exclude_agents=["w"])],
    )
    data = json.loads(json.dumps(trace.to_dict()))
    assert data["critiques"][0]["issues"][0]["category"] == "logical_leap"
    assert data["critiques"][0]["issues"][0]["severity"] == "critical"
    assert data["refinements"][0]["action"] == "reassign"
    back = ReasoningTrace.from_dict(data)
    assert back == trace
    assert back.critiques[0].max_severity is Severity.CRITICAL
    assert back.refinement_count == 1 and not back.verified and back.verifiers() == ["rev"]


def test_subtask_and_plan_roundtrip_with_reasoning() -> None:
    st = SubTask(id="a", role="coder", instruction="i", verified_by="rev")
    st.reasoning.critiques.append(Critique("rev", True, 1))
    st.reasoning.refinements.append(Refinement(RefinementAction.RETRY_SAME, "r", 1))
    plan = Plan(goal="g", subtasks=[st], task_id="t", conversation_id="c")
    data = json.loads(json.dumps(plan.to_dict()))
    assert data["subtasks"][0]["refinements"] == 1 and data["subtasks"][0]["verified_by"] == "rev"
    back = Plan.from_dict(data)
    assert back.subtasks[0] == st and back.subtasks[0].refinements == 1
    assert back.task_id == "t" and back.conversation_id == "c"


def test_from_dict_tolerates_invalid_structured_data() -> None:
    c = Critique.from_dict({"verifier": "x", "passed": "yes", "attempt": "2",
                            "issues": [{"category": "made_up", "severity": "huge", "summary": "s"}, "junk", 42]})
    assert c.passed is True and c.attempt == 2 and len(c.issues) == 1
    assert c.issues[0].category is IssueCategory.OTHER and c.issues[0].severity is Severity.MEDIUM
    r = Refinement.from_dict({"action": "teleport", "reason": "?", "after_attempt": 1})
    assert r.action is RefinementAction.GIVE_UP
    assert ReasoningTrace.from_dict(None) == ReasoningTrace()
    legacy = SubTask.from_dict({"id": "a", "role": "r", "instruction": "i"})  # payload from before this feature
    assert legacy.reasoning == ReasoningTrace() and legacy.refinements == 0 and legacy.verified_by is None


def test_message_type_review_request_replies_with_review_result() -> None:
    m = Message(sender="d", receiver="r", content="x", message_type=MessageType.REVIEW_REQUEST)
    assert m.reply("ok").message_type is MessageType.REVIEW_RESULT


# ============================================================ verifier / policy units
async def test_rule_based_verifier_detects_obvious_failures() -> None:
    plan = Plan(goal="g")
    st = SubTask(id="a", role="coder", instruction="write code", attempts=1,
                 metadata={"must_include": ["def "], "must_not_include": ["TODO_MARK"]})
    v = RuleBasedVerifier()
    assert (await v.verify(plan, st, "")).issues[0].category is IssueCategory.WORKER_FAILURE
    c = await v.verify(plan, st, "Error: I cannot do that")
    assert not c.passed and c.issues[0].severity is Severity.HIGH
    c = await v.verify(plan, st, "some prose without code")
    assert not c.passed and any(i.category is IssueCategory.SPEC_MISMATCH and "def " in i.summary for i in c.issues)
    c = await v.verify(plan, st, "def f(): pass  TODO_MARK")
    assert c.passed and c.issues[0].severity is Severity.MEDIUM  # medium alone does not block
    c = await v.verify(plan, st, "def f(): return 1")
    assert c.passed and c.issues == [] and c.attempt == 1 and c.independent
    v2 = RuleBasedVerifier(custom_checks=[
        lambda s, r: Issue(IssueCategory.TEST_FAILURE, Severity.CRITICAL, "tests red")])
    assert not (await v2.verify(plan, st, "def f(): pass")).passed


async def test_llm_verifier_parses_structured_verdict_and_enforces_consistency() -> None:
    plan = Plan(goal="g")
    st = SubTask(id="a", role="coder", instruction="i", attempts=2, assigned_to="w")
    adapter = MockLLMAdapter(scripted=[
        "```json\n" + _review_json(True, ("hallucination", "critical", "invented API")) + "\n```",
        "not json at all",
        _review_json(True),
    ])
    v = LLMVerifier(adapter)
    assert v.verifier_id == "llm:mock:mock-1"
    c1 = await v.verify(plan, st, "res")
    assert c1.passed is False and c1.issues[0].category is IssueCategory.HALLUCINATION  # critical => fail
    c2 = await v.verify(plan, st, "res")
    assert not c2.passed and c2.confidence == 0.0
    c3 = await v.verify(plan, st, "res")
    assert c3.passed and c3.attempt == 2
    prompt = adapter.calls[0][-1].content
    assert "RESULT TO REVIEW (produced by w)" in prompt


async def test_composite_verifier_short_circuits_on_first_failure() -> None:
    plan = Plan(goal="g")
    st = SubTask(id="a", role="coder", instruction="i", attempts=1)
    second = ScriptedVerifier(_pass())
    comp = CompositeVerifier([RuleBasedVerifier(), second])
    c = await comp.verify(plan, st, "")
    assert not c.passed and c.verifier == "rule_based" and second.calls == []
    c = await comp.verify(plan, st, "fine result")
    assert c.passed and c.verifier == "rule_based + scripted" and len(second.calls) == 1


def test_default_policy_decisions_and_feedback_text() -> None:
    plan = Plan(goal="g")
    st = SubTask(id="a", role="coder", instruction="i", assigned_to="w1")
    pol = DefaultRefinementPolicy(info_role="researcher", replan_after=3)
    kw: dict[str, Any] = {"max_refinements": 2}
    assert pol.decide(plan, st, _fail(IssueCategory.OMISSION, sev=Severity.LOW), refinements_so_far=0,
                      **kw).action is RefinementAction.ACCEPT
    assert pol.decide(plan, st, _fail(IssueCategory.SPEC_MISMATCH), refinements_so_far=2,
                      **kw).action is RefinementAction.GIVE_UP
    d = pol.decide(plan, st, _fail(IssueCategory.MISSING_PRECONDITION), refinements_so_far=0, **kw)
    assert d.action is RefinementAction.GATHER_INFO and d.target_role == "researcher"
    d = pol.decide(plan, st, _fail(IssueCategory.WORKER_FAILURE), refinements_so_far=1, **kw)
    assert d.action is RefinementAction.REASSIGN and d.exclude_agents == ["w1"]
    d = pol.decide(plan, st, _fail(IssueCategory.SPEC_MISMATCH), refinements_so_far=0, **kw)
    assert d.action is RefinementAction.RETRY_SAME and "spec_mismatch found" in d.instruction_delta
    txt = feedback_text(_fail(IssueCategory.LOGICAL_LEAP))
    assert "[high/logical_leap]" in txt and "evidence: e" in txt and "-> r" in txt
    assert DefaultRefinementPolicy(replan_after=1).decide(
        plan, st, _fail(IssueCategory.SPEC_MISMATCH), refinements_so_far=0, **kw).action is RefinementAction.REPLAN


def test_render_results_includes_structured_critique_for_replanner() -> None:
    st = SubTask(id="a", role="coder", instruction="i", status=SubTaskStatus.FAILED, error="rejected", attempts=1)
    st.reasoning.critiques.append(_fail(IssueCategory.TEST_FAILURE))
    ok = SubTask(id="b", role="tester", instruction="j", status=SubTaskStatus.DONE, result="fine", attempts=1)
    ok.reasoning.critiques.append(_pass())
    text = render_results(Plan(goal="g", subtasks=[st, ok]))
    assert "Review by scripted (attempt 0): FAILED - rejected" in text
    assert "[high/test_failure] test_failure found -> r" in text
    assert "Review by scripted" in text.split("## [b]")[0]  # only the failed one has a review line
    assert "## [b]" in text and text.split("## [b]")[1].count("Review by") == 0  # clean pass: no noise
