"""DirectorAgent: analyse a goal, decompose it, delegate, track, re-plan, integrate.

The director is a normal :class:`BaseAgent`: it only uses the public
CommunicationLayer API (``delegate`` / ``reply``), so the communication layer is
untouched and any agent with a matching *role* or *capability* can be a worker.

Execution model (per incoming TASK_REQUEST):

    plan = planner.plan(goal)                # 1-3  analyse + decompose
    loop:
        ready = plan.ready()                 # dependency resolution (DAG)
        for each ready sub-task (in parallel):   # _solve_subtask
            Question / Hypothesis recorded   # auditable summary, not chain-of-thought
            Execution  (_run_subtask: retry / fail-over)       # 4-6
            Critique   (independent Verifier, if configured)
            Refinement (RefinementPolicy: retry_same / reassign / replan /
                        gather_info / accept / give_up; max_refinements)
        mark blocked sub-tasks SKIPPED
        if plan finished:
            extra = planner.replan(plan)     # 7    follow-up tasks
            if extra: add & continue
            break
    final = planner.synthesize(plan)         # 8-9  integrate + final result
    reply TASK_RESULT with plan.to_dict() in metadata
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from typing import Any

from ..agent import BaseAgent
from ..errors import AgentCommError, MessageTimeoutError, RemoteAgentError
from ..layer import CommunicationLayer
from ..models import AgentInfo, Message, MessageType, new_id, utc_now
from .models import Plan, PlanStatus, SubTask, SubTaskStatus
from .planner import Planner
from .reasoning import (
    Critique,
    ExecutionRecord,
    Hypothesis,
    Issue,
    IssueCategory,
    Question,
    Refinement,
    RefinementAction,
    Severity,
    summarize,
)
from .verification import DefaultRefinementPolicy, RefinementPolicy, Verifier

_log = logging.getLogger("agentcomm.orchestration.director")

WorkerSelector = Callable[[SubTask, Sequence[AgentInfo]], AgentInfo | None]


def default_worker_selector(subtask: SubTask, candidates: Sequence[AgentInfo]) -> AgentInfo | None:
    """Pick the first candidate whose *role* matches, else first with the capability."""
    for a in candidates:
        if a.role == subtask.role:
            return a
    for a in candidates:
        if subtask.role in a.capabilities:
            return a
    return None


class DirectorAgent(BaseAgent):
    def __init__(
        self,
        layer: CommunicationLayer,
        info: AgentInfo,
        planner: Planner,
        *,
        task_timeout: float = 120.0,
        max_retries: int = 1,
        max_rounds: int = 5,
        worker_selector: WorkerSelector = default_worker_selector,
        verifier: Verifier | None = None,
        refinement_policy: RefinementPolicy | None = None,
        max_refinements: int = 2,
    ) -> None:
        super().__init__(layer, info)
        self.planner = planner
        self.task_timeout = task_timeout
        self.max_retries = max_retries  # transport-level failures per execution
        self.max_rounds = max_rounds  # planning rounds (initial + replans)
        self.select_worker = worker_selector
        # Critique / Refinement cycle. ``verifier=None`` keeps the original
        # execute-only behaviour (results are accepted without review).
        self.verifier = verifier
        self.refinement_policy: RefinementPolicy = refinement_policy or DefaultRefinementPolicy()
        self.max_refinements = max_refinements  # critique-driven refinements per sub-task
        self.plans: dict[str, Plan] = {}  # plan_id -> Plan (execution tracking)

    # ------------------------------------------------------------- messaging
    async def handle(self, message: Message) -> None:
        if message.message_type == MessageType.ERROR:
            self.log.warning("error from %s: %s", message.sender, message.content)
            return
        if not message.reply_required:
            return
        if message.message_type not in (MessageType.TASK_REQUEST, MessageType.QUESTION, MessageType.CHAT):
            await self.reply(message, f"{self.id} only accepts task_request/question",
                             message_type=MessageType.TASK_REJECTED)
            return

        context = message.metadata.get("context")
        plan = await self.run(
            message.content,
            conversation_id=message.conversation_id,
            task_id=message.task_id,
            context=context if isinstance(context, dict) else {},
        )
        await self.reply(
            message,
            plan.final_result or "",
            message_type=MessageType.TASK_RESULT if plan.status is not PlanStatus.FAILED
            else MessageType.TASK_REJECTED,
            metadata={"plan": plan.to_dict()},
        )

    # ------------------------------------------------------------ orchestration
    def workers(self) -> list[AgentInfo]:
        return [a for a in self.layer.registry.list(online_only=True) if a.id != self.id]

    async def run(
        self,
        goal: str,
        *,
        conversation_id: str | None = None,
        task_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> Plan:
        """Execute the full director workflow for *goal* and return the finished plan."""
        # ---- 1-3: analyse + decompose
        try:
            plan = await self.planner.plan(goal, self.workers(), context=context or {})
        except Exception as exc:
            self.log.exception("planning failed")
            plan = Plan(goal=goal, analysis=f"planning failed: {exc}")
            plan.status = PlanStatus.FAILED
            plan.final_result = f"Planning failed: {exc}"
            plan.finished_at = utc_now()
            plan.log("failed", detail=str(exc))
            self.plans[plan.plan_id] = plan
            return plan

        if conversation_id:
            plan.conversation_id = conversation_id
        if task_id:
            plan.task_id = task_id
        plan.rounds = 1
        self.plans[plan.plan_id] = plan

        problems = plan.validate()
        if problems:
            plan.status = PlanStatus.FAILED
            plan.final_result = "Invalid plan: " + "; ".join(problems)
            plan.finished_at = utc_now()
            plan.log("failed", detail=plan.final_result)
            return plan
        if not plan.subtasks:
            plan.status = PlanStatus.FAILED
            plan.final_result = "Planner produced no sub-tasks."
            plan.finished_at = utc_now()
            plan.log("failed", detail=plan.final_result)
            return plan

        plan.log("planned", detail=f"{len(plan.subtasks)} sub-tasks: "
                 + ", ".join(f"{s.id}({s.role})" for s in plan.subtasks))
        plan.status = PlanStatus.RUNNING
        self.log.info("plan %s: %d sub-tasks for goal %r", plan.plan_id, len(plan.subtasks), goal[:60])

        # ---- 4-7: execute DAG, re-plan
        await self._execute(plan)

        # ---- 8-9: integrate
        try:
            plan.final_result = await self.planner.synthesize(plan)
        except Exception as exc:
            self.log.exception("synthesis failed; falling back to raw report")
            from .planner import render_results

            plan.final_result = render_results(plan)
            plan.log("synthesis_failed", detail=str(exc))

        if not plan.done():
            plan.status = PlanStatus.FAILED
        elif plan.failed():
            plan.status = PlanStatus.PARTIAL
        else:
            plan.status = PlanStatus.COMPLETED
        plan.finished_at = utc_now()
        plan.log("finished", detail=plan.status.value)
        self.log.info("plan %s finished: %s", plan.plan_id, plan.status.value)
        return plan

    async def _execute(self, plan: Plan) -> None:
        while True:
            ready = plan.ready()
            if ready:
                await asyncio.gather(*(self._solve_subtask(plan, st) for st in ready))
                for st in plan.blocked():
                    st.status = SubTaskStatus.SKIPPED
                    st.error = "dependency failed"
                    st.finished_at = utc_now()
                    plan.log("skipped", st.id, "dependency failed")
                continue

            if not plan.is_finished():  # defensive: should not happen with a valid DAG
                for st in plan.subtasks:
                    if st.status in (SubTaskStatus.PENDING, SubTaskStatus.READY):
                        st.status = SubTaskStatus.SKIPPED
                        st.error = "unresolvable dependencies"
                        plan.log("skipped", st.id, st.error)
                break

            # ---- 7: follow-up tasks
            if plan.rounds >= self.max_rounds:
                break
            try:
                extra = await self.planner.replan(plan, self.workers())
            except Exception as exc:  # noqa: BLE001
                self.log.warning("replan failed: %s", exc)
                plan.log("replan_failed", detail=str(exc))
                break
            extra = [st for st in extra if not plan.has(st.id)]
            if not extra:
                break
            for st in extra:
                plan.add(st)
            problems = plan.validate()
            if problems:
                for st in extra:
                    plan.subtasks.remove(st)
                plan.log("replan_rejected", detail="; ".join(problems))
                break
            plan.rounds += 1
            plan.log("replanned", detail=f"round {plan.rounds}: +{len(extra)} sub-tasks "
                     + ", ".join(f"{s.id}({s.role})" for s in extra))
            self.log.info("plan %s: re-planned, +%d sub-tasks", plan.plan_id, len(extra))

    # ---------------------------------------------------------------- one task
    def build_instruction(self, plan: Plan, st: SubTask, *, extra: str = "") -> str:
        """Instruction sent to the worker: its task plus results of its dependencies."""
        parts = [st.instruction]
        deps = [plan.get(d) for d in st.depends_on if plan.has(d)]
        if deps:
            parts.append("\n--- Inputs from previous steps ---")
            for d in deps:
                parts.append(f"[{d.id}] ({d.role} / {d.assigned_to}):\n{d.result}")
        if extra:
            parts.append("\n" + extra)
        return "\n".join(parts)

    # ------------------------------------------------ Question -> ... -> Refinement
    def _record_question_and_hypothesis(self, plan: Plan, st: SubTask) -> None:
        """Stages 1-2: auditable summary of *what* is asked and *how* it will be approached."""
        reqs = [str(r) for r in (st.metadata.get("requirements") or st.metadata.get("must_include") or [])]
        st.reasoning.question = Question(
            summary=summarize(st.instruction, 300), inputs=list(st.depends_on), requirements=reqs,
            context={"goal": summarize(plan.goal, 200), "plan_id": plan.plan_id},
        )
        st.reasoning.hypothesis = Hypothesis(
            approach=f"delegate to an online agent with role/capability {st.role!r}"
                     + (f" using results of {', '.join(st.depends_on)}" if st.depends_on else ""),
            role=st.role,
            steps=["select worker", "send task_request", "await task_result"]
                  + (["independent critique", "refine if needed"] if self.verifier else []),
            expected_outcome=f"a result satisfying: {summarize(st.instruction, 120)}",
        )

    async def _solve_subtask(self, plan: Plan, st: SubTask) -> None:
        """Full cycle for one sub-task: Question -> Plan -> Execution -> Critique -> Refinement."""
        self._record_question_and_hypothesis(plan, st)
        extra_instruction = ""
        exclude: set[str] = set()

        while True:
            # ---- 3. Execution (existing retry / fail-over engine)
            await self._run_subtask(plan, st, extra=extra_instruction, exclude=exclude)
            if st.status is not SubTaskStatus.DONE or self.verifier is None:
                return  # transport-level failure already recorded, or no verification requested

            # ---- 4. Critique by an independent verifier
            critique = await self._critique(plan, st)
            if critique.passed:
                st.verified_by = critique.verifier
                plan.log("verified", st.id, f"attempt {st.attempts} passed by {critique.verifier}")
                return

            # ---- 5. Refinement decision
            decision = self.refinement_policy.decide(
                plan, st, critique,
                refinements_so_far=st.reasoning.refinement_count, max_refinements=self.max_refinements,
            )
            st.reasoning.refinements.append(decision)
            plan.log("refinement", st.id, f"{decision.action.value}: {decision.reason}")
            self.log.info("sub-task %s critique failed -> %s", st.id, decision.action.value)

            if decision.action is RefinementAction.ACCEPT:
                st.verified_by = critique.verifier
                plan.log("verified", st.id, f"accepted with minor issues by {critique.verifier}")
                return
            if decision.action is RefinementAction.GIVE_UP:
                self._fail_after_critique(plan, st, critique, decision)
                return
            if decision.action is RefinementAction.REPLAN:
                # Mark failed so dependants are skipped and the planner's replan()
                # (existing mechanism) can propose a different approach.
                self._fail_after_critique(plan, st, critique, decision)
                return
            if decision.action is RefinementAction.GATHER_INFO and decision.target_role:
                info = await self._gather_info(plan, st, decision)
                extra_instruction = (decision.instruction_delta + "\n\n--- Additional information ---\n" + info
                                     if info else decision.instruction_delta)
            else:
                extra_instruction = decision.instruction_delta
            if decision.action is RefinementAction.REASSIGN:
                exclude.update(decision.exclude_agents)

            # re-execute: reset per-attempt state but keep the audit trail
            st.status = SubTaskStatus.READY
            st.result = None
            st.error = None
            st.finished_at = None

    async def _critique(self, plan: Plan, st: SubTask) -> Critique:
        assert self.verifier is not None
        try:
            critique = await self.verifier.verify(plan, st, st.result or "")
        except Exception as exc:  # noqa: BLE001 - verifier infrastructure failure
            self.log.warning("verifier failed for %s: %s", st.id, exc)
            critique = Critique(
                verifier=getattr(self.verifier, "verifier_id", type(self.verifier).__name__),
                passed=False, attempt=st.attempts, confidence=0.0,
                summary=f"verification failed: {exc}",
                issues=[Issue(IssueCategory.OTHER, Severity.HIGH, "verifier unavailable",
                              evidence=str(exc)[:120], recommendation="retry verification later")],
            )
            plan.log("verification_failed", st.id, str(exc)[:120])
        st.reasoning.critiques.append(critique)
        plan.log("critique", st.id,
                 f"{'pass' if critique.passed else 'fail'} by {critique.verifier} "
                 f"({len(critique.issues)} issue(s))")
        return critique

    def _fail_after_critique(self, plan: Plan, st: SubTask, critique: Critique, decision: Refinement) -> None:
        st.status = SubTaskStatus.FAILED
        st.error = f"rejected by {critique.verifier}: {critique.summary or 'critique failed'} [{decision.action.value}]"
        st.finished_at = utc_now()
        plan.log("failed", st.id, st.error)

    async def _gather_info(self, plan: Plan, st: SubTask, decision: Refinement) -> str:
        """GATHER_INFO: ask another role for the missing information via the normal layer."""
        role = decision.target_role or ""
        probe = SubTask(id=f"{st.id}:info{st.reasoning.refinement_count}", role=role, instruction="")
        worker = self.select_worker(probe, [a for a in self.workers() if a.id != st.assigned_to])
        if worker is None:
            plan.log("gather_info_failed", st.id, f"no agent with role {role!r}")
            return ""
        req = Message(
            sender=self.id, receiver=worker.id,
            content=f"Provide the information needed to fix this task.\nTASK: {st.instruction}\n"
                    f"{decision.instruction_delta}",
            message_type=MessageType.QUESTION, conversation_id=plan.conversation_id,
            task_id=f"{st.task_id}:info", reply_required=True,
            metadata={"plan_id": plan.plan_id, "subtask_id": st.id},
        )
        try:
            reply = await self.layer.router.request(req, timeout=self.task_timeout)
        except AgentCommError as exc:
            plan.log("gather_info_failed", st.id, str(exc)[:120])
            return ""
        plan.log("gathered_info", st.id, f"from {worker.id} ({reply.message_id})")
        return reply.content

    # ------------------------------------------------------------- 3. Execution
    async def _run_subtask(self, plan: Plan, st: SubTask, *, extra: str = "",
                           exclude: set[str] | None = None) -> None:
        """Execute once with the existing retry / fail-over rules; record ExecutionRecords."""
        st.task_id = f"{plan.task_id}:{st.id}"
        st.started_at = st.started_at or utc_now()
        instruction = self.build_instruction(plan, st, extra=extra)
        tried: set[str] = set(exclude or ())
        base_attempt = st.attempts

        while st.attempts - base_attempt <= self.max_retries:
            # Prefer a worker we have not tried yet (fail-over); if there is no
            # alternative, retry on the same worker (the failure may be transient).
            available = self.workers()
            worker = self.select_worker(st, [a for a in available if a.id not in tried])
            if worker is None and tried:
                worker = self.select_worker(st, available)
            if worker is None:
                st.status = SubTaskStatus.FAILED
                st.error = (f"no online agent for role/capability {st.role!r}"
                            + (f" (tried: {', '.join(sorted(tried))})" if tried else ""))
                st.finished_at = utc_now()
                plan.log("failed", st.id, st.error)
                self.log.warning("sub-task %s failed: %s", st.id, st.error)
                return

            st.attempts += 1
            st.assigned_to = worker.id
            st.status = SubTaskStatus.RUNNING
            tried.add(worker.id)
            plan.log("dispatched", st.id, f"attempt {st.attempts} -> {worker.id}")

            req = Message(
                sender=self.id, receiver=worker.id, content=instruction,
                message_type=MessageType.TASK_REQUEST, conversation_id=plan.conversation_id,
                task_id=st.task_id, reply_required=True,
                metadata={"plan_id": plan.plan_id, "subtask_id": st.id, "attempt": st.attempts,
                          "refinement": st.reasoning.refinement_count},
            )
            st.request_message_id = req.message_id
            record = ExecutionRecord(attempt=st.attempts, agent_id=worker.id,
                                     request_message_id=req.message_id, inputs_used=list(st.depends_on))
            st.reasoning.executions.append(record)
            outcome = "error"
            try:
                result = await self.layer.router.request(req, timeout=self.task_timeout)
            except MessageTimeoutError as exc:
                error = f"timeout after {exc.timeout:.1f}s"
                outcome = "timeout"
            except RemoteAgentError as exc:
                error = f"worker error: {exc.content}"
            except AgentCommError as exc:
                error = f"{type(exc).__name__}: {exc}"
            else:
                record.result_message_id = result.message_id
                if result.message_type is MessageType.TASK_REJECTED:
                    error = f"rejected: {result.content}"
                    outcome = "rejected"
                else:
                    st.status = SubTaskStatus.DONE
                    st.result = result.content
                    st.result_message_id = result.message_id
                    st.finished_at = utc_now()
                    record.outcome = "success"
                    record.result_summary = summarize(result.content)
                    record.finished_at = st.finished_at
                    plan.log("completed", st.id, f"by {worker.id}")
                    self.log.info("sub-task %s done by %s", st.id, worker.id)
                    return

            record.outcome = outcome
            record.error = error
            record.finished_at = utc_now()
            st.error = error
            more = st.attempts - base_attempt <= self.max_retries
            plan.log("retry" if more else "failed", st.id,
                     f"attempt {st.attempts} on {worker.id}: {error}")
            self.log.warning("sub-task %s attempt %d on %s failed: %s", st.id, st.attempts, worker.id, error)

        st.status = SubTaskStatus.FAILED
        st.finished_at = utc_now()
        if not plan.events or plan.events[-1].event != "failed":
            plan.log("failed", st.id, st.error or "unknown error")


__all__ = ["DirectorAgent", "WorkerSelector", "default_worker_selector", "new_id"]
