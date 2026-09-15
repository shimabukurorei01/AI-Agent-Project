"""Critique (verification) and Refinement (decision) components.

Separation of duties
--------------------
* **Worker agents** execute and return results.  They never judge themselves.
* **Verifiers** produce a structured :class:`Critique` for one execution attempt.
  They are *separate* from the worker: a rule engine, an LLM adapter, or - the
  target architecture - an independent review/test agent reached through the
  normal communication layer (:class:`AgentVerifier`).
* **RefinementPolicy** decides what to do about a failed critique
  (retry / reassign / replan / gather info / accept / give up).
* **DirectorAgent** only orchestrates the cycle; it does not decide verdicts.

None of these components capture an LLM's private reasoning.  Verifiers are
asked for *conclusions with evidence* in a fixed JSON schema.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from ..errors import AgentCommError, MessageTimeoutError, RemoteAgentError
from ..layer import CommunicationLayer
from ..llm.base import ChatMessage, LLMAdapter
from ..models import AgentInfo, Message, MessageType
from .models import Plan, SubTask
from .planner import parse_json_object
from .reasoning import (
    Critique,
    Issue,
    IssueCategory,
    Refinement,
    RefinementAction,
    Severity,
)

_log = logging.getLogger("agentcomm.orchestration.verification")


# ------------------------------------------------------------------- protocols
class Verifier(Protocol):
    verifier_id: str

    async def verify(self, plan: Plan, subtask: SubTask, result: str) -> Critique:
        """Return a structured critique of *result* for *subtask*. Must not raise for
        ordinary review outcomes; raise only on verifier infrastructure failure."""
        ...


class RefinementPolicy(Protocol):
    def decide(self, plan: Plan, subtask: SubTask, critique: Critique,
               *, refinements_so_far: int, max_refinements: int) -> Refinement:
        """Choose the next action after a failed critique."""
        ...


# ----------------------------------------------------------- helper: checklist
VERIFICATION_CHECKLIST: tuple[tuple[IssueCategory, str], ...] = (
    (IssueCategory.HALLUCINATION, "claims not supported by the inputs or common knowledge"),
    (IssueCategory.FACT_SPECULATION_MIX, "speculation presented as fact"),
    (IssueCategory.LOGICAL_LEAP, "conclusions that do not follow from the reasoning given"),
    (IssueCategory.MISSING_PRECONDITION, "unstated assumptions or missing prerequisites"),
    (IssueCategory.SPEC_MISMATCH, "result does not satisfy the instruction / requirements"),
    (IssueCategory.RESULT_INCONSISTENCY, "internal contradictions or mismatch with prior step outputs"),
    (IssueCategory.WORKER_FAILURE, "error messages, refusals, placeholders instead of a result"),
    (IssueCategory.TEST_FAILURE, "failing tests or unverified behaviour claims"),
    (IssueCategory.OMISSION, "required parts of the task not addressed"),
)


def _blocking(issues: Sequence[Issue], threshold: Severity) -> bool:
    order = list(Severity)
    return any(order.index(i.severity) >= order.index(threshold) for i in issues)


# ------------------------------------------------------------ RuleBasedVerifier
_ERROR_PATTERNS = re.compile(
    r"^\s*(error|exception|traceback|i cannot|i can't|as an ai|unable to)\b|\bTODO\b|\bTBD\b|\bFIXME\b|\.\.\.\s*$",
    re.IGNORECASE | re.MULTILINE,
)


class RuleBasedVerifier:
    """Deterministic, cheap checks that catch obvious worker failures.

    * empty / trivially short result
    * error-like or placeholder output
    * required keywords (``subtask.metadata["must_include"]``) missing -> spec mismatch
    * forbidden phrases (``subtask.metadata["must_not_include"]``)
    * optional custom checks: callables ``(subtask, result) -> Issue | None``
    """

    verifier_id = "rule_based"

    def __init__(
        self,
        *,
        min_length: int = 1,
        fail_threshold: Severity = Severity.HIGH,
        custom_checks: Sequence[Callable[[SubTask, str], Issue | None]] = (),
    ) -> None:
        self.min_length = min_length
        self.fail_threshold = fail_threshold
        self.custom_checks = list(custom_checks)

    async def verify(self, plan: Plan, subtask: SubTask, result: str) -> Critique:
        issues: list[Issue] = []
        text = result.strip()
        if len(text) < self.min_length:
            issues.append(Issue(IssueCategory.WORKER_FAILURE, Severity.CRITICAL,
                                "empty result", evidence=repr(text[:40]),
                                recommendation="re-run the task"))
        elif _ERROR_PATTERNS.search(text):
            m = _ERROR_PATTERNS.search(text)
            issues.append(Issue(IssueCategory.WORKER_FAILURE, Severity.HIGH,
                                "result looks like an error or placeholder",
                                evidence=(m.group(0) if m else "")[:60],
                                recommendation="re-run with clearer instruction"))
        for kw in subtask.metadata.get("must_include", []) or []:
            if str(kw).lower() not in text.lower():
                issues.append(Issue(IssueCategory.SPEC_MISMATCH, Severity.HIGH,
                                    f"required element {kw!r} missing",
                                    evidence=f"{kw!r} not found in result",
                                    recommendation=f"include {kw!r}"))
        for kw in subtask.metadata.get("must_not_include", []) or []:
            if str(kw).lower() in text.lower():
                issues.append(Issue(IssueCategory.SPEC_MISMATCH, Severity.MEDIUM,
                                    f"forbidden element {kw!r} present", evidence=repr(kw),
                                    recommendation=f"remove {kw!r}"))
        for check in self.custom_checks:
            issue = check(subtask, result)
            if issue is not None:
                issues.append(issue)
        passed = not _blocking(issues, self.fail_threshold)
        return Critique(
            verifier=self.verifier_id, passed=passed, attempt=subtask.attempts,
            summary=("no blocking issues" if passed else f"{len(issues)} issue(s) found"),
            issues=issues, independent=True,
        )


# ------------------------------------------------------------------ LLM prompt
_REVIEW_SYSTEM = """You are an independent reviewer. You did NOT produce the result you are reviewing.
Check the RESULT against the TASK and INPUTS for each category:
{checklist}
Do not include your private reasoning. Report only conclusions with evidence.
Respond with JSON only:
{{
  "passed": true|false,
  "summary": "<one sentence verdict>",
  "confidence": 0.0-1.0,
  "issues": [
    {{"category": "<one of: {categories}>", "severity": "info|low|medium|high|critical",
      "summary": "<what is wrong>", "evidence": "<quote or observation>", "recommendation": "<fix>"}}
  ]
}}
Set "passed" to false if any issue is high or critical."""


def review_prompt(plan: Plan, subtask: SubTask, result: str) -> str:
    inputs = [plan.get(d) for d in subtask.depends_on if plan.has(d)]
    parts = [f"GOAL:\n{plan.goal}", f"\nTASK [{subtask.id}] (role {subtask.role}):\n{subtask.instruction}"]
    reqs = subtask.metadata.get("requirements") or subtask.metadata.get("must_include")
    if reqs:
        parts.append("\nREQUIREMENTS:\n- " + "\n- ".join(str(r) for r in reqs))
    if inputs:
        parts.append("\nINPUTS (results of previous steps):")
        parts.extend(f"[{d.id}] ({d.role}):\n{d.result}" for d in inputs)
    parts.append(f"\nRESULT TO REVIEW (produced by {subtask.assigned_to}):\n{result}")
    return "\n".join(parts)


def review_system_prompt() -> str:
    checklist = "\n".join(f"- {c.value}: {desc}" for c, desc in VERIFICATION_CHECKLIST)
    categories = ", ".join(c.value for c in IssueCategory)
    return _REVIEW_SYSTEM.format(checklist=checklist, categories=categories)


def critique_from_json(data: dict[str, Any], *, verifier: str, attempt: int,
                       message_ids: Sequence[str] = ()) -> Critique:
    """Build a :class:`Critique` from the reviewer JSON, tolerating sloppy fields."""
    critique = Critique.from_dict({
        "verifier": verifier, "passed": data.get("passed", False), "attempt": attempt,
        "summary": data.get("summary", ""), "issues": data.get("issues", []),
        "confidence": data.get("confidence", 1.0), "independent": True,
        "verifier_message_ids": list(message_ids),
    })
    # Consistency rule: blocking issues always fail, regardless of the flag.
    if _blocking(critique.issues, Severity.HIGH):
        critique.passed = False
    return critique


# ---------------------------------------------------------------- LLMVerifier
class LLMVerifier:
    """Direct LLM review through any :class:`LLMAdapter` (no extra agent needed).

    Independent from the worker as long as a different adapter/model or at least a
    different prompt context is used; ``independent`` is recorded accordingly.
    """

    def __init__(self, adapter: LLMAdapter, *, temperature: float = 0.0,
                 independent: bool = True) -> None:
        self.adapter = adapter
        self.temperature = temperature
        self.independent = independent
        self.verifier_id = f"llm:{adapter.provider}:{adapter.model}"

    async def verify(self, plan: Plan, subtask: SubTask, result: str) -> Critique:
        resp = await self.adapter.complete(
            [ChatMessage(role="user", content=review_prompt(plan, subtask, result))],
            system=review_system_prompt(), temperature=self.temperature,
        )
        try:
            data = parse_json_object(resp.content)
        except ValueError as exc:
            # Verifier produced garbage: this is a verifier failure, not a pass.
            return Critique(
                verifier=self.verifier_id, passed=False, attempt=subtask.attempts,
                summary=f"verifier output unparsable: {exc}", confidence=0.0,
                issues=[Issue(IssueCategory.OTHER, Severity.HIGH, "verifier returned invalid JSON",
                              evidence=resp.content[:120], recommendation="retry verification")],
                independent=self.independent,
            )
        critique = critique_from_json(data, verifier=self.verifier_id, attempt=subtask.attempts)
        critique.independent = self.independent
        return critique


# -------------------------------------------------------------- AgentVerifier
class AgentVerifier:
    """Verification by an **independent agent** via the communication layer.

    Sends a ``REVIEW_REQUEST`` (same ``conversation_id``, ``task_id`` =
    ``<subtask task_id>:review``) to an online agent with ``role``/capability
    ``reviewer_role`` and expects a ``REVIEW_RESULT`` whose content is the JSON
    schema above.  This is the hook for future *Test Agent -> Review Agent* chains.
    """

    def __init__(
        self,
        layer: CommunicationLayer,
        requester_id: str,
        *,
        reviewer_role: str = "reviewer",
        timeout: float = 60.0,
        exclude_worker: bool = True,
    ) -> None:
        self.layer = layer
        self.requester_id = requester_id
        self.reviewer_role = reviewer_role
        self.timeout = timeout
        self.exclude_worker = exclude_worker
        self.verifier_id = f"agent:{reviewer_role}"

    def pick_reviewer(self, subtask: SubTask) -> AgentInfo | None:
        reg = self.layer.registry
        candidates = reg.find_by_role(self.reviewer_role) or reg.find_by_capability(self.reviewer_role)
        for a in candidates:
            if a.id == self.requester_id:
                continue
            if self.exclude_worker and a.id == subtask.assigned_to:
                continue  # never let the worker review itself
            return a
        return None

    async def verify(self, plan: Plan, subtask: SubTask, result: str) -> Critique:
        reviewer = self.pick_reviewer(subtask)
        if reviewer is None:
            raise AgentCommError(f"no independent reviewer with role {self.reviewer_role!r} online")
        req = Message(
            sender=self.requester_id, receiver=reviewer.id,
            content=review_system_prompt() + "\n\n" + review_prompt(plan, subtask, result),
            message_type=MessageType.REVIEW_REQUEST, conversation_id=plan.conversation_id,
            task_id=f"{subtask.task_id}:review", reply_required=True,
            metadata={"plan_id": plan.plan_id, "subtask_id": subtask.id,
                      "reviewed_attempt": subtask.attempts, "worker": subtask.assigned_to},
        )
        try:
            reply = await self.layer.router.request(req, timeout=self.timeout)
        except (MessageTimeoutError, RemoteAgentError) as exc:
            raise AgentCommError(f"reviewer {reviewer.id} failed: {exc}") from exc
        ids = [req.message_id, reply.message_id]
        try:
            data = parse_json_object(reply.content)
        except ValueError as exc:
            return Critique(
                verifier=reviewer.id, passed=False, attempt=subtask.attempts,
                summary=f"reviewer output unparsable: {exc}", confidence=0.0,
                issues=[Issue(IssueCategory.OTHER, Severity.HIGH, "reviewer returned invalid JSON",
                              evidence=reply.content[:120], recommendation="retry verification")],
                verifier_message_ids=ids,
            )
        return critique_from_json(data, verifier=reviewer.id, attempt=subtask.attempts, message_ids=ids)


# ----------------------------------------------------------- CompositeVerifier
class CompositeVerifier:
    """Run verifiers in order; stop at the first failure (cheap rules before LLM/agents)."""

    verifier_id = "composite"

    def __init__(self, verifiers: Sequence[Verifier], *, require_all: bool = True) -> None:
        self.verifiers = list(verifiers)
        self.require_all = require_all

    async def verify(self, plan: Plan, subtask: SubTask, result: str) -> Critique:
        last: Critique | None = None
        for v in self.verifiers:
            last = await v.verify(plan, subtask, result)
            if not last.passed:
                return last
            if not self.require_all:
                return last
        if last is None:
            return Critique(verifier=self.verifier_id, passed=True, attempt=subtask.attempts,
                            summary="no verifiers configured", independent=False)
        last.verifier = " + ".join(v.verifier_id for v in self.verifiers)
        return last


# ----------------------------------------------------- DefaultRefinementPolicy
class DefaultRefinementPolicy:
    """Deterministic mapping from critique -> refinement action.

    * budget exhausted                  -> GIVE_UP
    * only low/info issues              -> ACCEPT
    * missing precondition/omission     -> GATHER_INFO (if ``info_role`` given) else RETRY_SAME
    * worker failure / hallucination    -> REASSIGN on 2nd strike, else RETRY_SAME
    * spec mismatch / inconsistency ... -> RETRY_SAME with feedback; REPLAN after
                                           ``replan_after`` failed refinements
    """

    def __init__(self, *, info_role: str | None = None, replan_after: int = 2,
                 accept_below: Severity = Severity.MEDIUM) -> None:
        self.info_role = info_role
        self.replan_after = replan_after
        self.accept_below = accept_below

    def decide(self, plan: Plan, subtask: SubTask, critique: Critique,
               *, refinements_so_far: int, max_refinements: int) -> Refinement:
        n = refinements_so_far
        if not _blocking(critique.issues, self.accept_below) and critique.issues:
            return Refinement(RefinementAction.ACCEPT, "only minor issues", critique.attempt)
        if n >= max_refinements:
            return Refinement(RefinementAction.GIVE_UP, f"refinement budget ({max_refinements}) exhausted",
                              critique.attempt)
        cats = {i.category for i in critique.issues}
        delta = feedback_text(critique)
        if cats & {IssueCategory.MISSING_PRECONDITION, IssueCategory.OMISSION} and self.info_role:
            return Refinement(RefinementAction.GATHER_INFO, "missing information", critique.attempt,
                              instruction_delta=delta, target_role=self.info_role)
        if n + 1 >= self.replan_after and self.replan_after > 0:
            return Refinement(RefinementAction.REPLAN, "repeated failures; ask planner for a new approach",
                              critique.attempt, instruction_delta=delta)
        if cats & {IssueCategory.WORKER_FAILURE, IssueCategory.HALLUCINATION} and n >= 1:
            return Refinement(RefinementAction.REASSIGN, "worker unreliable; try another agent",
                              critique.attempt, instruction_delta=delta,
                              exclude_agents=[subtask.assigned_to] if subtask.assigned_to else [])
        return Refinement(RefinementAction.RETRY_SAME, "retry with reviewer feedback", critique.attempt,
                          instruction_delta=delta)


def feedback_text(critique: Critique) -> str:
    """Human-readable feedback appended to the worker's next instruction."""
    lines = [f"--- Reviewer feedback ({critique.verifier}) on attempt {critique.attempt} ---",
             critique.summary or "The previous result was rejected."]
    for i in critique.issues:
        line = f"* [{i.severity.value}/{i.category.value}] {i.summary}"
        if i.evidence:
            line += f" (evidence: {i.evidence})"
        if i.recommendation:
            line += f" -> {i.recommendation}"
        lines.append(line)
    lines.append("Address every point above and return the corrected result.")
    return "\n".join(lines)


__all__ = [
    "VERIFICATION_CHECKLIST",
    "AgentVerifier",
    "CompositeVerifier",
    "DefaultRefinementPolicy",
    "LLMVerifier",
    "RefinementPolicy",
    "RuleBasedVerifier",
    "Verifier",
    "critique_from_json",
    "feedback_text",
    "review_prompt",
    "review_system_prompt",
]
