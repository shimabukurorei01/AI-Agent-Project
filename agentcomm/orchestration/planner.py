"""Planners: *how* a goal is analysed, decomposed and synthesised.

The :class:`DirectorAgent` is a deterministic execution engine; everything that
requires judgement is delegated to a :class:`Planner`.  Three implementations:

* :class:`SequentialPlanner` - rule based, no LLM.  Builds a pipeline over the
  given roles (research -> coding -> ...).  Deterministic, used for tests and
  the MVP workflow.
* :class:`StaticPlanner`     - executes a user supplied :class:`Plan` DAG.
* :class:`LLMPlanner`        - uses any :class:`LLMAdapter` to analyse the goal,
  emit a JSON plan, optionally add follow-up tasks after each round, and write
  the final synthesis.  Provider agnostic by construction.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Any, Protocol

from ..llm.base import ChatMessage, LLMAdapter
from ..models import AgentInfo
from .models import Plan, SubTask

_log = logging.getLogger("agentcomm.orchestration.planner")


class Planner(Protocol):
    async def plan(self, goal: str, agents: Sequence[AgentInfo], *, context: dict[str, Any]) -> Plan:
        """Analyse *goal* and return an initial :class:`Plan` (may be empty)."""
        ...

    async def replan(self, plan: Plan, agents: Sequence[AgentInfo]) -> list[SubTask]:
        """Called after each execution round. Return *additional* sub-tasks (or [])."""
        ...

    async def synthesize(self, plan: Plan) -> str:
        """Combine the sub-task results into the final answer."""
        ...


# --------------------------------------------------------------------------- helpers
def _online_roles(agents: Sequence[AgentInfo]) -> list[str]:
    roles: list[str] = []
    for a in agents:
        if a.role not in roles:
            roles.append(a.role)
    return roles


def render_results(plan: Plan) -> str:
    """Markdown report of all sub-tasks, used as a fallback synthesis and as LLM input."""
    lines = [f"# Goal\n{plan.goal}"]
    if plan.analysis:
        lines.append(f"\n# Analysis\n{plan.analysis}")
    lines.append("\n# Sub-task results")
    for st in plan.subtasks:
        head = f"\n## [{st.id}] {st.role} -> {st.assigned_to or '-'} ({st.status.value})"
        if st.status.value == "done":
            lines.append(f"{head}\n{st.result}")
        else:
            lines.append(f"{head}\nERROR: {st.error or 'not executed'}")
        # Structured review findings (if any) so a re-planner / synthesiser can
        # react to *why* a result was rejected - conclusions only, no reasoning dump.
        critique = st.reasoning.last_critique
        if critique is not None and (critique.issues or not critique.passed):
            verdict = "passed" if critique.passed else "FAILED"
            lines.append(f"Review by {critique.verifier} (attempt {critique.attempt}): {verdict}"
                         + (f" - {critique.summary}" if critique.summary else ""))
            for issue in critique.issues:
                line = f"  - [{issue.severity.value}/{issue.category.value}] {issue.summary}"
                if issue.recommendation:
                    line += f" -> {issue.recommendation}"
                lines.append(line)
    return "\n".join(lines)


# ----------------------------------------------------------------- SequentialPlanner
class SequentialPlanner:
    """Deterministic pipeline: each role receives the goal plus the previous result.

    ``roles`` fixes the order (e.g. ``["researcher", "coder"]``). Roles that have
    no online agent are skipped at planning time unless ``strict=True``.
    """

    def __init__(self, roles: Sequence[str], *, strict: bool = False) -> None:
        self.roles = list(roles)
        self.strict = strict

    async def plan(self, goal: str, agents: Sequence[AgentInfo], *, context: dict[str, Any]) -> Plan:
        available = set(_online_roles(agents)) | {c for a in agents for c in a.capabilities}
        plan = Plan(goal=goal, analysis=f"Sequential pipeline over roles: {', '.join(self.roles)}")
        prev: str | None = None
        for i, role in enumerate(self.roles, start=1):
            if role not in available and not self.strict:
                plan.log("planned", detail=f"skipping role {role!r}: no online agent")
                continue
            st = SubTask(
                id=f"s{i}_{role}",
                role=role,
                instruction=self._instruction(role, goal, prev),
                depends_on=[prev] if prev else [],
            )
            plan.add(st)
            prev = st.id
        return plan

    @staticmethod
    def _instruction(role: str, goal: str, prev: str | None) -> str:
        base = f"Goal: {goal}\nYour role: {role}."
        if prev:
            base += f"\nUse the output of previous step [{prev}] (provided below) as input."
        return base

    async def replan(self, plan: Plan, agents: Sequence[AgentInfo]) -> list[SubTask]:
        return []

    async def synthesize(self, plan: Plan) -> str:
        return render_results(plan)


# --------------------------------------------------------------------- StaticPlanner
class StaticPlanner:
    """Run a caller-provided DAG unchanged."""

    def __init__(self, subtasks: Sequence[SubTask], *, analysis: str = "static plan") -> None:
        self.subtasks = list(subtasks)
        self.analysis = analysis

    async def plan(self, goal: str, agents: Sequence[AgentInfo], *, context: dict[str, Any]) -> Plan:
        plan = Plan(goal=goal, analysis=self.analysis)
        for st in self.subtasks:
            plan.add(SubTask(id=st.id, role=st.role, instruction=st.instruction,
                             depends_on=list(st.depends_on), metadata=dict(st.metadata)))
        return plan

    async def replan(self, plan: Plan, agents: Sequence[AgentInfo]) -> list[SubTask]:
        return []

    async def synthesize(self, plan: Plan) -> str:
        return render_results(plan)


# ------------------------------------------------------------------------ LLMPlanner
_PLAN_SYSTEM = """You are the planning brain of a Director Agent that coordinates a team of AI agents.
Decompose the user's goal into the smallest useful set of sub-tasks and assign each to one of the
AVAILABLE ROLES only. Express dependencies so that later steps can use earlier results.
Respond with JSON only, no prose, matching exactly:
{
  "analysis": "<one paragraph: what the goal requires and why this decomposition>",
  "subtasks": [
    {"id": "s1", "role": "<role>", "instruction": "<clear, self-contained instruction>", "depends_on": []},
    {"id": "s2", "role": "<role>", "instruction": "...", "depends_on": ["s1"]}
  ]
}
Rules: ids unique; depends_on only references earlier ids; use at most {max_subtasks} sub-tasks;
never invent roles that are not available."""

_REPLAN_SYSTEM = """You are the planning brain of a Director Agent. The team has executed the plan so far.
Decide whether ADDITIONAL sub-tasks are needed to fully achieve the goal (e.g. a failed step
should be retried with a different role, a result needs review, or something is missing).
Respond with JSON only: {"subtasks": [ ...same schema as before... ]}.
Return {"subtasks": []} if the goal is achieved or nothing useful can be added.
New ids must not collide with existing ones. Only use AVAILABLE ROLES."""

_SYNTH_SYSTEM = """You are the Director Agent. Write the final deliverable for the user by integrating the
sub-task results below. Be complete but concise. Mention clearly if some parts failed."""


class LLMPlanner:
    """LLM-driven dynamic planning. Works with *any* :class:`LLMAdapter`."""

    def __init__(
        self,
        adapter: LLMAdapter,
        *,
        max_subtasks: int = 8,
        max_replans: int = 1,
        temperature: float = 0.2,
    ) -> None:
        self.adapter = adapter
        self.max_subtasks = max_subtasks
        self.max_replans = max_replans
        self.temperature = temperature

    # ---------------------------------------------------------------- prompts
    @staticmethod
    def describe_agents(agents: Sequence[AgentInfo]) -> str:
        lines = ["AVAILABLE ROLES:"]
        for a in agents:
            caps = ", ".join(a.capabilities) or "-"
            lines.append(f"- role={a.role} (agent {a.id}, name={a.name}, capabilities: {caps})")
        return "\n".join(lines)

    async def plan(self, goal: str, agents: Sequence[AgentInfo], *, context: dict[str, Any]) -> Plan:
        system = _PLAN_SYSTEM.replace("{max_subtasks}", str(self.max_subtasks))
        user = f"{self.describe_agents(agents)}\n\nGOAL:\n{goal}"
        if context:
            user += f"\n\nCONTEXT:\n{json.dumps(context, ensure_ascii=False)}"
        resp = await self.adapter.complete(
            [ChatMessage(role="user", content=user)], system=system, temperature=self.temperature
        )
        data = parse_json_object(resp.content)
        plan = Plan(goal=goal, analysis=str(data.get("analysis", "")).strip())
        roles = set(_online_roles(agents)) | {c for a in agents for c in a.capabilities}
        for raw in _coerce_subtasks(data.get("subtasks"), roles)[: self.max_subtasks]:
            plan.add(raw)
        plan.log("planned", detail=f"llm produced {len(plan.subtasks)} sub-tasks")
        return plan

    async def replan(self, plan: Plan, agents: Sequence[AgentInfo]) -> list[SubTask]:
        if plan.rounds > self.max_replans:  # rounds counts the initial plan as 1
            return []
        user = f"{self.describe_agents(agents)}\n\n{render_results(plan)}"
        resp = await self.adapter.complete(
            [ChatMessage(role="user", content=user)], system=_REPLAN_SYSTEM,
            temperature=self.temperature,
        )
        try:
            data = parse_json_object(resp.content)
        except ValueError as exc:
            _log.warning("replan: could not parse LLM output (%s); no extra tasks", exc)
            return []
        roles = set(_online_roles(agents)) | {c for a in agents for c in a.capabilities}
        existing = {st.id for st in plan.subtasks}
        new = [st for st in _coerce_subtasks(data.get("subtasks"), roles) if st.id not in existing]
        return new[: self.max_subtasks]

    async def synthesize(self, plan: Plan) -> str:
        resp = await self.adapter.complete(
            [ChatMessage(role="user", content=render_results(plan))], system=_SYNTH_SYSTEM,
            temperature=self.temperature,
        )
        return resp.content.strip()


# ---------------------------------------------------------------------- parsing
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from *text* (tolerates ``` fences and prose)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(text)
        if not m:
            raise ValueError("no JSON object found in LLM output") from None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in LLM output: {exc}") from None
    if not isinstance(obj, dict):
        raise ValueError("LLM output is not a JSON object")  # noqa: TRY004 - callers catch ValueError
    return obj


def _coerce_subtasks(raw: Any, allowed_roles: set[str]) -> list[SubTask]:
    """Turn loosely-typed LLM output into validated :class:`SubTask` objects."""
    if not isinstance(raw, list):
        return []
    out: list[SubTask] = []
    seen: set[str] = set()
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        instruction = str(item.get("instruction", "")).strip()
        if not role or not instruction:
            continue
        if allowed_roles and role not in allowed_roles:
            _log.warning("planner: dropping sub-task with unknown role %r", role)
            continue
        sid = str(item.get("id") or f"s{i}").strip()
        if sid in seen:
            sid = f"{sid}_{i}"
        seen.add(sid)
        deps_raw = item.get("depends_on") or []
        deps = [str(d) for d in deps_raw if isinstance(d, (str, int))] if isinstance(deps_raw, list) else []
        out.append(SubTask(id=sid, role=role, instruction=instruction, depends_on=deps))
    return out
