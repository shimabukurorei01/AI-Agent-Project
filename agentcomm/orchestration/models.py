"""Structured plan / sub-task / result models used by the Director Agent.

These are plain dataclasses (JSON-serialisable via ``to_dict``) so a plan can be
attached to ``Message.metadata``, persisted in history, or produced by an LLM.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from ..models import new_id, utc_now
from .reasoning import ReasoningTrace


class SubTaskStatus(str, Enum):
    PENDING = "pending"  # waiting for dependencies
    READY = "ready"  # dependencies satisfied, not yet dispatched
    RUNNING = "running"  # task_request sent, waiting for result
    DONE = "done"
    FAILED = "failed"  # all attempts exhausted
    SKIPPED = "skipped"  # a dependency failed


class PlanStatus(str, Enum):
    PLANNING = "planning"
    RUNNING = "running"
    COMPLETED = "completed"  # every sub-task done
    PARTIAL = "partial"  # finished, but some sub-tasks failed/skipped
    FAILED = "failed"  # nothing useful produced / planning failed


@dataclass(slots=True)
class SubTask:
    """One unit of work delegated to a single agent."""

    id: str
    role: str  # role or capability used to pick the worker
    instruction: str  # what the worker should do
    depends_on: list[str] = field(default_factory=list)  # ids of prerequisite sub-tasks
    status: SubTaskStatus = SubTaskStatus.PENDING
    assigned_to: str | None = None  # agent id actually used
    task_id: str | None = None  # "<plan task_id>:<sub-task id>"
    request_message_id: str | None = None
    result_message_id: str | None = None
    result: str | None = None
    error: str | None = None
    attempts: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    verified_by: str | None = None  # verifier that issued the final passing critique
    reasoning: ReasoningTrace = field(default_factory=ReasoningTrace)

    @property
    def refinements(self) -> int:
        """Number of critique-driven refinements applied to this sub-task."""
        return self.reasoning.refinement_count

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["reasoning"] = self.reasoning.to_dict()
        d["refinements"] = self.refinements
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubTask:
        data = dict(data)
        data.pop("refinements", None)  # derived field
        data["status"] = SubTaskStatus(data.get("status", SubTaskStatus.PENDING))
        data["reasoning"] = ReasoningTrace.from_dict(data.get("reasoning"))
        return cls(**data)


@dataclass(slots=True)
class PlanEvent:
    """Audit-trail entry describing something that happened while executing a plan."""

    event: str  # planned | dispatched | completed | failed | retry | skipped | replanned | ...
    subtask_id: str | None = None
    detail: str = ""
    timestamp: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Plan:
    """A DAG of :class:`SubTask` objects for one user goal."""

    goal: str
    subtasks: list[SubTask] = field(default_factory=list)
    plan_id: str = field(default_factory=lambda: new_id("plan"))
    task_id: str = field(default_factory=lambda: new_id("task"))
    conversation_id: str = field(default_factory=lambda: new_id("conv"))
    status: PlanStatus = PlanStatus.PLANNING
    analysis: str = ""  # planner's reasoning / task analysis
    final_result: str | None = None
    events: list[PlanEvent] = field(default_factory=list)
    rounds: int = 0  # how many planning rounds (initial + replans)
    created_at: str = field(default_factory=utc_now)
    finished_at: str | None = None

    # ---------------------------------------------------------------- lookup
    def get(self, subtask_id: str) -> SubTask:
        for st in self.subtasks:
            if st.id == subtask_id:
                return st
        raise KeyError(subtask_id)

    def has(self, subtask_id: str) -> bool:
        return any(st.id == subtask_id for st in self.subtasks)

    def add(self, subtask: SubTask) -> SubTask:
        if self.has(subtask.id):
            raise ValueError(f"duplicate sub-task id {subtask.id!r}")
        self.subtasks.append(subtask)
        return subtask

    def ready(self) -> list[SubTask]:
        """Sub-tasks whose dependencies are all DONE and that have not started yet."""
        out: list[SubTask] = []
        for st in self.subtasks:
            if st.status not in (SubTaskStatus.PENDING, SubTaskStatus.READY):
                continue
            if all(self.get(d).status is SubTaskStatus.DONE for d in st.depends_on if self.has(d)):
                out.append(st)
        return out

    def blocked(self) -> list[SubTask]:
        """Pending sub-tasks that can never run because a dependency failed/was skipped."""
        bad = {SubTaskStatus.FAILED, SubTaskStatus.SKIPPED}
        return [
            st
            for st in self.subtasks
            if st.status in (SubTaskStatus.PENDING, SubTaskStatus.READY)
            and any(self.has(d) and self.get(d).status in bad for d in st.depends_on)
        ]

    def is_finished(self) -> bool:
        terminal = {SubTaskStatus.DONE, SubTaskStatus.FAILED, SubTaskStatus.SKIPPED}
        return all(st.status in terminal for st in self.subtasks)

    def done(self) -> list[SubTask]:
        return [st for st in self.subtasks if st.status is SubTaskStatus.DONE]

    def failed(self) -> list[SubTask]:
        return [st for st in self.subtasks if st.status in (SubTaskStatus.FAILED, SubTaskStatus.SKIPPED)]

    def results(self) -> dict[str, str]:
        """``subtask_id -> result`` for completed sub-tasks (in plan order)."""
        return {st.id: st.result or "" for st in self.done()}

    def log(self, event: str, subtask_id: str | None = None, detail: str = "") -> None:
        self.events.append(PlanEvent(event=event, subtask_id=subtask_id, detail=detail))

    def validate(self) -> list[str]:
        """Return a list of structural problems (empty list == valid)."""
        problems: list[str] = []
        ids = [st.id for st in self.subtasks]
        if len(ids) != len(set(ids)):
            problems.append("duplicate sub-task ids")
        for st in self.subtasks:
            for d in st.depends_on:
                if d not in ids:
                    problems.append(f"{st.id}: unknown dependency {d!r}")
                if d == st.id:
                    problems.append(f"{st.id}: depends on itself")
        if not problems and _has_cycle(self.subtasks):
            problems.append("dependency cycle detected")
        return problems

    # ----------------------------------------------------------- serialise
    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "task_id": self.task_id,
            "conversation_id": self.conversation_id,
            "goal": self.goal,
            "status": self.status.value,
            "analysis": self.analysis,
            "rounds": self.rounds,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "final_result": self.final_result,
            "subtasks": [st.to_dict() for st in self.subtasks],
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Plan:
        return cls(
            goal=data["goal"],
            subtasks=[SubTask.from_dict(s) for s in data.get("subtasks", [])],
            plan_id=data.get("plan_id", new_id("plan")),
            task_id=data.get("task_id", new_id("task")),
            conversation_id=data.get("conversation_id", new_id("conv")),
            status=PlanStatus(data.get("status", PlanStatus.PLANNING)),
            analysis=data.get("analysis", ""),
            final_result=data.get("final_result"),
            events=[PlanEvent(**e) for e in data.get("events", [])],
            rounds=int(data.get("rounds", 0)),
            created_at=data.get("created_at", utc_now()),
            finished_at=data.get("finished_at"),
        )


def _has_cycle(subtasks: list[SubTask]) -> bool:
    graph = {st.id: list(st.depends_on) for st in subtasks}
    state: dict[str, int] = {}  # 0 = unvisited, 1 = visiting, 2 = done

    def visit(node: str) -> bool:
        s = state.get(node, 0)
        if s == 1:
            return True
        if s == 2:
            return False
        state[node] = 1
        for dep in graph.get(node, []):
            if visit(dep):
                return True
        state[node] = 2
        return False

    return any(visit(n) for n in graph)
