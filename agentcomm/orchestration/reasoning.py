"""Auditable reasoning trace: Question -> Hypothesis -> Execution -> Critique -> Refinement.

Design rule
-----------
This module records **structured, auditable facts** about how a sub-task was
solved - *not* an LLM's private chain-of-thought.  Every record answers a
concrete question ("what was asked", "who executed it", "who verified it",
"what was found", "what was changed") and is plain JSON.

All records are attached to :class:`agentcomm.orchestration.models.SubTask`
(``subtask.reasoning``) so they travel with the plan, share its
``conversation_id`` / ``task_id`` and reference real ``message_id`` values from
the communication layer's history.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from ..models import utc_now


# ------------------------------------------------------------------- 1. Question
@dataclass(slots=True)
class Question:
    """What has to be solved (summary of the task the worker receives)."""

    summary: str
    inputs: list[str] = field(default_factory=list)  # sub-task ids whose results are inputs
    requirements: list[str] = field(default_factory=list)  # explicit acceptance criteria
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Question:
        return cls(**d)


# ----------------------------------------------------------- 2. Hypothesis / Plan
@dataclass(slots=True)
class Hypothesis:
    """Execution approach chosen by the director/planner for this sub-task."""

    approach: str  # e.g. "delegate to role 'coder' using researcher output"
    role: str
    steps: list[str] = field(default_factory=list)
    expected_outcome: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Hypothesis:
        return cls(**d)


# ------------------------------------------------------- 3. Reasoning / Execution
@dataclass(slots=True)
class ExecutionRecord:
    """One attempt to execute the sub-task (who, with what, result summary)."""

    attempt: int
    agent_id: str | None
    request_message_id: str | None = None
    result_message_id: str | None = None
    inputs_used: list[str] = field(default_factory=list)
    outcome: str = "unknown"  # success | error | timeout | rejected
    result_summary: str = ""
    error: str | None = None
    started_at: str = field(default_factory=utc_now)
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExecutionRecord:
        return cls(**d)


# -------------------------------------------------------------------- 4. Critique
class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IssueCategory(str, Enum):
    HALLUCINATION = "hallucination"
    FACT_SPECULATION_MIX = "fact_speculation_mix"
    LOGICAL_LEAP = "logical_leap"
    MISSING_PRECONDITION = "missing_precondition"
    SPEC_MISMATCH = "spec_mismatch"
    RESULT_INCONSISTENCY = "result_inconsistency"
    WORKER_FAILURE = "worker_failure"
    TEST_FAILURE = "test_failure"
    OMISSION = "omission"
    OTHER = "other"


@dataclass(slots=True)
class Issue:
    category: IssueCategory
    severity: Severity
    summary: str
    evidence: str = ""  # quote / observation that supports the finding
    recommendation: str = ""  # suggested fix

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        d["severity"] = self.severity.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Issue:
        d = dict(d)
        d["category"] = _coerce_enum(IssueCategory, d.get("category"), IssueCategory.OTHER)
        d["severity"] = _coerce_enum(Severity, d.get("severity"), Severity.MEDIUM)
        return cls(**d)


@dataclass(slots=True)
class Critique:
    """Verdict of an (ideally independent) verifier on one execution attempt."""

    verifier: str  # verifier id: agent id, "rule_based", "llm:<provider>", ...
    passed: bool
    attempt: int  # execution attempt that was reviewed
    summary: str = ""
    issues: list[Issue] = field(default_factory=list)
    confidence: float = 1.0  # verifier's own confidence in the verdict, 0..1
    independent: bool = True  # False if the worker judged its own output
    verifier_message_ids: list[str] = field(default_factory=list)  # review_request/result ids
    created_at: str = field(default_factory=utc_now)

    @property
    def max_severity(self) -> Severity | None:
        if not self.issues:
            return None
        order = list(Severity)
        return max((i.severity for i in self.issues), key=order.index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verifier": self.verifier,
            "passed": self.passed,
            "attempt": self.attempt,
            "summary": self.summary,
            "issues": [i.to_dict() for i in self.issues],
            "confidence": self.confidence,
            "independent": self.independent,
            "verifier_message_ids": list(self.verifier_message_ids),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Critique:
        return cls(
            verifier=str(d.get("verifier", "unknown")),
            passed=bool(d.get("passed", False)),
            attempt=int(d.get("attempt", 0)),
            summary=str(d.get("summary", "")),
            issues=[Issue.from_dict(i) for i in d.get("issues", []) if isinstance(i, dict)],
            confidence=float(d.get("confidence", 1.0)),
            independent=bool(d.get("independent", True)),
            verifier_message_ids=[str(x) for x in d.get("verifier_message_ids", [])],
            created_at=str(d.get("created_at", utc_now())),
        )


# ------------------------------------------------------------------ 5. Refinement
class RefinementAction(str, Enum):
    RETRY_SAME = "retry_same"  # same worker, with critique feedback
    REASSIGN = "reassign"  # another agent with the same role/capability
    REPLAN = "replan"  # mark failed and let the planner add follow-up tasks
    GATHER_INFO = "gather_info"  # ask another role for missing information first
    ACCEPT = "accept"  # accept despite issues (e.g. only low severity)
    GIVE_UP = "give_up"  # stop refining; sub-task fails


@dataclass(slots=True)
class Refinement:
    """Decision taken after a failed critique."""

    action: RefinementAction
    reason: str
    after_attempt: int
    instruction_delta: str = ""  # extra guidance appended to the worker instruction
    target_role: str | None = None  # for GATHER_INFO / REASSIGN
    exclude_agents: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action"] = self.action.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Refinement:
        d = dict(d)
        d["action"] = _coerce_enum(RefinementAction, d.get("action"), RefinementAction.GIVE_UP)
        return cls(**d)


# ------------------------------------------------------------------- container
@dataclass(slots=True)
class ReasoningTrace:
    question: Question | None = None
    hypothesis: Hypothesis | None = None
    executions: list[ExecutionRecord] = field(default_factory=list)
    critiques: list[Critique] = field(default_factory=list)
    refinements: list[Refinement] = field(default_factory=list)

    # ------------------------------------------------------------- helpers
    @property
    def refinement_count(self) -> int:
        return len(self.refinements)

    @property
    def last_critique(self) -> Critique | None:
        return self.critiques[-1] if self.critiques else None

    @property
    def verified(self) -> bool:
        """True if the most recent critique passed."""
        c = self.last_critique
        return c is not None and c.passed

    def verifiers(self) -> list[str]:
        return sorted({c.verifier for c in self.critiques})

    def executors(self) -> list[str]:
        return sorted({e.agent_id for e in self.executions if e.agent_id})

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question.to_dict() if self.question else None,
            "hypothesis": self.hypothesis.to_dict() if self.hypothesis else None,
            "executions": [e.to_dict() for e in self.executions],
            "critiques": [c.to_dict() for c in self.critiques],
            "refinements": [r.to_dict() for r in self.refinements],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> ReasoningTrace:
        if not d:
            return cls()
        return cls(
            question=Question.from_dict(d["question"]) if d.get("question") else None,
            hypothesis=Hypothesis.from_dict(d["hypothesis"]) if d.get("hypothesis") else None,
            executions=[ExecutionRecord.from_dict(e) for e in d.get("executions", [])],
            critiques=[Critique.from_dict(c) for c in d.get("critiques", [])],
            refinements=[Refinement.from_dict(r) for r in d.get("refinements", [])],
        )


def _coerce_enum(enum_cls: Any, value: Any, default: Any) -> Any:
    try:
        return enum_cls(value)
    except (ValueError, TypeError):
        return default


def summarize(text: str | None, limit: int = 200) -> str:
    """Short, single-line summary used in execution records (never the full payload)."""
    if not text:
        return ""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
