"""Dynamic planning / orchestration on top of the communication layer.

    User -> DirectorAgent -> (Planner) -> worker agents -> (Verifier) -> DirectorAgent -> final result

Each sub-task carries an auditable reasoning trace:
Question -> Hypothesis -> Execution -> Critique -> Refinement.
"""

from .director import DirectorAgent, WorkerSelector, default_worker_selector
from .models import Plan, PlanEvent, PlanStatus, SubTask, SubTaskStatus
from .planner import (
    LLMPlanner,
    Planner,
    SequentialPlanner,
    StaticPlanner,
    parse_json_object,
    render_results,
)
from .reasoning import (
    Critique,
    ExecutionRecord,
    Hypothesis,
    Issue,
    IssueCategory,
    Question,
    ReasoningTrace,
    Refinement,
    RefinementAction,
    Severity,
)
from .verification import (
    VERIFICATION_CHECKLIST,
    AgentVerifier,
    CompositeVerifier,
    DefaultRefinementPolicy,
    LLMVerifier,
    RefinementPolicy,
    RuleBasedVerifier,
    Verifier,
    feedback_text,
)

__all__ = [
    "VERIFICATION_CHECKLIST",
    "AgentVerifier",
    "CompositeVerifier",
    "Critique",
    "DefaultRefinementPolicy",
    "DirectorAgent",
    "ExecutionRecord",
    "Hypothesis",
    "Issue",
    "IssueCategory",
    "LLMPlanner",
    "LLMVerifier",
    "Plan",
    "PlanEvent",
    "PlanStatus",
    "Planner",
    "Question",
    "ReasoningTrace",
    "Refinement",
    "RefinementAction",
    "RefinementPolicy",
    "RuleBasedVerifier",
    "SequentialPlanner",
    "Severity",
    "StaticPlanner",
    "SubTask",
    "SubTaskStatus",
    "Verifier",
    "WorkerSelector",
    "default_worker_selector",
    "feedback_text",
    "parse_json_object",
    "render_results",
]
