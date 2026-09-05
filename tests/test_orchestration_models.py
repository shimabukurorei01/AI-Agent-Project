"""Plan / SubTask structure, DAG helpers, JSON round-trip, and planner parsing."""

from __future__ import annotations

import pytest

from agentcomm import AgentInfo, AgentStatus
from agentcomm.orchestration import (
    Plan,
    PlanStatus,
    SequentialPlanner,
    StaticPlanner,
    SubTask,
    SubTaskStatus,
    parse_json_object,
    render_results,
)


def _plan() -> Plan:
    p = Plan(goal="g", task_id="task_1", conversation_id="conv_1")
    p.add(SubTask(id="a", role="researcher", instruction="research"))
    p.add(SubTask(id="b", role="coder", instruction="code", depends_on=["a"]))
    p.add(SubTask(id="c", role="reviewer", instruction="review", depends_on=["b"]))
    p.add(SubTask(id="d", role="tester", instruction="test", depends_on=["a"]))
    return p


def test_ready_respects_dependencies() -> None:
    p = _plan()
    assert [s.id for s in p.ready()] == ["a"]
    p.get("a").status = SubTaskStatus.DONE
    assert [s.id for s in p.ready()] == ["b", "d"]
    p.get("b").status = SubTaskStatus.RUNNING
    assert [s.id for s in p.ready()] == ["d"]
    assert not p.is_finished()


def test_blocked_and_finished() -> None:
    p = _plan()
    p.get("a").status = SubTaskStatus.DONE
    p.get("b").status = SubTaskStatus.FAILED
    assert [s.id for s in p.blocked()] == ["c"]
    p.get("c").status = SubTaskStatus.SKIPPED
    p.get("d").status = SubTaskStatus.DONE
    assert p.is_finished()
    assert [s.id for s in p.done()] == ["a", "d"]
    assert [s.id for s in p.failed()] == ["b", "c"]


def test_validate_detects_problems() -> None:
    p = Plan(goal="g")
    p.add(SubTask(id="a", role="r", instruction="i", depends_on=["b"]))
    p.add(SubTask(id="b", role="r", instruction="i", depends_on=["a"]))
    assert "dependency cycle detected" in p.validate()

    p2 = Plan(goal="g")
    p2.add(SubTask(id="a", role="r", instruction="i", depends_on=["zzz", "a"]))
    problems = p2.validate()
    assert any("unknown dependency" in x for x in problems)
    assert any("depends on itself" in x for x in problems)

    with pytest.raises(ValueError):
        _plan().add(SubTask(id="a", role="r", instruction="dup"))

    assert _plan().validate() == []


def test_plan_json_roundtrip() -> None:
    p = _plan()
    p.get("a").status = SubTaskStatus.DONE
    p.get("a").result = "found"
    p.status = PlanStatus.RUNNING
    p.log("dispatched", "a", "attempt 1")
    data = p.to_dict()
    assert data["status"] == "running" and data["subtasks"][0]["status"] == "done"
    back = Plan.from_dict(data)
    assert back.task_id == "task_1" and back.conversation_id == "conv_1"
    assert back.get("a").result == "found"
    assert back.events[0].event == "dispatched"
    assert back.results() == {"a": "found"}


def test_render_results_marks_failures() -> None:
    p = _plan()
    p.get("a").status = SubTaskStatus.DONE
    p.get("a").result = "ok"
    p.get("b").status = SubTaskStatus.FAILED
    p.get("b").error = "boom"
    text = render_results(p)
    assert "[a] researcher" in text and "ok" in text
    assert "ERROR: boom" in text
    assert "ERROR: not executed" in text  # c / d never ran


def _agents(*roles: str) -> list[AgentInfo]:
    return [AgentInfo(id=f"{r}_agent", name=r, role=r, status=AgentStatus.ONLINE) for r in roles]


async def test_sequential_planner_builds_pipeline_and_skips_missing_roles() -> None:
    planner = SequentialPlanner(["researcher", "designer", "coder"])
    plan = await planner.plan("goal", _agents("researcher", "coder"), context={})
    assert [s.id for s in plan.subtasks] == ["s1_researcher", "s3_coder"]
    assert plan.subtasks[1].depends_on == ["s1_researcher"]
    assert "Goal: goal" in plan.subtasks[0].instruction
    assert any("skipping role 'designer'" in e.detail for e in plan.events)
    assert await planner.replan(plan, []) == []


async def test_sequential_planner_strict_keeps_missing_roles() -> None:
    plan = await SequentialPlanner(["designer"], strict=True).plan("g", _agents("coder"), context={})
    assert [s.role for s in plan.subtasks] == ["designer"]


async def test_static_planner_copies_dag() -> None:
    src = [SubTask(id="x", role="coder", instruction="i"), SubTask(id="y", role="tester", instruction="j", depends_on=["x"])]
    plan = await StaticPlanner(src).plan("g", [], context={})
    assert [s.id for s in plan.subtasks] == ["x", "y"]
    plan.get("x").status = SubTaskStatus.DONE
    assert src[0].status is SubTaskStatus.PENDING  # not shared


def test_parse_json_object_tolerates_fences_and_prose() -> None:
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! Here it is:\n{"a": {"b": [1, 2]}}\nHope it helps.') == {"a": {"b": [1, 2]}}
    with pytest.raises(ValueError):
        parse_json_object("no json here")
    with pytest.raises(ValueError):
        parse_json_object("[1, 2, 3]")
