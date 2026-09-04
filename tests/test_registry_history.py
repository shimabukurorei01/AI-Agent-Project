from __future__ import annotations

from pathlib import Path

import pytest

from agentcomm import (
    AgentAlreadyRegisteredError,
    AgentInfo,
    AgentNotFoundError,
    AgentRegistry,
    AgentStatus,
    InMemoryHistory,
    JsonlHistory,
    Message,
    MessageStatus,
)


def _info(i: str, role: str, caps: list[str]) -> AgentInfo:
    return AgentInfo(id=i, name=i, role=role, capabilities=caps)


def test_registry_register_lookup_unregister() -> None:
    reg = AgentRegistry()
    reg.register(_info("a", "researcher", ["research"]))
    assert reg.exists("a") and len(reg) == 1
    assert not reg.is_online("a")  # offline until status set
    reg.set_status("a", AgentStatus.ONLINE)
    assert reg.is_online("a")

    with pytest.raises(AgentAlreadyRegisteredError):
        reg.register(_info("a", "x", []))
    reg.register(_info("a", "x", []), replace=True)

    reg.unregister("a")
    assert not reg.exists("a")
    with pytest.raises(AgentNotFoundError):
        reg.get("a")
    with pytest.raises(AgentNotFoundError):
        reg.unregister("a")


def test_registry_find_by_role_and_capability() -> None:
    reg = AgentRegistry()
    for i, role, caps in [("r", "researcher", ["research", "analysis"]),
                          ("c", "coder", ["coding", "analysis"]),
                          ("t", "tester", ["testing"])]:
        reg.register(_info(i, role, caps))
        reg.set_status(i, AgentStatus.ONLINE)
    reg.set_status("t", AgentStatus.OFFLINE)

    assert [a.id for a in reg.find_by_role("coder")] == ["c"]
    assert {a.id for a in reg.find_by_capability("analysis")} == {"r", "c"}
    assert {a.id for a in reg.find_by_capability("research", "analysis")} == {"r"}
    assert {a.id for a in reg.find_by_capability("research", "coding", match_all=False)} == {"r", "c"}
    assert reg.find_by_capability("testing") == []  # offline excluded by default
    assert [a.id for a in reg.find_by_capability("testing", online_only=False)] == ["t"]


def test_inmemory_history_queries() -> None:
    h = InMemoryHistory()
    assert len(h) == 0
    m1 = Message(sender="a", receiver="b", content="1", conversation_id="c1", task_id="t1")
    m2 = m1.reply("2")
    m3 = Message(sender="x", receiver="y", content="3", conversation_id="c2")
    for m in (m1, m2, m3):
        h.append(m)
    assert [m.content for m in h.by_conversation("c1")] == ["1", "2"]
    assert [m.content for m in h.by_task("t1")] == ["1", "2"]
    assert [m.content for m in h.by_agent("a")] == ["1", "2"]
    assert [m.content for m in h.replies_to(m1.message_id)] == ["2"]
    h.update_status(m1.message_id, MessageStatus.REPLIED)
    assert h.get(m1.message_id).status is MessageStatus.REPLIED  # type: ignore[union-attr]


def test_jsonl_history_persists_and_replays(tmp_path: Path) -> None:
    path = tmp_path / "hist.jsonl"
    h = JsonlHistory(path)
    m = Message(sender="a", receiver="b", content="persist me", conversation_id="c1")
    h.append(m)
    h.update_status(m.message_id, MessageStatus.DELIVERED)
    assert path.read_text().count("\n") == 2

    h2 = JsonlHistory(path)
    assert len(h2) == 1
    restored = h2.get(m.message_id)
    assert restored is not None
    assert restored.content == "persist me"
    assert restored.status is MessageStatus.DELIVERED
