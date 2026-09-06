from __future__ import annotations

import json

from agentcomm import AgentInfo, AgentStatus, Message, MessageStatus, MessageType


def test_message_defaults_and_ids() -> None:
    m = Message(sender="a", receiver="b", content="hi")
    assert m.message_id.startswith("msg_")
    assert m.conversation_id.startswith("conv_")
    assert m.task_id is None
    assert m.message_type is MessageType.CHAT
    assert m.status is MessageStatus.CREATED
    assert m.reply_required is False
    assert m.timestamp.endswith("+00:00")


def test_message_roundtrip_json() -> None:
    m = Message(sender="a", receiver="b", content="hi", message_type=MessageType.TASK_REQUEST,
                task_id="task_1", reply_required=True, metadata={"k": 1})
    data = json.loads(json.dumps(m.to_dict()))
    assert data["message_type"] == "task_request"
    back = Message.from_dict(data)
    assert back == m


def test_reply_links_and_infers_type() -> None:
    q = Message(sender="a", receiver="b", content="?", message_type=MessageType.QUESTION,
                task_id="t1", reply_required=True)
    r = q.reply("!")
    assert r.sender == "b" and r.receiver == "a"
    assert r.in_reply_to == q.message_id
    assert r.conversation_id == q.conversation_id
    assert r.task_id == "t1"
    assert r.message_type is MessageType.ANSWER

    t = Message(sender="a", receiver="b", content="do", message_type=MessageType.TASK_REQUEST)
    assert t.reply("done").message_type is MessageType.TASK_RESULT
    assert t.reply("x", message_type=MessageType.TASK_REJECTED).message_type is MessageType.TASK_REJECTED


def test_agent_info_roundtrip() -> None:
    info = AgentInfo(id="agent_001", name="Research Agent", role="researcher", model="claude",
                     capabilities=["research", "analysis"], status=AgentStatus.ONLINE)
    data = info.to_dict()
    assert data["status"] == "online"
    assert AgentInfo.from_dict(data) == info
