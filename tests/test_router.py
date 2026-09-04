"""Router / CommunicationLayer behaviour without agent loops (manual receive)."""

from __future__ import annotations

import asyncio

import pytest

from agentcomm import (
    BROADCAST,
    AgentInfo,
    AgentNotFoundError,
    AgentOfflineError,
    AgentStatus,
    CommunicationLayer,
    Message,
    MessageStatus,
    MessageTimeoutError,
    MessageType,
    UnauthorizedSenderError,
)
from agentcomm.auth import TokenAuthenticator


async def _register(layer: CommunicationLayer, *ids: str, online: bool = True) -> None:
    for i in ids:
        await layer.register(AgentInfo(id=i, name=i, role="test"), online=online)


async def test_send_and_receive_one_to_one(layer: CommunicationLayer) -> None:
    await _register(layer, "a", "b")
    sent = await layer.send_to("a", "b", "hello", message_type=MessageType.CHAT)
    got = await layer.receive("b", timeout=1)
    assert got.message_id == sent.message_id
    assert got.content == "hello"
    assert layer.history.get(sent.message_id).status is MessageStatus.DELIVERED  # type: ignore[union-attr]
    assert layer.transport.pending("b") == 0


async def test_send_to_unknown_agent_raises_and_records_failure(layer: CommunicationLayer) -> None:
    await _register(layer, "a")
    msg = Message(sender="a", receiver="ghost", content="?")
    with pytest.raises(AgentNotFoundError):
        await layer.send(msg)
    assert layer.history.get(msg.message_id).status is MessageStatus.FAILED  # type: ignore[union-attr]
    assert msg.metadata["error"] == "receiver not found"


async def test_send_to_offline_agent_raises(layer: CommunicationLayer) -> None:
    await _register(layer, "a")
    await _register(layer, "b", online=False)
    with pytest.raises(AgentOfflineError):
        await layer.send_to("a", "b", "anyone home?")
    layer.set_status("b", AgentStatus.ONLINE)
    await layer.send_to("a", "b", "now?")
    assert (await layer.receive("b", timeout=1)).content == "now?"


async def test_unregistered_sender_is_rejected(layer: CommunicationLayer) -> None:
    await _register(layer, "b")
    with pytest.raises(UnauthorizedSenderError):
        await layer.send_to("impostor", "b", "hi")


async def test_token_authenticator() -> None:
    layer = CommunicationLayer()
    auth = TokenAuthenticator(layer.registry)
    layer.router.auth = auth
    await _register(layer, "a", "b")
    token = auth.issue("a")
    msg = Message(sender="a", receiver="b", content="secure")
    with pytest.raises(UnauthorizedSenderError):
        await layer.router.send(msg)  # no credential
    with pytest.raises(UnauthorizedSenderError):
        await layer.router.send(Message(sender="a", receiver="b", content="x"), credential="wrong")
    await layer.router.send(Message(sender="a", receiver="b", content="ok"), credential=token)
    assert (await layer.receive("b", timeout=1)).content == "ok"


async def test_broadcast_reaches_all_online_except_sender(layer: CommunicationLayer) -> None:
    await _register(layer, "a", "b", "c")
    await _register(layer, "d", online=False)
    delivered = await layer.broadcast("a", "hello all")
    assert set(delivered) == {"b", "c"}
    for target in ("b", "c"):
        m = await layer.receive(target, timeout=1)
        assert m.content == "hello all"
        assert m.receiver == target
        assert m.message_type is MessageType.BROADCAST
        assert "broadcast_of" in m.metadata
    assert layer.transport.pending("a") == 0
    originals = [m for m in layer.history.all() if m.receiver == BROADCAST]
    assert len(originals) == 1 and originals[0].status is MessageStatus.DELIVERED


async def test_request_reply_correlation(layer: CommunicationLayer) -> None:
    await _register(layer, "a", "b")

    async def responder() -> None:
        m = await layer.receive("b", timeout=1)
        await layer.reply(m, f"re: {m.content}")

    task = asyncio.create_task(responder())
    reply = await layer.request("a", "b", "ping", message_type=MessageType.QUESTION)
    await task
    assert reply.content == "re: ping"
    assert reply.message_type is MessageType.ANSWER
    # the reply is delivered to the Future, not to A's mailbox
    assert layer.transport.pending("a") == 0
    question = next(m for m in layer.history.all() if m.sender == "a")
    assert question.reply_required is True
    assert question.status is MessageStatus.REPLIED


async def test_request_times_out(layer: CommunicationLayer) -> None:
    await _register(layer, "a", "b")
    with pytest.raises(MessageTimeoutError):
        await layer.request("a", "b", "anyone?", timeout=0.1)
    q = next(m for m in layer.history.all() if m.sender == "a")
    assert q.status is MessageStatus.TIMEOUT
    assert layer.router._pending == {}


async def test_unregister_agent_does_not_break_others(layer: CommunicationLayer) -> None:
    await _register(layer, "a", "b", "c")
    await layer.unregister("b")
    assert not layer.registry.exists("b")
    with pytest.raises(AgentNotFoundError):
        await layer.send_to("a", "b", "gone")
    await layer.send_to("a", "c", "still fine")
    assert (await layer.receive("c", timeout=1)).content == "still fine"
    # unregistering twice / unknown id is harmless
    await layer.unregister("b")
    await layer.unregister("nobody")


async def test_send_error_goes_back_to_sender(layer: CommunicationLayer) -> None:
    await _register(layer, "a", "b")
    original = await layer.send_to("a", "b", "do it", message_type=MessageType.TASK_REQUEST, task_id="t1")
    await layer.receive("b", timeout=1)
    await layer.send_error(original, "boom")
    err = await layer.receive("a", timeout=1)
    assert err.message_type is MessageType.ERROR
    assert err.sender == "b" and err.in_reply_to == original.message_id
    assert err.task_id == "t1" and err.content == "boom"


async def test_task_history_and_conversation_history(layer: CommunicationLayer) -> None:
    await _register(layer, "a", "b")
    m1 = await layer.send_to("a", "b", "step 1", task_id="t9", conversation_id="c9")
    got = await layer.receive("b", timeout=1)
    await layer.reply(got, "ack 1")
    await layer.receive("a", timeout=1)
    assert [m.content for m in layer.task_history("t9")] == ["step 1", "ack 1"]
    assert [m.content for m in layer.conversation("c9")] == ["step 1", "ack 1"]
    assert m1 in layer.agent_history("a")
