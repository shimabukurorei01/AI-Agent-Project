"""End-to-end tests with running agent loops (all offline, MockLLMAdapter)."""

from __future__ import annotations

import asyncio

import pytest

from agentcomm import (
    AgentInfo,
    AgentNotFoundError,
    AgentStatus,
    BaseAgent,
    CommunicationLayer,
    EchoAgent,
    LLMAgent,
    ManagerAgent,
    Message,
    MessageStatus,
    MessageType,
    RemoteAgentError,
)
from agentcomm.llm import ChatMessage, MockLLMAdapter


async def test_roundtrip_a_to_b_to_a(layer: CommunicationLayer) -> None:
    """The MVP: Agent A -> Communication Layer -> Agent B -> reply -> Agent A."""
    a = EchoAgent(layer, "agent_a")
    b = EchoAgent(layer, "agent_b")
    await a.start()
    await b.start()
    try:
        assert layer.registry.is_online("agent_a") and layer.registry.is_online("agent_b")
        reply = await a.ask("agent_b", "ping")
        assert reply.sender == "agent_b" and reply.receiver == "agent_a"
        assert reply.content == "agent_b received: ping"
        assert reply.message_type is MessageType.ANSWER

        conv = layer.conversation(reply.conversation_id)
        assert [m.sender for m in conv] == ["agent_a", "agent_b"]
        assert conv[0].status is MessageStatus.REPLIED
        assert conv[1].in_reply_to == conv[0].message_id
    finally:
        await a.stop()
        await b.stop()
    assert not layer.registry.exists("agent_a")


async def test_llm_agent_replies_using_adapter(layer: CommunicationLayer) -> None:
    adapter = MockLLMAdapter(scripted=["The answer is 42."])
    a = EchoAgent(layer, "asker")
    b = LLMAgent(layer, AgentInfo(id="brain", name="Brain", role="assistant",
                                  capabilities=["chat"]), adapter)
    await a.start()
    await b.start()
    try:
        assert layer.registry.get("brain").model == "mock:mock-1"
        reply = await a.ask("brain", "What is the answer?")
        assert reply.content == "The answer is 42."
        assert reply.metadata["llm"]["provider"] == "mock"
        # The adapter saw the incoming question (with sender tag) as the last user turn
        assert adapter.calls[0][-1].role == "user"
        assert "What is the answer?" in adapter.calls[0][-1].content
    finally:
        await a.stop()
        await b.stop()


async def test_llm_agent_memory_uses_conversation_history(layer: CommunicationLayer) -> None:
    adapter = MockLLMAdapter(scripted=["first", "second"])
    a = EchoAgent(layer, "asker")
    b = LLMAgent(layer, AgentInfo(id="brain", name="Brain", role="assistant"), adapter)
    await a.start()
    await b.start()
    try:
        r1 = await a.ask("brain", "q1", conversation_id="conv_x")
        await a.ask("brain", "q2", conversation_id="conv_x")
        assert r1.content == "first"
        second_prompt = adapter.calls[1]
        # history: q1 (user), first (assistant), then q2 (user)
        assert [m.role for m in second_prompt] == ["user", "assistant", "user"]
        assert "q1" in second_prompt[0].content and "first" in second_prompt[1].content
    finally:
        await a.stop()
        await b.stop()


async def test_handler_exception_becomes_error_message(layer: CommunicationLayer) -> None:
    adapter = MockLLMAdapter(fail_with=RuntimeError("provider down"))
    b = LLMAgent(layer, AgentInfo(id="brain", name="Brain", role="assistant"), adapter)
    await layer.register(AgentInfo(id="asker", name="asker", role="test"))
    await b.start()
    try:
        # request() fails fast with the remote error instead of waiting for the timeout
        with pytest.raises(RemoteAgentError) as exc_info:
            await layer.request("asker", "brain", "hello", timeout=5)
        assert "provider down" in str(exc_info.value)
        assert exc_info.value.sender == "brain"
        q = next(m for m in layer.history.all() if m.sender == "asker")
        assert q.status is MessageStatus.FAILED
        # fire-and-forget: the ERROR arrives in the mailbox instead
        await layer.send_to("asker", "brain", "again", reply_required=True)
        err = await layer.receive("asker", timeout=1)
        assert err.message_type is MessageType.ERROR
        assert "provider down" in err.content
        # agent loop survived and is back online
        assert layer.registry.get("brain").status is AgentStatus.ONLINE
    finally:
        await b.stop()


async def test_handler_timeout_is_reported(layer: CommunicationLayer) -> None:
    class Slow(BaseAgent):
        async def handle(self, message: Message) -> None:
            await asyncio.sleep(5)

    slow = Slow(layer, AgentInfo(id="slow", name="slow", role="slow"), handle_timeout=0.1)
    await layer.register(AgentInfo(id="asker", name="asker", role="test"))
    await slow.start()
    try:
        await layer.send_to("asker", "slow", "hurry")
        err = await layer.receive("asker", timeout=2)
        assert err.message_type is MessageType.ERROR and "timed out" in err.content
    finally:
        await slow.stop()


async def test_four_agents_concurrent_and_removal(layer: CommunicationLayer) -> None:
    agents = [
        LLMAgent(layer, AgentInfo(id=f"agent_{x}", name=x, role=role, capabilities=[role]),
                 MockLLMAdapter(model=f"m-{x}"))
        for x, role in (("a", "generalist"), ("b", "researcher"), ("c", "coder"), ("d", "reviewer"))
    ]
    for ag in agents:
        await ag.start()
    a, b, c, d = agents
    try:
        assert len(layer.agents(online_only=True)) == 4
        delivered = await a.broadcast("hello team")
        assert set(delivered) == {"agent_b", "agent_c", "agent_d"}

        replies = await asyncio.gather(
            a.ask("agent_b", "research"), a.ask("agent_c", "code"), a.ask("agent_d", "review"),
        )
        assert [r.sender for r in replies] == ["agent_b", "agent_c", "agent_d"]
        assert all("[m-" in r.content for r in replies)

        await c.stop()
        with pytest.raises(AgentNotFoundError):
            await a.ask("agent_c", "still there?")
        r = await b.ask("agent_d", "review please")
        assert r.sender == "agent_d"
    finally:
        for ag in (a, b, d):
            await ag.stop()


async def test_manager_delegates_by_role_and_aggregates(layer: CommunicationLayer) -> None:
    workers = {
        "researcher": MockLLMAdapter(scripted=["research done"]),
        "coder": MockLLMAdapter(scripted=["code done"]),
        "reviewer": MockLLMAdapter(scripted=["review done"]),
        "tester": MockLLMAdapter(scripted=["tests done"]),
    }
    agents: list[BaseAgent] = [
        LLMAgent(layer, AgentInfo(id=f"{role}_agent", name=role, role=role), ad)
        for role, ad in workers.items()
    ]
    manager = ManagerAgent(layer, AgentInfo(id="manager", name="Manager", role="manager"),
                           task_timeout=2)
    user = EchoAgent(layer, "user")
    agents += [manager, user]
    for ag in agents:
        await ag.start()
    try:
        plan = {role: f"please do {role}" for role in workers}
        result = await user.ask("manager", "build feature X", message_type=MessageType.TASK_REQUEST,
                                task_id="task_1", metadata={"plan": plan})
        assert result.message_type is MessageType.TASK_RESULT
        assert result.metadata["results"] == {
            "researcher": "research done", "coder": "code done",
            "reviewer": "review done", "tester": "tests done",
        }
        for role in workers:
            assert f"### {role} ({role}_agent)" in result.content
        # sub-tasks share the parent task id prefix
        sub = [m for m in layer.history.all() if m.task_id and m.task_id.startswith("task_1:")]
        assert len(sub) == 8  # 4 requests + 4 results
    finally:
        for ag in agents:
            await ag.stop()


async def test_manager_reports_missing_and_timed_out_workers(layer: CommunicationLayer) -> None:
    slow = LLMAgent(layer, AgentInfo(id="coder_agent", name="c", role="coder"),
                    MockLLMAdapter(latency=5))
    manager = ManagerAgent(layer, AgentInfo(id="manager", name="M", role="manager"), task_timeout=0.2)
    user = EchoAgent(layer, "user")
    for ag in (slow, manager, user):
        await ag.start()
    try:
        result = await user.ask("manager", "x", message_type=MessageType.TASK_REQUEST,
                                metadata={"plan": {"coder": "code", "designer": "design"}}, timeout=3)
        res = result.metadata["results"]
        assert "no reply" in res["coder"] and res["coder"].startswith("ERROR")
        assert "no online agent" in res["designer"]
    finally:
        for ag in (slow, manager, user):
            await ag.stop()


async def test_manager_uses_adapter_to_summarise(layer: CommunicationLayer) -> None:
    def summariser(msgs: list[ChatMessage], system: str | None) -> str:
        assert system and "manager" in system.lower()
        return "SUMMARY: " + msgs[-1].content.count("###") .__str__() + " sections"

    worker = LLMAgent(layer, AgentInfo(id="w", name="w", role="researcher"),
                      MockLLMAdapter(scripted=["found it"]))
    manager = ManagerAgent(layer, AgentInfo(id="manager", name="M", role="manager"),
                           adapter=MockLLMAdapter(responder=summariser))
    user = EchoAgent(layer, "user")
    for ag in (worker, manager, user):
        await ag.start()
    try:
        # no explicit plan -> manager fans out to every worker role
        result = await user.ask("manager", "go", message_type=MessageType.TASK_REQUEST)
        assert result.content == "SUMMARY: 2 sections"  # researcher + echo(user) roles
    finally:
        for ag in (worker, manager, user):
            await ag.stop()
