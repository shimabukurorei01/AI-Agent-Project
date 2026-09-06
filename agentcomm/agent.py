"""Agent base classes.

* :class:`BaseAgent`  - registration + receive loop + error handling. Subclass
  and implement :meth:`BaseAgent.handle` to create any kind of agent.
* :class:`EchoAgent`  - trivial agent used for smoke tests.
* :class:`LLMAgent`   - answers via an :class:`LLMAdapter`, with per-conversation
  memory built from the shared history.
* :class:`ManagerAgent` - delegates tasks to workers by role/capability and
  aggregates their results.

Agents talk to the world **only** through :class:`CommunicationLayer`; they never
reference each other directly, so adding/removing agents cannot break others.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .errors import AgentCommError, MessageTimeoutError
from .layer import CommunicationLayer
from .llm.base import ChatMessage, LLMAdapter, Role
from .models import AgentInfo, AgentStatus, Message, MessageType, new_id

_log = logging.getLogger("agentcomm.agent")


class BaseAgent:
    def __init__(
        self,
        layer: CommunicationLayer,
        info: AgentInfo,
        *,
        handle_timeout: float | None = None,
    ) -> None:
        self.layer = layer
        self.info = info
        self.handle_timeout = handle_timeout
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self.log = logging.getLogger(f"agentcomm.agent.{info.id}")

    # ------------------------------------------------------------- identity
    @property
    def id(self) -> str:
        return self.info.id

    @property
    def role(self) -> str:
        return self.info.role

    # ------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        """Register with the layer and start the background receive loop."""
        await self.layer.register(self.info, online=True)
        self._stopping.clear()
        self._task = asyncio.create_task(self._run(), name=f"agent:{self.id}")
        self.log.info("started (%s / %s)", self.info.role, self.info.model)

    async def stop(self, *, unregister: bool = True) -> None:
        self._stopping.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001
                self.log.warning("receive loop ended with error: %s", exc)
            self._task = None
        if unregister:
            await self.layer.unregister(self.id)
        else:
            self.layer.set_status(self.id, AgentStatus.OFFLINE)
        self.log.info("stopped")

    async def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                message = await self.layer.receive(self.id, timeout=0.5)
            except TimeoutError:
                continue
            except AgentCommError as exc:
                self.log.error("receive failed: %s", exc)
                await asyncio.sleep(0.1)
                continue
            await self._dispatch(message)

    async def _dispatch(self, message: Message) -> None:
        self.layer.set_status(self.id, AgentStatus.BUSY)
        try:
            if self.handle_timeout:
                await asyncio.wait_for(self.handle(message), self.handle_timeout)
            else:
                await self.handle(message)
            self.layer.mark_processed(message)
        except TimeoutError:
            self.log.error("handling %s timed out", message.message_id)
            await self.layer.send_error(message, f"{self.id}: handler timed out")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log.exception("error handling %s", message.message_id)
            await self.layer.send_error(message, f"{self.id}: {type(exc).__name__}: {exc}")
        finally:
            if self.layer.registry.exists(self.id):
                self.layer.set_status(self.id, AgentStatus.ONLINE)

    # ------------------------------------------------------------ behaviour
    async def handle(self, message: Message) -> None:
        """Process one inbound message. Override in subclasses."""
        raise NotImplementedError

    # ------------------------------------------------------------ shortcuts
    async def send(self, receiver: str, content: str, **kwargs: Any) -> Message:
        return await self.layer.send_to(self.id, receiver, content, **kwargs)

    async def ask(self, receiver: str, content: str, **kwargs: Any) -> Message:
        return await self.layer.request(self.id, receiver, content, **kwargs)

    async def delegate(
        self, receiver: str, content: str, *, task_id: str | None = None, **kwargs: Any
    ) -> Message:
        return await self.layer.request(
            self.id,
            receiver,
            content,
            message_type=MessageType.TASK_REQUEST,
            task_id=task_id or new_id("task"),
            **kwargs,
        )

    async def reply(self, original: Message, content: str, **kwargs: Any) -> Message:
        return await self.layer.reply(original, content, **kwargs)

    async def broadcast(self, content: str, **kwargs: Any) -> list[str]:
        return await self.layer.broadcast(self.id, content, **kwargs)


class EchoAgent(BaseAgent):
    """Replies to every message that requires a reply with an echo."""

    def __init__(self, layer: CommunicationLayer, agent_id: str, name: str | None = None) -> None:
        super().__init__(
            layer,
            AgentInfo(id=agent_id, name=name or agent_id, role="echo", model="none",
                      capabilities=["echo"]),
        )

    async def handle(self, message: Message) -> None:
        if message.message_type == MessageType.ERROR:
            self.log.warning("received error: %s", message.content)
            return
        if message.reply_required:
            await self.reply(message, f"{self.id} received: {message.content}")


class LLMAgent(BaseAgent):
    """Agent whose behaviour is produced by an LLM adapter.

    Memory: the prompt is rebuilt from the shared conversation history, filtered
    to messages this agent saw, so restarting the process with a persistent
    :class:`JsonlHistory` keeps context.
    """

    def __init__(
        self,
        layer: CommunicationLayer,
        info: AgentInfo,
        adapter: LLMAdapter,
        *,
        system_prompt: str | None = None,
        max_history: int = 20,
        handle_timeout: float | None = 120.0,
    ) -> None:
        super().__init__(layer, info, handle_timeout=handle_timeout)
        self.adapter = adapter
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.max_history = max_history
        if info.model == "none":
            info.model = f"{adapter.provider}:{adapter.model}"

    def _default_system_prompt(self) -> str:
        caps = ", ".join(self.info.capabilities) or "general assistance"
        return (
            f"You are {self.info.name} (id: {self.info.id}), an autonomous agent with the role "
            f"'{self.info.role}'. Your capabilities: {caps}. You collaborate with other agents "
            "by exchanging messages. Be concise and precise."
        )

    def build_prompt(self, message: Message) -> list[ChatMessage]:
        history = self.layer.conversation(message.conversation_id)
        history = [m for m in history if self.id in (m.sender, m.receiver)
                   and m.message_type != MessageType.ERROR
                   and m.message_id != message.message_id]
        prompt: list[ChatMessage] = []
        for m in history[-self.max_history:]:
            role: Role = "assistant" if m.sender == self.id else "user"
            prompt.append(ChatMessage(role=role, content=f"[{m.sender} | {m.message_type.value}] {m.content}"))
        prompt.append(ChatMessage(
            role="user",
            content=f"[{message.sender} | {message.message_type.value}] {message.content}",
        ))
        return prompt

    async def handle(self, message: Message) -> None:
        if message.message_type in (MessageType.ERROR, MessageType.ACK, MessageType.STATUS):
            self.log.info("%s from %s: %s", message.message_type.value, message.sender, message.content)
            return
        if not message.reply_required and message.message_type == MessageType.BROADCAST:
            return  # informational; nothing to do
        if not message.reply_required and message.in_reply_to is not None:
            return  # a reply to something we sent fire-and-forget

        response = await self.adapter.complete(self.build_prompt(message), system=self.system_prompt)
        if message.reply_required:
            await self.reply(
                message,
                response.content,
                metadata={"llm": {"provider": response.provider, "model": response.model,
                                  "usage": response.usage}},
            )


class ManagerAgent(BaseAgent):
    """Coordinates worker agents: splits a request into sub-tasks by role.

    ``plan`` maps a *role* to the instruction that role should receive. The
    manager delegates to the first online agent having that role (or the
    given capability), waits for all results concurrently, and returns a
    combined report. If an optional ``adapter`` is supplied it is used to
    synthesise the final answer; otherwise results are concatenated.
    """

    def __init__(
        self,
        layer: CommunicationLayer,
        info: AgentInfo,
        *,
        adapter: LLMAdapter | None = None,
        task_timeout: float = 60.0,
    ) -> None:
        super().__init__(layer, info)
        self.adapter = adapter
        self.task_timeout = task_timeout

    def find_worker(self, role: str) -> str | None:
        candidates = self.layer.registry.find_by_role(role) or \
            self.layer.registry.find_by_capability(role)
        candidates = [c for c in candidates if c.id != self.id]
        return candidates[0].id if candidates else None

    async def run_plan(
        self,
        plan: dict[str, str],
        *,
        conversation_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Message | Exception]:
        """Delegate each ``role -> instruction`` concurrently; returns results per role."""
        task_id = task_id or new_id("task")

        async def _one(role: str, instruction: str) -> Message | Exception:
            worker = self.find_worker(role)
            if worker is None:
                return AgentCommError(f"no online agent with role/capability {role!r}")
            try:
                return await self.delegate(
                    worker,
                    instruction,
                    task_id=f"{task_id}:{role}",
                    conversation_id=conversation_id,
                    timeout=self.task_timeout,
                )
            except (MessageTimeoutError, AgentCommError) as exc:
                return exc

        results = await asyncio.gather(*(_one(r, i) for r, i in plan.items()))
        return dict(zip(plan.keys(), results, strict=True))

    async def handle(self, message: Message) -> None:
        if message.message_type == MessageType.ERROR:
            self.log.warning("error from %s: %s", message.sender, message.content)
            return
        if not message.reply_required:
            return
        plan = message.metadata.get("plan")
        if not isinstance(plan, dict):
            # Default plan: forward the same request to every distinct worker role.
            roles = {a.role for a in self.layer.registry.list(online_only=True)} - {self.role}
            plan = {role: message.content for role in sorted(roles)}
        results = await self.run_plan(
            plan, conversation_id=message.conversation_id, task_id=message.task_id
        )
        report = await self.summarise(message, results)
        await self.reply(message, report, message_type=MessageType.TASK_RESULT,
                         metadata={"results": {
                             r: (m.content if isinstance(m, Message) else f"ERROR: {m}")
                             for r, m in results.items()}})

    async def summarise(self, request: Message, results: dict[str, Message | Exception]) -> str:
        lines = [f"## Results for: {request.content}"]
        for role, res in results.items():
            if isinstance(res, Message):
                lines.append(f"\n### {role} ({res.sender})\n{res.content}")
            else:
                lines.append(f"\n### {role}\nERROR: {res}")
        combined = "\n".join(lines)
        if self.adapter is None:
            return combined
        resp = await self.adapter.complete(
            [ChatMessage(role="user", content=combined)],
            system="You are a manager. Synthesise the team's results into one clear answer.",
        )
        return resp.content
