"""CommunicationLayer: the single object agents talk to.

It wires together :class:`AgentRegistry`, :class:`Transport`,
:class:`HistoryStore` and :class:`MessageRouter` and offers a small, stable API:

    register / unregister / set_status
    send / broadcast / request / reply / receive
    history queries
"""

from __future__ import annotations

import logging
from typing import Any

from .auth import Authenticator
from .history import HistoryStore, InMemoryHistory
from .models import BROADCAST, AgentInfo, AgentStatus, Message, MessageType
from .registry import AgentRegistry
from .router import MessageRouter
from .transport import InMemoryTransport, Transport

_log = logging.getLogger("agentcomm.layer")


class CommunicationLayer:
    def __init__(
        self,
        *,
        transport: Transport | None = None,
        history: HistoryStore | None = None,
        authenticator: Authenticator | None = None,
        default_timeout: float = 30.0,
    ) -> None:
        self.registry = AgentRegistry()
        self.transport: Transport = InMemoryTransport() if transport is None else transport
        self.history: HistoryStore = InMemoryHistory() if history is None else history
        self.router = MessageRouter(
            self.registry,
            self.transport,
            self.history,
            authenticator,
            default_timeout=default_timeout,
        )

    # ---------------------------------------------------------- registration
    async def register(self, info: AgentInfo, *, online: bool = True) -> AgentInfo:
        self.registry.register(info)
        await self.transport.open_mailbox(info.id)
        if online:
            self.registry.set_status(info.id, AgentStatus.ONLINE)
        return info

    async def unregister(self, agent_id: str) -> None:
        """Remove an agent. Other agents keep working; messages to it will fail cleanly."""
        await self.transport.close_mailbox(agent_id)
        if self.registry.exists(agent_id):
            self.registry.unregister(agent_id)

    def set_status(self, agent_id: str, status: AgentStatus) -> None:
        self.registry.set_status(agent_id, status)

    def agents(self, *, online_only: bool = False) -> list[AgentInfo]:
        return self.registry.list(online_only=online_only)

    # -------------------------------------------------------------- messaging
    async def send(self, message: Message) -> list[str]:
        return await self.router.send(message)

    async def send_to(
        self,
        sender: str,
        receiver: str,
        content: str,
        *,
        message_type: MessageType = MessageType.CHAT,
        conversation_id: str | None = None,
        task_id: str | None = None,
        reply_required: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        kwargs: dict[str, Any] = {}
        if conversation_id is not None:
            kwargs["conversation_id"] = conversation_id
        msg = Message(
            sender=sender,
            receiver=receiver,
            content=content,
            message_type=message_type,
            task_id=task_id,
            reply_required=reply_required,
            metadata=metadata or {},
            **kwargs,
        )
        await self.router.send(msg)
        return msg

    async def broadcast(
        self,
        sender: str,
        content: str,
        *,
        message_type: MessageType = MessageType.BROADCAST,
        conversation_id: str | None = None,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        kwargs: dict[str, Any] = {}
        if conversation_id is not None:
            kwargs["conversation_id"] = conversation_id
        msg = Message(
            sender=sender,
            receiver=BROADCAST,
            content=content,
            message_type=message_type,
            task_id=task_id,
            metadata=metadata or {},
            **kwargs,
        )
        return await self.router.send(msg)

    async def request(
        self,
        sender: str,
        receiver: str,
        content: str,
        *,
        message_type: MessageType = MessageType.QUESTION,
        conversation_id: str | None = None,
        task_id: str | None = None,
        timeout: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Send and wait for the reply (raises :class:`MessageTimeoutError`)."""
        kwargs: dict[str, Any] = {}
        if conversation_id is not None:
            kwargs["conversation_id"] = conversation_id
        msg = Message(
            sender=sender,
            receiver=receiver,
            content=content,
            message_type=message_type,
            task_id=task_id,
            reply_required=True,
            metadata=metadata or {},
            **kwargs,
        )
        return await self.router.request(msg, timeout=timeout)

    async def reply(
        self,
        original: Message,
        content: str,
        *,
        message_type: MessageType | None = None,
        reply_required: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        msg = original.reply(
            content,
            message_type=message_type,
            reply_required=reply_required,
            metadata=metadata,
        )
        await self.router.send(msg)
        return msg

    async def receive(self, agent_id: str, *, timeout: float | None = None) -> Message:
        return await self.router.receive(agent_id, timeout=timeout)

    async def send_error(self, original: Message, error: str) -> None:
        await self.router.send_error(original, error)

    def mark_processed(self, message: Message) -> None:
        self.router.mark_processed(message)

    # ---------------------------------------------------------------- history
    def conversation(self, conversation_id: str) -> list[Message]:
        return self.history.by_conversation(conversation_id)

    def task_history(self, task_id: str) -> list[Message]:
        return self.history.by_task(task_id)

    def agent_history(self, agent_id: str) -> list[Message]:
        return self.history.by_agent(agent_id)
