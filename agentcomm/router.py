"""Message Router: validates, records, and dispatches messages between agents.

Responsibilities
----------------
* authenticate the sender
* resolve the receiver (single agent or broadcast)
* refuse delivery to unknown / offline agents (raising or returning an ERROR)
* persist every message + status transition to the history store
* correlate replies with pending ``request()`` calls (Future based) and
  enforce time-outs
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .auth import Authenticator, RegistryAuthenticator
from .errors import (
    AgentNotFoundError,
    AgentOfflineError,
    DeliveryError,
    MessageTimeoutError,
    RemoteAgentError,
    UnauthorizedSenderError,
)
from .history import HistoryStore, InMemoryHistory
from .models import BROADCAST, Message, MessageStatus, MessageType
from .registry import AgentRegistry
from .transport import InMemoryTransport, Transport

_log = logging.getLogger("agentcomm.router")


class MessageRouter:
    def __init__(
        self,
        registry: AgentRegistry,
        transport: Transport | None = None,
        history: HistoryStore | None = None,
        authenticator: Authenticator | None = None,
        *,
        default_timeout: float = 30.0,
    ) -> None:
        self.registry = registry
        self.transport: Transport = InMemoryTransport() if transport is None else transport
        self.history: HistoryStore = InMemoryHistory() if history is None else history
        self.auth: Authenticator = RegistryAuthenticator(registry) if authenticator is None else authenticator
        self.default_timeout = default_timeout
        self._pending: dict[str, asyncio.Future[Message]] = {}

    # ------------------------------------------------------------------ send
    async def send(self, message: Message, *, credential: str | None = None) -> list[str]:
        """Deliver *message*. Returns the list of receiver ids it was delivered to.

        Raises :class:`UnauthorizedSenderError`, :class:`AgentNotFoundError`,
        :class:`AgentOfflineError` or :class:`DeliveryError`.
        A failed message is still recorded in history with status FAILED.
        """
        if not self.auth.authenticate(message, credential):
            self._record(message, MessageStatus.FAILED, error="unauthorized sender")
            raise UnauthorizedSenderError(message.sender)

        if message.receiver == BROADCAST:
            return await self._broadcast(message)

        if not self.registry.exists(message.receiver):
            self._record(message, MessageStatus.FAILED, error="receiver not found")
            raise AgentNotFoundError(message.receiver)
        if not self.registry.is_online(message.receiver):
            self._record(message, MessageStatus.FAILED, error="receiver offline")
            raise AgentOfflineError(message.receiver)

        self._record(message, MessageStatus.SENT)

        # A reply to an in-flight ``request()`` is handed straight to the waiting
        # Future instead of the mailbox so the requester never sees it twice.
        if self._resolve_pending(message):
            self.history.update_status(message.message_id, MessageStatus.DELIVERED)
            return [message.receiver]

        try:
            await self.transport.deliver(message)
        except DeliveryError:
            self.history.update_status(message.message_id, MessageStatus.FAILED)
            raise
        self.history.update_status(message.message_id, MessageStatus.DELIVERED)
        return [message.receiver]

    async def _broadcast(self, message: Message) -> list[str]:
        targets = [a for a in self.registry.ids(online_only=True) if a != message.sender]
        self._record(message, MessageStatus.SENT)
        delivered: list[str] = []
        for target in targets:
            copy = Message(
                sender=message.sender,
                receiver=target,
                content=message.content,
                message_type=message.message_type,
                message_id=f"{message.message_id}:{target}",
                conversation_id=message.conversation_id,
                task_id=message.task_id,
                timestamp=message.timestamp,
                reply_required=message.reply_required,
                in_reply_to=message.in_reply_to,
                metadata={**message.metadata, "broadcast_of": message.message_id},
            )
            try:
                await self.transport.deliver(copy)
                delivered.append(target)
            except DeliveryError as exc:
                _log.warning("broadcast to %s failed: %s", target, exc)
        status = MessageStatus.DELIVERED if delivered else MessageStatus.FAILED
        self.history.update_status(message.message_id, status)
        _log.info("broadcast %s from %s -> %d agents", message.message_id, message.sender, len(delivered))
        return delivered

    # --------------------------------------------------------------- request
    async def request(
        self,
        message: Message,
        *,
        timeout: float | None = None,
        credential: str | None = None,
    ) -> Message:
        """Send *message* and wait for a reply (``in_reply_to == message.message_id``).

        Raises :class:`MessageTimeoutError` if no reply arrives in time and
        :class:`RemoteAgentError` if the receiver answers with an ERROR message
        (fail fast instead of waiting for the timeout).
        """
        timeout = self.default_timeout if timeout is None else timeout
        message.reply_required = True
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Message] = loop.create_future()
        self._pending[message.message_id] = fut
        try:
            await self.send(message, credential=credential)
            reply = await asyncio.wait_for(fut, timeout)
            if reply.message_type == MessageType.ERROR:
                self.history.update_status(message.message_id, MessageStatus.FAILED)
                raise RemoteAgentError(reply.sender, reply.content, reply.message_id)
            return reply
        except TimeoutError:
            self.history.update_status(message.message_id, MessageStatus.TIMEOUT)
            _log.warning("request %s timed out after %.1fs", message.message_id, timeout)
            raise MessageTimeoutError(message.message_id, timeout) from None
        finally:
            self._pending.pop(message.message_id, None)

    def _resolve_pending(self, message: Message) -> bool:
        """Complete a waiting ``request()`` Future. Returns True if one was resolved."""
        if message.in_reply_to is None:
            return False
        fut = self._pending.get(message.in_reply_to)
        if fut is None or fut.done():
            return False
        fut.set_result(message)
        self.history.update_status(message.in_reply_to, MessageStatus.REPLIED)
        return True

    # --------------------------------------------------------------- receive
    async def receive(self, agent_id: str, *, timeout: float | None = None) -> Message:
        msg = await self.transport.receive(agent_id, timeout=timeout)
        self.registry.heartbeat(agent_id)
        return msg

    def mark_processed(self, message: Message) -> None:
        """Mark as PROCESSED unless a reply has already been correlated (REPLIED wins)."""
        current = self.history.get(message.message_id)
        if current is not None and current.status == MessageStatus.REPLIED:
            return
        self.history.update_status(message.message_id, MessageStatus.PROCESSED)

    # ---------------------------------------------------------------- errors
    async def send_error(self, original: Message, error: str, *, sender: str | None = None) -> None:
        """Send an ERROR message back to the sender of *original* (best effort)."""
        err = Message(
            sender=sender or original.receiver,
            receiver=original.sender,
            content=error,
            message_type=MessageType.ERROR,
            conversation_id=original.conversation_id,
            task_id=original.task_id,
            in_reply_to=original.message_id,
            metadata={"failed_message_id": original.message_id},
        )
        try:
            await self.send(err)
        except Exception as exc:  # noqa: BLE001 - best effort
            _log.error("could not deliver error for %s: %s", original.message_id, exc)

    # -------------------------------------------------------------- internal
    def _record(self, message: Message, status: MessageStatus, **extra: Any) -> None:
        message.status = status
        if extra:
            message.metadata.update(extra)
        self.history.append(message)
        _log.debug("%s %s -> %s [%s] %s", status.value, message.sender, message.receiver,
                   message.message_type.value, message.message_id)
