"""Transport abstraction: *how* bytes/messages move between agents.

The router only talks to :class:`Transport`. Swapping the in-memory
implementation for WebSocket / Redis / NATS / an A2A-protocol client requires
implementing this small interface and nothing else.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

from .errors import DeliveryError
from .models import Message

_log = logging.getLogger("agentcomm.transport")


class Transport(ABC):
    """Point-to-point delivery of :class:`Message` objects to named mailboxes."""

    @abstractmethod
    async def open_mailbox(self, agent_id: str) -> None:
        """Create the inbound channel for *agent_id* (idempotent)."""

    @abstractmethod
    async def close_mailbox(self, agent_id: str) -> None:
        """Destroy the inbound channel; pending messages are dropped."""

    @abstractmethod
    async def deliver(self, message: Message) -> None:
        """Put *message* into ``message.receiver``'s mailbox. Raise DeliveryError on failure."""

    @abstractmethod
    async def receive(self, agent_id: str, *, timeout: float | None = None) -> Message:
        """Wait for the next message for *agent_id*. Raise ``asyncio.TimeoutError`` on timeout."""

    @abstractmethod
    def pending(self, agent_id: str) -> int:
        """Number of undelivered messages waiting for *agent_id*."""


class InMemoryTransport(Transport):
    """asyncio.Queue-based transport for agents living in the same process."""

    def __init__(self, maxsize: int = 0) -> None:
        self._maxsize = maxsize
        self._queues: dict[str, asyncio.Queue[Message]] = {}

    async def open_mailbox(self, agent_id: str) -> None:
        self._queues.setdefault(agent_id, asyncio.Queue(maxsize=self._maxsize))

    async def close_mailbox(self, agent_id: str) -> None:
        self._queues.pop(agent_id, None)

    async def deliver(self, message: Message) -> None:
        queue = self._queues.get(message.receiver)
        if queue is None:
            raise DeliveryError(f"no mailbox for receiver {message.receiver!r}")
        try:
            queue.put_nowait(message)
        except asyncio.QueueFull as exc:  # pragma: no cover - only with maxsize
            raise DeliveryError(f"mailbox full for {message.receiver!r}") from exc
        _log.debug("delivered %s -> %s (%s)", message.sender, message.receiver, message.message_id)

    async def receive(self, agent_id: str, *, timeout: float | None = None) -> Message:
        queue = self._queues.get(agent_id)
        if queue is None:
            raise DeliveryError(f"no mailbox for {agent_id!r}")
        if timeout is None:
            return await queue.get()
        return await asyncio.wait_for(queue.get(), timeout)

    def pending(self, agent_id: str) -> int:
        queue = self._queues.get(agent_id)
        return queue.qsize() if queue else 0
