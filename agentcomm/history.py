"""Conversation history / memory storage.

Two implementations are provided:

* :class:`InMemoryHistory` - fast, volatile, good for tests.
* :class:`JsonlHistory`   - append-only JSON-lines file, survives restarts.

Both implement :class:`HistoryStore`, so a database-backed store can be added
later without touching the router.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from .models import Message, MessageStatus

_log = logging.getLogger("agentcomm.history")


class HistoryStore(ABC):
    @abstractmethod
    def append(self, message: Message) -> None: ...

    @abstractmethod
    def update_status(self, message_id: str, status: MessageStatus) -> None: ...

    @abstractmethod
    def get(self, message_id: str) -> Message | None: ...

    @abstractmethod
    def all(self) -> list[Message]: ...

    def by_conversation(self, conversation_id: str) -> list[Message]:
        return [m for m in self.all() if m.conversation_id == conversation_id]

    def by_task(self, task_id: str) -> list[Message]:
        return [m for m in self.all() if m.task_id == task_id]

    def by_agent(self, agent_id: str) -> list[Message]:
        return [m for m in self.all() if agent_id in (m.sender, m.receiver)]

    def replies_to(self, message_id: str) -> list[Message]:
        return [m for m in self.all() if m.in_reply_to == message_id]

    def __len__(self) -> int:
        return len(self.all())


class InMemoryHistory(HistoryStore):
    def __init__(self) -> None:
        self._messages: dict[str, Message] = {}
        self._order: list[str] = []

    def append(self, message: Message) -> None:
        if message.message_id not in self._messages:
            self._order.append(message.message_id)
        self._messages[message.message_id] = message

    def update_status(self, message_id: str, status: MessageStatus) -> None:
        msg = self._messages.get(message_id)
        if msg is not None:
            msg.status = status

    def get(self, message_id: str) -> Message | None:
        return self._messages.get(message_id)

    def all(self) -> list[Message]:
        return [self._messages[i] for i in self._order]

    def clear(self) -> None:
        self._messages.clear()
        self._order.clear()


class JsonlHistory(InMemoryHistory):
    """In-memory index + append-only JSONL log on disk.

    Every ``append`` and ``update_status`` writes a line; on start-up the file is
    replayed so the latest state of each message is restored.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        super().__init__()
        self._path = Path(path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._replay()

    def _replay(self) -> None:
        if not self._path.is_file():
            return
        count = 0
        with self._path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    _log.warning("skipping corrupt history line")
                    continue
                if record.get("_op") == "status":
                    super().update_status(record["message_id"], MessageStatus(record["status"]))
                else:
                    super().append(Message.from_dict(record))
                count += 1
        _log.info("replayed %d history records from %s", count, self._path)

    def _write(self, record: dict[str, object]) -> None:
        with self._lock, self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def append(self, message: Message) -> None:
        super().append(message)
        self._write(message.to_dict())

    def update_status(self, message_id: str, status: MessageStatus) -> None:
        super().update_status(message_id, status)
        self._write({"_op": "status", "message_id": message_id, "status": status.value})
