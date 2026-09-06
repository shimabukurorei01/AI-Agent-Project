"""LLM Adapter interface.

The communication layer never imports anything from here; only agents do.
An adapter turns a list of :class:`ChatMessage` into an :class:`LLMResponse`.
That is the *entire* contract, so any provider (or a local model) can be added.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from ..errors import LLMError

_log = logging.getLogger("agentcomm.llm")

Role = Literal["system", "user", "assistant"]


@dataclass(slots=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(slots=True)
class LLMResponse:
    content: str
    model: str
    provider: str
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class LLMAdapter(ABC):
    """Provider-agnostic chat completion."""

    provider: str = "unknown"

    def __init__(self, model: str, *, timeout: float = 60.0) -> None:
        self.model = model
        self.timeout = timeout

    @abstractmethod
    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(model={self.model!r})"


class HTTPLLMAdapter(LLMAdapter):
    """Helper base for JSON-over-HTTPS providers using only the standard library.

    The blocking ``urllib`` call is executed in a worker thread so the event loop
    (and the other agents) keep running.
    """

    async def _post_json(
        self, url: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
        def _do() -> dict[str, Any]:
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=body, method="POST")
            req.add_header("Content-Type", "application/json")
            for k, v in headers.items():
                req.add_header(k, v)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
                    return data
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                raise LLMError(f"{self.provider} HTTP {exc.code}: {detail}") from exc
            except urllib.error.URLError as exc:
                raise LLMError(f"{self.provider} connection error: {exc.reason}") from exc
            except TimeoutError as exc:
                raise LLMError(f"{self.provider} request timed out after {self.timeout}s") from exc

        return await asyncio.to_thread(_do)
