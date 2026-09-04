"""Deterministic adapter for tests and offline demos (no network, no keys)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from .base import ChatMessage, LLMAdapter, LLMResponse

Responder = Callable[[list[ChatMessage], str | None], str]


class MockLLMAdapter(LLMAdapter):
    provider = "mock"

    def __init__(
        self,
        model: str = "mock-1",
        *,
        responder: Responder | None = None,
        scripted: list[str] | None = None,
        latency: float = 0.0,
        fail_with: Exception | None = None,
    ) -> None:
        super().__init__(model)
        self._responder = responder
        self._scripted = list(scripted or [])
        self._latency = latency
        self._fail_with = fail_with
        self.calls: list[list[ChatMessage]] = []

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        if self._latency:
            await asyncio.sleep(self._latency)
        if self._fail_with is not None:
            raise self._fail_with
        if self._scripted:
            text = self._scripted.pop(0)
        elif self._responder is not None:
            text = self._responder(messages, system)
        else:
            last = messages[-1].content if messages else ""
            text = f"[{self.model}] echo: {last}"
        return LLMResponse(content=text, model=self.model, provider=self.provider)
