"""Anthropic Messages API adapter (Claude)."""

from __future__ import annotations

from typing import Any

from ..config import get_env
from ..errors import LLMConfigurationError, LLMError
from .base import ChatMessage, HTTPLLMAdapter, LLMResponse


class AnthropicAdapter(HTTPLLMAdapter):
    provider = "anthropic"

    def __init__(
        self,
        model: str = "claude-3-5-haiku-latest",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        api_version: str = "2023-06-01",
        timeout: float = 60.0,
    ) -> None:
        super().__init__(model, timeout=timeout)
        key = api_key or get_env("ANTHROPIC_API_KEY")
        if not key:
            raise LLMConfigurationError("ANTHROPIC_API_KEY is not set (see .env.example)")
        self._api_key: str = key
        default = "https://api.anthropic.com"
        self._base_url = (base_url or get_env("ANTHROPIC_BASE_URL", default) or default).rstrip("/")
        self._api_version = api_version

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        # Anthropic requires alternating user/assistant and a separate system field.
        payload_msgs = [
            {"role": m.role, "content": m.content} for m in messages if m.role != "system"
        ]
        extra_system = "\n".join(m.content for m in messages if m.role == "system")
        full_system = "\n".join(s for s in (system, extra_system) if s) or None
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": payload_msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if full_system:
            payload["system"] = full_system
        data = await self._post_json(
            f"{self._base_url}/v1/messages",
            payload,
            {"x-api-key": self._api_key, "anthropic-version": self._api_version},
        )
        try:
            content = "".join(
                block.get("text", "") for block in data["content"] if block.get("type") == "text"
            )
        except (KeyError, TypeError) as exc:
            raise LLMError("anthropic: malformed response") from exc
        usage = {k: int(v) for k, v in (data.get("usage") or {}).items() if isinstance(v, int)}
        return LLMResponse(content=content, model=data.get("model", self.model),
                           provider=self.provider, usage=usage, raw=data)
