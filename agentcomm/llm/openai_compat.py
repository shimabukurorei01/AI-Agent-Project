"""OpenAI Chat Completions adapter.

Also used for **xAI (Grok)** and any other OpenAI-compatible endpoint
(Ollama, vLLM, Groq, Together, OpenRouter, ...) by changing ``base_url``.
"""

from __future__ import annotations

from typing import Any

from ..config import get_env
from ..errors import LLMConfigurationError, LLMError
from .base import ChatMessage, HTTPLLMAdapter, LLMResponse


class OpenAICompatAdapter(HTTPLLMAdapter):
    provider = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url_env: str = "OPENAI_BASE_URL",
        default_base_url: str = "https://api.openai.com/v1",
        provider: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(model, timeout=timeout)
        if provider:
            self.provider = provider
        key = api_key or get_env(api_key_env)
        if not key:
            raise LLMConfigurationError(f"{api_key_env} is not set (see .env.example)")
        self._api_key: str = key
        self._base_url = (base_url or get_env(base_url_env, default_base_url) or default_base_url).rstrip("/")

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        payload_msgs: list[dict[str, str]] = []
        if system:
            payload_msgs.append({"role": "system", "content": system})
        payload_msgs.extend({"role": m.role, "content": m.content} for m in messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": payload_msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = await self._post_json(
            f"{self._base_url}/chat/completions",
            payload,
            {"Authorization": f"Bearer {self._api_key}"},
        )
        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.provider}: malformed response") from exc
        usage = {k: int(v) for k, v in (data.get("usage") or {}).items() if isinstance(v, int)}
        return LLMResponse(content=content, model=data.get("model", self.model),
                           provider=self.provider, usage=usage, raw=data)


class XAIAdapter(OpenAICompatAdapter):
    """xAI Grok via its OpenAI-compatible API."""

    provider = "xai"

    def __init__(self, model: str = "grok-3-mini", **kwargs: Any) -> None:
        kwargs.setdefault("api_key_env", "XAI_API_KEY")
        kwargs.setdefault("base_url_env", "XAI_BASE_URL")
        kwargs.setdefault("default_base_url", "https://api.x.ai/v1")
        kwargs.setdefault("provider", "xai")
        super().__init__(model, **kwargs)
