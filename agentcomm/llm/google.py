"""Google Gemini (Generative Language API) adapter."""

from __future__ import annotations

from typing import Any

from ..config import get_env
from ..errors import LLMConfigurationError, LLMError
from .base import ChatMessage, HTTPLLMAdapter, LLMResponse


class GoogleAdapter(HTTPLLMAdapter):
    provider = "google"

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(model, timeout=timeout)
        key = api_key or get_env("GOOGLE_API_KEY")
        if not key:
            raise LLMConfigurationError("GOOGLE_API_KEY is not set (see .env.example)")
        self._api_key: str = key
        default = "https://generativelanguage.googleapis.com"
        self._base_url = (base_url or get_env("GOOGLE_BASE_URL", default) or default).rstrip("/")

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        contents = [
            {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
            for m in messages
            if m.role != "system"
        ]
        extra_system = "\n".join(m.content for m in messages if m.role == "system")
        full_system = "\n".join(s for s in (system, extra_system) if s) or None
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if full_system:
            payload["systemInstruction"] = {"parts": [{"text": full_system}]}
        url = f"{self._base_url}/v1beta/models/{self.model}:generateContent"
        data = await self._post_json(url, payload, {"x-goog-api-key": self._api_key})
        try:
            parts = data["candidates"][0]["content"]["parts"]
            content = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("google: malformed response") from exc
        usage_raw = data.get("usageMetadata") or {}
        usage = {k: int(v) for k, v in usage_raw.items() if isinstance(v, int)}
        return LLMResponse(content=content, model=self.model, provider=self.provider,
                           usage=usage, raw=data)
