"""Create adapters from a ``"provider:model"`` string.

    create_adapter("anthropic:claude-3-5-sonnet-latest")
    create_adapter("openai:gpt-4o")
    create_adapter("google:gemini-2.0-flash")
    create_adapter("xai:grok-3")
    create_adapter("mock")            # offline / tests

New providers: call :func:`register_provider` with a callable
``(model: str | None, **kwargs) -> LLMAdapter``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..errors import LLMConfigurationError
from .base import LLMAdapter

AdapterFactory = Callable[..., LLMAdapter]

_PROVIDERS: dict[str, AdapterFactory] = {}


def register_provider(name: str, factory: AdapterFactory) -> None:
    _PROVIDERS[name.lower()] = factory


def available_providers() -> list[str]:
    return sorted(_PROVIDERS)


def create_adapter(spec: str, **kwargs: Any) -> LLMAdapter:
    provider, _, model = spec.partition(":")
    provider = provider.strip().lower()
    factory = _PROVIDERS.get(provider)
    if factory is None:
        raise LLMConfigurationError(
            f"unknown LLM provider {provider!r}; available: {available_providers()}"
        )
    return factory(model.strip() or None, **kwargs)


# --------------------------------------------------------------------- builtins
def _mock(model: str | None, **kw: Any) -> LLMAdapter:
    from .mock import MockLLMAdapter

    return MockLLMAdapter(model or "mock-1", **kw)


def _openai(model: str | None, **kw: Any) -> LLMAdapter:
    from .openai_compat import OpenAICompatAdapter

    return OpenAICompatAdapter(model or "gpt-4o-mini", **kw)


def _xai(model: str | None, **kw: Any) -> LLMAdapter:
    from .openai_compat import XAIAdapter

    return XAIAdapter(model or "grok-3-mini", **kw)


def _anthropic(model: str | None, **kw: Any) -> LLMAdapter:
    from .anthropic import AnthropicAdapter

    return AnthropicAdapter(model or "claude-3-5-haiku-latest", **kw)


def _google(model: str | None, **kw: Any) -> LLMAdapter:
    from .google import GoogleAdapter

    return GoogleAdapter(model or "gemini-2.0-flash", **kw)


register_provider("mock", _mock)
register_provider("openai", _openai)
register_provider("chatgpt", _openai)
register_provider("xai", _xai)
register_provider("grok", _xai)
register_provider("anthropic", _anthropic)
register_provider("claude", _anthropic)
register_provider("google", _google)
register_provider("gemini", _google)
