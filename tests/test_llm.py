"""LLM adapter layer tests. No network: HTTP calls are monkeypatched."""

from __future__ import annotations

from typing import Any

import pytest

from agentcomm import LLMConfigurationError, LLMError, load_dotenv
from agentcomm.llm import (
    ChatMessage,
    LLMAdapter,
    available_providers,
    create_adapter,
    register_provider,
)
from agentcomm.llm.anthropic import AnthropicAdapter
from agentcomm.llm.google import GoogleAdapter
from agentcomm.llm.mock import MockLLMAdapter
from agentcomm.llm.openai_compat import OpenAICompatAdapter, XAIAdapter


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must not depend on whatever the host environment has configured."""
    for var in ("OPENAI_BASE_URL", "XAI_BASE_URL", "ANTHROPIC_BASE_URL", "GOOGLE_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


def test_factory_builtin_providers() -> None:
    assert {"mock", "openai", "anthropic", "google", "xai"} <= set(available_providers())
    ad = create_adapter("mock:my-model")
    assert isinstance(ad, MockLLMAdapter) and ad.model == "my-model"
    with pytest.raises(LLMConfigurationError):
        create_adapter("nope:model")


def test_factory_register_custom_provider() -> None:
    class Local(MockLLMAdapter):
        provider = "local"

    register_provider("local", lambda model, **kw: Local(model or "llama"))
    ad = create_adapter("local")
    assert ad.provider == "local" and ad.model == "llama"


def test_missing_api_key_is_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    for spec in ("openai:gpt-4o", "anthropic:claude", "google:gemini", "xai:grok"):
        with pytest.raises(LLMConfigurationError):
            create_adapter(spec)


async def _patch_post(monkeypatch: pytest.MonkeyPatch, adapter: LLMAdapter, response: dict[str, Any],
                      captured: dict[str, Any]) -> None:
    async def fake_post(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        captured.update(url=url, payload=payload, headers=headers)
        return response

    monkeypatch.setattr(adapter, "_post_json", fake_post)


async def test_openai_adapter_payload_and_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    ad = OpenAICompatAdapter("gpt-4o-mini", api_key="sk-test")
    cap: dict[str, Any] = {}
    await _patch_post(monkeypatch, ad, {
        "model": "gpt-4o-mini-2024", "choices": [{"message": {"content": "hi!"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    }, cap)
    resp = await ad.complete([ChatMessage("user", "hello")], system="be nice")
    assert resp.content == "hi!" and resp.provider == "openai" and resp.usage["prompt_tokens"] == 3
    assert cap["url"] == "https://api.openai.com/v1/chat/completions"
    assert cap["headers"]["Authorization"] == "Bearer sk-test"
    assert cap["payload"]["messages"][0] == {"role": "system", "content": "be nice"}


async def test_xai_adapter_uses_xai_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    ad = XAIAdapter(api_key="xai-test")
    cap: dict[str, Any] = {}
    await _patch_post(monkeypatch, ad, {"choices": [{"message": {"content": "grok"}}]}, cap)
    resp = await ad.complete([ChatMessage("user", "?")])
    assert resp.provider == "xai" and resp.content == "grok"
    assert cap["url"].startswith("https://api.x.ai/v1/")


async def test_anthropic_adapter_payload_and_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    ad = AnthropicAdapter("claude-x", api_key="ak")
    cap: dict[str, Any] = {}
    await _patch_post(monkeypatch, ad, {
        "model": "claude-x", "content": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "there"}],
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }, cap)
    resp = await ad.complete([ChatMessage("system", "sys2"), ChatMessage("user", "hi")], system="sys1")
    assert resp.content == "Hello there" and resp.usage["input_tokens"] == 5
    assert cap["headers"]["x-api-key"] == "ak"
    assert cap["payload"]["system"] == "sys1\nsys2"
    assert cap["payload"]["messages"] == [{"role": "user", "content": "hi"}]


async def test_google_adapter_payload_and_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    ad = GoogleAdapter("gemini-x", api_key="gk")
    cap: dict[str, Any] = {}
    await _patch_post(monkeypatch, ad, {
        "candidates": [{"content": {"parts": [{"text": "gem"}]}}],
        "usageMetadata": {"promptTokenCount": 4},
    }, cap)
    resp = await ad.complete([ChatMessage("user", "q"), ChatMessage("assistant", "a"), ChatMessage("user", "q2")],
                             system="s")
    assert resp.content == "gem" and resp.usage["promptTokenCount"] == 4
    assert cap["url"].endswith("/v1beta/models/gemini-x:generateContent")
    assert [c["role"] for c in cap["payload"]["contents"]] == ["user", "model", "user"]
    assert cap["payload"]["systemInstruction"]["parts"][0]["text"] == "s"


async def test_malformed_response_raises_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    ad = OpenAICompatAdapter(api_key="k")
    await _patch_post(monkeypatch, ad, {"unexpected": True}, {})
    with pytest.raises(LLMError):
        await ad.complete([ChatMessage("user", "x")])


def test_load_dotenv(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text('# comment\nFOO_TEST_KEY="quoted"\nexport BAR_TEST_KEY=bare\n\nBAD LINE\n')
    monkeypatch.delenv("FOO_TEST_KEY", raising=False)
    monkeypatch.setenv("BAR_TEST_KEY", "existing")
    assert load_dotenv(env) == 1  # BAR not overridden
    import os

    assert os.environ["FOO_TEST_KEY"] == "quoted"
    assert os.environ["BAR_TEST_KEY"] == "existing"
    assert load_dotenv(tmp_path / "missing.env") == 0
