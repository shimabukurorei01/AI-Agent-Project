"""Sender authentication hooks.

Inside a single process the main risk is *accidental* impersonation (a bug that
sets the wrong ``sender``). :class:`RegistryAuthenticator` guards against that.
When agents move to separate processes, :class:`TokenAuthenticator` (or a JWT /
mTLS based implementation of :class:`Authenticator`) should be used instead.
"""

from __future__ import annotations

import secrets
from typing import Protocol

from .models import Message
from .registry import AgentRegistry


class Authenticator(Protocol):
    def authenticate(self, message: Message, credential: str | None) -> bool:
        """Return ``True`` if *message.sender* is allowed to send this message."""
        ...


class RegistryAuthenticator:
    """Sender must be a registered agent (credential is ignored)."""

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry

    def authenticate(self, message: Message, credential: str | None) -> bool:
        return self._registry.exists(message.sender)


class TokenAuthenticator:
    """Sender must present the secret token issued at registration time."""

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry
        self._tokens: dict[str, str] = {}

    def issue(self, agent_id: str) -> str:
        token = secrets.token_urlsafe(24)
        self._tokens[agent_id] = token
        return token

    def revoke(self, agent_id: str) -> None:
        self._tokens.pop(agent_id, None)

    def authenticate(self, message: Message, credential: str | None) -> bool:
        if not self._registry.exists(message.sender) or credential is None:
            return False
        expected = self._tokens.get(message.sender)
        return expected is not None and secrets.compare_digest(expected, credential)
