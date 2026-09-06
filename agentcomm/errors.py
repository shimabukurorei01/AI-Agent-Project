"""Exception hierarchy for the communication layer."""

from __future__ import annotations


class AgentCommError(Exception):
    """Base class for all agentcomm errors."""


class AgentNotFoundError(AgentCommError):
    def __init__(self, agent_id: str) -> None:
        super().__init__(f"agent not found: {agent_id!r}")
        self.agent_id = agent_id


class AgentAlreadyRegisteredError(AgentCommError):
    def __init__(self, agent_id: str) -> None:
        super().__init__(f"agent already registered: {agent_id!r}")
        self.agent_id = agent_id


class AgentOfflineError(AgentCommError):
    def __init__(self, agent_id: str) -> None:
        super().__init__(f"agent is offline: {agent_id!r}")
        self.agent_id = agent_id


class UnauthorizedSenderError(AgentCommError):
    """Raised when a message claims a sender that is not registered / not allowed."""

    def __init__(self, sender: str) -> None:
        super().__init__(f"unauthorized sender: {sender!r}")
        self.sender = sender


class MessageTimeoutError(AgentCommError):
    def __init__(self, message_id: str, timeout: float) -> None:
        super().__init__(f"no reply to {message_id!r} within {timeout:.1f}s")
        self.message_id = message_id
        self.timeout = timeout


class DeliveryError(AgentCommError):
    """Transport failed to deliver a message."""


class RemoteAgentError(AgentCommError):
    """The receiving agent answered a ``request()`` with an ERROR message."""

    def __init__(self, sender: str, content: str, message_id: str) -> None:
        super().__init__(f"agent {sender!r} reported error: {content}")
        self.sender = sender
        self.content = content
        self.message_id = message_id


class LLMError(AgentCommError):
    """Raised by LLM adapters (network, auth, quota, malformed response...)."""


class LLMConfigurationError(LLMError):
    """Missing API key / unknown provider etc."""
