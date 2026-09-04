"""agentcomm - Agent-to-Agent Communication Layer.

Two independent layers:

    Agent -> CommunicationLayer -> other Agent      (this package's core)
    Agent -> LLMAdapter        -> provider API      (``agentcomm.llm``)
"""

from .agent import BaseAgent, EchoAgent, LLMAgent, ManagerAgent
from .config import Settings, configure_logging, load_dotenv
from .errors import (
    AgentAlreadyRegisteredError,
    AgentCommError,
    AgentNotFoundError,
    AgentOfflineError,
    DeliveryError,
    LLMConfigurationError,
    LLMError,
    MessageTimeoutError,
    RemoteAgentError,
    UnauthorizedSenderError,
)
from .history import HistoryStore, InMemoryHistory, JsonlHistory
from .layer import CommunicationLayer
from .models import (
    BROADCAST,
    AgentInfo,
    AgentStatus,
    Message,
    MessageStatus,
    MessageType,
    new_id,
)
from .registry import AgentRegistry
from .router import MessageRouter
from .transport import InMemoryTransport, Transport

__version__ = "0.1.0"

__all__ = [
    "BROADCAST",
    "AgentAlreadyRegisteredError",
    "AgentCommError",
    "AgentInfo",
    "AgentNotFoundError",
    "AgentOfflineError",
    "AgentRegistry",
    "AgentStatus",
    "BaseAgent",
    "CommunicationLayer",
    "DeliveryError",
    "EchoAgent",
    "HistoryStore",
    "InMemoryHistory",
    "InMemoryTransport",
    "JsonlHistory",
    "LLMAgent",
    "LLMConfigurationError",
    "LLMError",
    "ManagerAgent",
    "Message",
    "MessageRouter",
    "MessageStatus",
    "MessageTimeoutError",
    "MessageType",
    "RemoteAgentError",
    "Settings",
    "Transport",
    "UnauthorizedSenderError",
    "configure_logging",
    "load_dotenv",
    "new_id",
]
