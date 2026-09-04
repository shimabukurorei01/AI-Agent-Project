"""Agent Registry: who exists, what they can do, and whether they are online."""

from __future__ import annotations

import builtins
import logging
from collections.abc import Iterator

from .errors import AgentAlreadyRegisteredError, AgentNotFoundError
from .models import AgentInfo, AgentStatus, utc_now

_log = logging.getLogger("agentcomm.registry")


class AgentRegistry:
    """In-memory registry of :class:`AgentInfo` records.

    The registry is intentionally decoupled from agent *objects*: it only knows
    descriptors. This keeps it usable in a future networked deployment where
    agents live in other processes and register over the wire.
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentInfo] = {}

    # ------------------------------------------------------------ registration
    def register(self, info: AgentInfo, *, replace: bool = False) -> AgentInfo:
        if info.id in self._agents and not replace:
            raise AgentAlreadyRegisteredError(info.id)
        info.registered_at = utc_now()
        info.last_seen = info.registered_at
        self._agents[info.id] = info
        _log.info("registered agent %s (%s, role=%s, model=%s)", info.id, info.name, info.role, info.model)
        return info

    def unregister(self, agent_id: str) -> AgentInfo:
        try:
            info = self._agents.pop(agent_id)
        except KeyError:
            raise AgentNotFoundError(agent_id) from None
        _log.info("unregistered agent %s", agent_id)
        return info

    # -------------------------------------------------------------- lookups
    def get(self, agent_id: str) -> AgentInfo:
        try:
            return self._agents[agent_id]
        except KeyError:
            raise AgentNotFoundError(agent_id) from None

    def exists(self, agent_id: str) -> bool:
        return agent_id in self._agents

    def is_online(self, agent_id: str) -> bool:
        info = self._agents.get(agent_id)
        return info is not None and info.status in (AgentStatus.ONLINE, AgentStatus.BUSY)

    def list(self, *, online_only: bool = False) -> builtins.list[AgentInfo]:
        agents = builtins.list(self._agents.values())
        if online_only:
            agents = [a for a in agents if self.is_online(a.id)]
        return agents

    def ids(self, *, online_only: bool = False) -> builtins.list[str]:
        return [a.id for a in self.list(online_only=online_only)]

    def find_by_role(self, role: str, *, online_only: bool = True) -> builtins.list[AgentInfo]:
        return [a for a in self.list(online_only=online_only) if a.role == role]

    def find_by_capability(
        self, *capabilities: str, online_only: bool = True, match_all: bool = True
    ) -> builtins.list[AgentInfo]:
        wanted = set(capabilities)
        out: builtins.list[AgentInfo] = []
        for a in self.list(online_only=online_only):
            have = set(a.capabilities)
            if (match_all and wanted <= have) or (not match_all and wanted & have):
                out.append(a)
        return out

    # ---------------------------------------------------------------- status
    def set_status(self, agent_id: str, status: AgentStatus) -> AgentInfo:
        info = self.get(agent_id)
        if info.status != status:
            _log.debug("agent %s status %s -> %s", agent_id, info.status.value, status.value)
        info.status = status
        info.last_seen = utc_now()
        return info

    def heartbeat(self, agent_id: str) -> None:
        self.get(agent_id).last_seen = utc_now()

    # ----------------------------------------------------------------- misc
    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, agent_id: object) -> bool:
        return agent_id in self._agents

    def __iter__(self) -> Iterator[AgentInfo]:  # pragma: no cover - trivial
        return iter(builtins.list(self._agents.values()))
