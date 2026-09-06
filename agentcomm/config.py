"""Minimal ``.env`` loading and typed settings access (no third-party deps).

API keys are *never* hard-coded. They are read from the process environment,
optionally pre-populated from a ``.env`` file that is git-ignored.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger("agentcomm.config")


def load_dotenv(path: str | os.PathLike[str] = ".env", *, override: bool = False) -> int:
    """Load ``KEY=VALUE`` lines from *path* into ``os.environ``.

    Returns the number of variables set. Missing file is not an error.
    """
    p = Path(path)
    if not p.is_file():
        return 0
    count = 0
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        line = line.removeprefix("export ")
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and (override or key not in os.environ):
            os.environ[key] = value
            count += 1
    _log.debug("loaded %d variables from %s", count, p)
    return count


def get_env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def require_env(name: str) -> str:
    value = get_env(name)
    if value is None:
        raise KeyError(f"required environment variable {name!r} is not set")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    default_timeout: float = 30.0
    log_level: str = "INFO"
    history_path: str | None = None

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            default_timeout=float(get_env("AGENTCOMM_DEFAULT_TIMEOUT", "30") or "30"),
            log_level=(get_env("AGENTCOMM_LOG_LEVEL", "INFO") or "INFO").upper(),
            history_path=get_env("AGENTCOMM_HISTORY_PATH"),
        )


def configure_logging(level: str | int | None = None) -> None:
    """Configure a sane default logger for the ``agentcomm`` namespace."""
    if level is None:
        level = Settings.from_env().log_level
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
