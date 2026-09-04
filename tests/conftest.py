from __future__ import annotations

import pytest

from agentcomm import CommunicationLayer


@pytest.fixture
def layer() -> CommunicationLayer:
    return CommunicationLayer(default_timeout=2.0)
