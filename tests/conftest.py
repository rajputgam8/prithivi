"""Shared fixtures.

Tests never touch the network or the real data directory: the LLM is scripted,
the audit log is in-memory, and settings are the offline defaults.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from prithivi.core.agent import Agent
from prithivi.core.config import Settings
from prithivi.core.events import EventBus
from prithivi.core.narrator import Narrator
from prithivi.llm.provider import Completion
from prithivi.llm.scripted import ScriptedProvider
from prithivi.security.audit import AuditLog
from prithivi.security.confirmation import ConfirmationBroker
from prithivi.security.permissions import PermissionEngine
from prithivi.tools.basic import build_registry
from prithivi.tools.registry import ToolRegistry


@pytest.fixture
def settings() -> Settings:
    return Settings.for_testing()


@pytest.fixture
def audit() -> AuditLog:
    return AuditLog(":memory:")


@pytest.fixture
def registry(audit: AuditLog) -> ToolRegistry:
    reg = ToolRegistry(audit=audit)
    reg.register_all(build_registry())
    return reg


@pytest.fixture
def permissions() -> PermissionEngine:
    return PermissionEngine(default_level=2)


@pytest.fixture
def confirmations() -> ConfirmationBroker:
    """Declines everything unless a test supplies its own handler."""
    async def decline(request):
        from prithivi.security.confirmation import ConfirmationOutcome

        return ConfirmationOutcome.decline("no handler in tests")

    return ConfirmationBroker(decline)


@pytest.fixture
def agent(
    registry: ToolRegistry,
    permissions: PermissionEngine,
    confirmations: ConfirmationBroker,
    audit: AuditLog,
    settings: Settings,
) -> Agent:
    return Agent(
        provider=ScriptedProvider([]),
        registry=registry,
        permissions=permissions,
        confirmations=confirmations,
        audit=audit,
        settings=settings,
        bus=EventBus(),
        narrator=Narrator(),
    )


@pytest.fixture
def scripted() -> Iterator[ScriptedProvider]:
    """A provider whose replies the test fills in before running the agent."""
    provider = ScriptedProvider([])
    yield provider


def completions(*items: Completion) -> ScriptedProvider:
    """Convenience: a provider pre-loaded with ``items``."""
    return ScriptedProvider(list(items))