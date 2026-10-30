"""Phase 7: durable memory, knowledge ingestion, and their tools.

These are integration tests on purpose: the failure modes that matter (a memory
that is stored but not recallable, a deleted document whose chunks still answer
questions, a re-ingested file that keeps its old text) only appear when the
repository, vector index and tool layer are exercised together.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prithivi.core.agent import Agent
from prithivi.core.config import Settings
from prithivi.core.state import TurnSource
from prithivi.db.connection import Database
from prithivi.db.repositories.sessions import SessionRepository
from prithivi.db.repositories.settings import SettingsRepository
from prithivi.llm.provider import Completion
from prithivi.llm.scripted import ScriptedProvider
from prithivi.memory.long_term import MemoryStore
from prithivi.rag.knowledge_base import KnowledgeBase
from prithivi.security.confirmation import ConfirmationBroker
from prithivi.security.permissions import PermissionEngine
from prithivi.tools import build_tool_specs
from prithivi.tools.base import ToolContext
from prithivi.tools.registry import ToolRegistry


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "prithivi.db")
    database.migrate()
    return database


@pytest.fixture
def memory(db: Database) -> MemoryStore:
    return MemoryStore(db)


@pytest.fixture
def knowledge(db: Database) -> KnowledgeBase:
    return KnowledgeBase(db, top_k=5)


def test_memory_round_trip_and_disable(memory: MemoryStore) -> None:
    memory.remember(key="preference.python", text="I prefer Python", type="preference", pinned=True)

    hits = memory.recall("what language do I prefer")
    assert hits
    assert hits[0].record.key == "preference.python"

    memory.set_enabled(False)
    assert memory.recall("python") == []
    with pytest.raises(RuntimeError):
        memory.remember(key="preference.go", text="I prefer Go")

    memory.set_enabled(True)
    assert memory.get("preference.python") is not None


def test_forget_removes_row_and_vector(memory: MemoryStore) -> None:
    memory.remember(key="note.vault", text="The deploy key is in the vault")
    assert memory.vectors.stats()["total"] == 1

    assert memory.forget("note.vault") is True
    assert memory.get("note.vault") is None
    assert memory.vectors.stats()["total"] == 0
    assert memory.recall("deploy key") == []


def test_forget_all_keeps_pinned_unless_asked(memory: MemoryStore) -> None:
    memory.remember(key="profile.name", text="My name is Ada", type="profile", pinned=True)
    memory.remember(key="note.tmp", text="a temporary note")

    removed = memory.forget_all()
    assert removed == 1
    assert memory.get("profile.name") is not None
    assert memory.get("note.tmp") is None


def test_ingest_search_cite_and_delete(knowledge: KnowledgeBase) -> None:
    document = knowledge.add_text(
        "The launch code for the greenhouse is ORCHID-42. "
        "Only the duty gardener may read it.",
        uri="note://greenhouse",
        title="Greenhouse",
    )
    assert document.chunk_count >= 1
    assert document.status == "indexed"

    hits = knowledge.search("what is the greenhouse launch code")
    assert hits
    assert any("ORCHID-42" in hit.text for hit in hits)
    assert "note://greenhouse#L" in hits[0].citation()

    assert knowledge.delete(document.document_id) is True
    assert knowledge.search("greenhouse launch code") == []
    assert knowledge.vectors.stats()["total"] == 0


def test_reingest_same_uri_replaces_old_text(knowledge: KnowledgeBase) -> None:
    first = knowledge.add_text("The password is ALPHA.", uri="note://rotating")
    assert knowledge.search("ALPHA")

    second = knowledge.add_text("The password is BETA.", uri="note://rotating")
    assert second.document_id == first.document_id
    assert not second.duplicate

    hits = knowledge.search("password")
    joined = "\n".join(hit.text for hit in hits)
    assert "BETA" in joined
    assert "ALPHA" not in joined
    assert knowledge.vectors.stats()["total"] == len(
        knowledge.ingestor.repo.list_chunks(first.document_id)
    )


async def test_memory_tools_are_registered_and_usable(db: Database) -> None:
    settings = Settings.for_testing()
    store = MemoryStore(db)
    registry = ToolRegistry()
    registry.register_all(
        build_tool_specs(include_web=False, include_memory=True, include_knowledge=False)
    )
    assert {"remember_memory", "recall_memory", "forget_memory"} <= set(registry.names)

    permissions = PermissionEngine(default_level=2)
    ctx = ToolContext(settings=settings, memory=store)
    decision = permissions.check(scope="remember_memory:*", level=1)

    remembered = await registry.execute(
        "remember_memory",
        {"text": "I prefer Python"},
        decision,
        ctx,
        source=TurnSource.CLI,
    )
    assert remembered.success
    assert remembered.data["key"] == "preference.python"
    assert store.get("preference.python") is not None

    recalled = await registry.execute(
        "recall_memory",
        {"query": "which language do I prefer"},
        permissions.check(scope="recall_memory:*", level=0),
        ctx,
        source=TurnSource.CLI,
    )
    assert recalled.success
    assert "Python" in recalled.data["result"]

    forgotten = await registry.execute(
        "forget_memory",
        {"key": "preference.python"},
        permissions.check(scope="forget_memory:*", level=1),
        ctx,
        source=TurnSource.CLI,
    )
    assert forgotten.success
    assert store.get("preference.python") is None


async def test_knowledge_tools_mark_context_untrusted(db: Database) -> None:
    settings = Settings.for_testing()
    kb = KnowledgeBase(db)
    kb.add_text("The vault code is FERN-7.", uri="note://vault", title="Vault")
    registry = ToolRegistry()
    registry.register_all(
        build_tool_specs(include_web=False, include_memory=False, include_knowledge=True)
    )
    permissions = PermissionEngine(default_level=2)
    ctx = ToolContext(settings=settings, knowledge=kb)

    result = await registry.execute(
        "search_knowledge",
        {"query": "vault code"},
        permissions.check(scope="search_knowledge:*", level=0),
        ctx,
        source=TurnSource.CLI,
    )
    assert result.success
    assert result.untrusted_source == "knowledge base"
    assert "FERN-7" in result.data["result"]
    assert "note://vault#L" in result.data["citations"][0]


async def test_agent_recalls_memory_and_persists_turn(db: Database) -> None:
    settings = Settings.for_testing()
    store = MemoryStore(db)
    store.remember(key="preference.python", text="I prefer Python", type="preference", pinned=True)
    sessions = SessionRepository(db)
    registry = ToolRegistry()
    registry.register_all(build_tool_specs(include_web=False))
    provider = ScriptedProvider([Completion(text="I remember you prefer Python.")])
    agent = Agent(
        provider=provider,
        registry=registry,
        permissions=PermissionEngine(default_level=2),
        confirmations=ConfirmationBroker(None),
        settings=settings,
        memory=store,
        sessions=sessions,
    )

    turn = await agent.run("which language do I prefer?", source=TurnSource.CLI)
    assert turn.succeeded
    # The recalled memory is part of the prompt the model actually saw.
    prompt = "\n".join(message.content or "" for message in provider.calls[0][0])
    assert "I prefer Python" in prompt
    assert any(chip.startswith("from what I remember") for chip in turn.narration.chips)

    rows = sessions.list_messages(agent.session.session_id)
    assert [row.role for row in rows] == ["user", "assistant"]
    assert rows[1].content == turn.text


async def test_agent_auto_cites_knowledge_context(db: Database) -> None:
    settings = Settings.for_testing()
    kb = KnowledgeBase(db)
    kb.add_text("The backup drive is called HERON.", uri="note://backup", title="Backup")
    registry = ToolRegistry()
    registry.register_all(build_tool_specs(include_web=False, include_memory=False))
    provider = ScriptedProvider([Completion(text="The backup drive is HERON [1].")])
    agent = Agent(
        provider=provider,
        registry=registry,
        permissions=PermissionEngine(default_level=2),
        settings=settings,
        knowledge=kb,
    )

    turn = await agent.run("what is the backup drive called?", source=TurnSource.CLI)
    assert turn.succeeded
    prompt = "\n".join(message.content or "" for message in provider.calls[0][0])
    assert "HERON" in prompt
    assert "untrusted" in prompt.lower()
    assert any(chip.startswith("from your saved notes") for chip in turn.narration.chips)


async def test_memory_toggle_persists_in_settings(db: Database) -> None:
    store = MemoryStore(db, enabled=False)
    repo = SettingsRepository(db)
    repo.set("memory.enabled", False)
    assert repo.get("memory.enabled") is False
    assert store.enabled is False
    store.set_enabled(True)
    repo.set("memory.enabled", True)
    assert repo.get("memory.enabled") is True


def test_exported_tool_order_is_stable() -> None:
    first = [spec.name for spec in build_tool_specs(include_web=False)]
    second = [spec.name for spec in build_tool_specs(include_web=False)]
    assert first == second
    assert {"recall_memory", "remember_memory", "search_knowledge"} <= set(first)
