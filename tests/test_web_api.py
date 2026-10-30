"""The local web interface: auth, streaming, and the Phase 7 management API.

The server is a real ``ThreadingHTTPServer`` on an ephemeral loopback port, and
requests go through ``http.client``, so the test covers the actual handler
threads and the hand-off to the asyncio loop that owns the agent.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from prithivi.api.server import WebApp, make_handler
from prithivi.core.agent import Agent
from prithivi.core.config import Settings
from prithivi.core.events import EventBus
from prithivi.core.narrator import Narrator
from prithivi.core.router import Router
from prithivi.llm.provider import Completion
from prithivi.llm.scripted import ScriptedProvider
from prithivi.security.audit import AuditLog
from prithivi.security.confirmation import ConfirmationBroker, ConfirmationRequest
from prithivi.security.permissions import PermissionEngine
from prithivi.tools import build_tool_specs
from prithivi.tools.registry import ToolRegistry


class WebHarness:
    def __init__(self, server: object, app: WebApp, port: int) -> None:
        self.server = server
        self.app = app
        self.port = port

    def request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        *,
        token: str | None = "__default__",
        timeout: float = 20.0,
    ) -> tuple[int, bytes, str]:
        headers = {}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if token == "__default__":
            token = self.app.token
        if token:
            headers["Authorization"] = f"Bearer {token}"
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            data = response.read()
            return response.status, data, response.getheader("Content-Type") or ""
        finally:
            connection.close()

    def json(self, method: str, path: str, payload: dict | None = None, **kwargs):
        status, data, _ = self.request(method, path, payload, **kwargs)
        return status, json.loads(data.decode("utf-8"))


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[WebHarness]:
    settings = Settings.for_testing(data_dir=tmp_path)
    provider = ScriptedProvider([Completion(text="Hello from the web test.")])
    audit = AuditLog(":memory:")
    registry = ToolRegistry(audit=audit)
    registry.register_all(build_tool_specs(include_web=False))
    agent = Agent(
        provider=provider,
        registry=registry,
        permissions=PermissionEngine(default_level=2),
        confirmations=ConfirmationBroker(None),
        audit=audit,
        settings=settings,
        bus=EventBus(),
        narrator=Narrator(),
        router=Router(provider=provider, app_names=registry.names),
    )
    runtime = type(
        "Runtime",
        (),
        {
            "settings": settings,
            "agent": agent,
            "audit": audit,
            "confirmations": agent.confirmations,
            "memory": None,
            "knowledge": None,
            "sessions": None,
            "settings_repo": None,
        },
    )()

    # Phase 7 services, wired the same way main.Runtime does it.
    from prithivi.db.connection import Database
    from prithivi.db.repositories.sessions import SessionRepository
    from prithivi.memory.long_term import MemoryStore
    from prithivi.rag.knowledge_base import KnowledgeBase

    db = Database(settings.db_path)
    db.migrate()
    runtime.db = db
    runtime.memory = MemoryStore(db)
    runtime.knowledge = KnowledgeBase(db)
    runtime.sessions = SessionRepository(db)
    agent.memory = runtime.memory
    agent.knowledge = runtime.knowledge
    agent.sessions = runtime.sessions

    # The web app captures the loop it is built on. Run that loop on its own
    # thread so synchronous HTTP calls in the tests cannot deadlock it.
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    async def make_app() -> WebApp:
        return WebApp(runtime, token="test-token")

    app = asyncio.run_coroutine_threadsafe(make_app(), loop).result(timeout=10)

    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        yield WebHarness(server, app, server.server_port)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=3)
        db.close()

def test_static_ui_and_auth(harness: WebHarness) -> None:
    status, body, content_type = harness.request("GET", "/", token=None)
    assert status == 200
    assert "Prithivi" in body.decode("utf-8")
    assert content_type.startswith("text/html")

    status, _, _ = harness.request("GET", "/api/status", token=None)
    assert status == 401

    status, payload = harness.json("GET", "/api/status")
    assert status == 200
    assert payload["provider"] == "openai_compatible"
    assert payload["state"] == "idle"


def test_chat_streams_and_persists(harness: WebHarness) -> None:
    status, body, content_type = harness.request(
        "POST", "/api/chat", {"message": "hello"}
    )
    assert status == 200
    assert content_type.startswith("text/event-stream")
    text = body.decode("utf-8")
    assert "event: final" in text
    assert "Hello from the web test." in text
    assert "event: done" in text

    status, payload = harness.json("GET", "/api/sessions")
    assert status == 200
    assert payload["sessions"]
    assert payload["sessions"][0]["id"] == harness.app.runtime.agent.session.session_id


def test_memory_crud_over_http(harness: WebHarness) -> None:
    status, payload = harness.json("POST", "/api/memory", {"text": "I prefer Python"})
    assert status == 201
    assert payload["key"] == "preference.python"
    assert payload["pinned"] is True

    status, listing = harness.json("GET", "/api/memory")
    assert status == 200
    assert listing["enabled"] is True
    assert any(item["key"] == "preference.python" for item in listing["memories"])

    status, result = harness.json("DELETE", "/api/memory/preference.python")
    assert status == 200
    assert result["deleted"] is True

    status, listing = harness.json("GET", "/api/memory")
    assert listing["memories"] == []


def test_memory_toggle_removes_tools(harness: WebHarness) -> None:
    status, _ = harness.json("POST", "/api/memory/toggle", {"enabled": False})
    assert status == 200
    assert harness.app.runtime.agent.registry.get("remember_memory") is None
    assert harness.app.runtime.memory.enabled is False

    status, _ = harness.json("POST", "/api/memory/toggle", {"enabled": True})
    assert status == 200
    assert harness.app.runtime.agent.registry.get("remember_memory") is not None


def test_knowledge_ingest_search_and_delete(harness: WebHarness) -> None:
    status, document = harness.json(
        "POST",
        "/api/knowledge/ingest",
        {"text": "The reserve key is KESTREL-9.", "title": "Reserve"},
    )
    assert status == 201
    assert document["chunk_count"] >= 1

    hits = harness.app.runtime.knowledge.search("reserve key")
    assert any("KESTREL-9" in hit.text for hit in hits)

    status, listing = harness.json("GET", "/api/knowledge")
    assert status == 200
    assert any(item["id"] == document["document_id"] for item in listing["documents"])

    status, result = harness.json("DELETE", f"/api/knowledge/{document['document_id']}")
    assert status == 200
    assert result["deleted"] is True
    assert harness.app.runtime.knowledge.search("reserve key") == []


def test_unknown_endpoint_is_404(harness: WebHarness) -> None:
    status, _, _ = harness.request("GET", "/api/nope")
    assert status == 404


async def test_web_confirmation_bridge_requires_a_real_answer() -> None:
    """A confirmation is pending until the browser answers with its token."""
    import datetime as dt
    import queue as queue_module

    from prithivi.api.server import WebConfirmationHandler

    bridge = WebConfirmationHandler()
    outbox: queue_module.Queue = queue_module.Queue()
    bridge.attach(outbox)
    now = dt.datetime.now(dt.UTC)
    request = ConfirmationRequest(
        token="tok-1",
        tool="ingest_document",
        fingerprint="f",
        level=2,
        consequence="Index a file",
        args_preview="{}",
        created_at=now,
        expires_at=now + dt.timedelta(seconds=60),
    )

    task = asyncio.create_task(bridge(request))
    await asyncio.sleep(0.05)
    event = outbox.get_nowait()
    assert event["event"] == "confirm"
    assert bridge.resolve("wrong-token", approve=True) is False
    assert bridge.resolve("tok-1", approve=True) is True
    outcome = await task
    assert outcome.approved

    # An unanswered request is a decline, not an approval.
    bridge2 = WebConfirmationHandler()
    bridge2.attach(queue_module.Queue())
    request2 = ConfirmationRequest(
        token="tok-2",
        tool="x",
        fingerprint="f",
        level=4,
        consequence="critical",
        args_preview="{}",
        created_at=now - dt.timedelta(seconds=5),
        expires_at=now - dt.timedelta(seconds=1),
    )
    declined = await bridge2(request2)
    assert declined.approved is False
    assert request2.typed_phrase == ""  # minted only by the broker
