"""Loopback HTTP API and browser UI for Prithivi.

Design constraints that shaped this file:

* **Stdlib only.** ``http.server`` on a background thread, JSON in and out, and
  server-sent events for the chat stream. No ASGI framework, no websockets, no
  build step.
* **One agent, one loop.** The HTTP server runs in threads but every call that
  touches the agent is submitted to the asyncio loop that built the runtime, so
  the provider's connection pool and the agent's single-flight assumption hold.
* **No weaker path.** The UI goes through the same ``Agent``, permission engine,
  confirmation broker and audit log as the terminal. A confirmation in the
  browser is a real second factor with the same token binding; it cannot approve
  a different action than the one shown.
* **Loopback by default.** Binding to a routable address requires an explicit
  opt-in *and* a pre-configured token, so a phone on a café Wi-Fi is not running
  an unauthenticated control panel.
"""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import queue
import secrets
import threading
import time
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from ..core.state import Modality, TurnSource
from ..observability.logging import get_logger
from ..security.confirmation import ConfirmationOutcome, ConfirmationRequest
from ..tools.memory_tools import build_memory_registry

__all__ = ["WebApp", "serve"]

log = get_logger("prithivi.api")

_STATIC = Path(__file__).with_name("static")
_STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/bg.js": ("bg.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}
_MAX_BODY = 2 * 1024 * 1024
_LOOPBACK = {"127.0.0.1", "localhost", "::1", "::ffff:127.0.0.1"}


def _is_loopback(host: str) -> bool:
    if host in _LOOPBACK:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class WebConfirmationHandler:
    """A confirmation interface that speaks over the open chat stream.

    The broker calls this while the agent is suspended. We push a ``confirm``
    event to the browser, then await a future resolved by ``POST /api/confirm``.
    The token in that response is the broker's own token, so approval stays bound
    to the exact tool call — the UI cannot approve a later action with an old
    answer.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[dict[str, Any]] | None = None
        self._pending: dict[str, asyncio.Future[ConfirmationOutcome]] = {}

    def attach(self, outbox: queue.Queue[dict[str, Any]] | None) -> None:
        self._queue = outbox

    async def __call__(self, request: ConfirmationRequest) -> ConfirmationOutcome:
        if self._queue is None:
            return ConfirmationOutcome.decline("no browser window is listening")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ConfirmationOutcome] = loop.create_future()
        self._pending[request.token] = future
        self._queue.put({"event": "confirm", "data": request.to_dict()})
        import datetime as dt

        seconds = max(1.0, (request.expires_at - dt.datetime.now(dt.UTC)).total_seconds())
        try:
            return await asyncio.wait_for(future, timeout=seconds)
        except TimeoutError:
            return ConfirmationOutcome.decline("confirmation timed out waiting for the browser")
        finally:
            self._pending.pop(request.token, None)

    def resolve(self, token: str, *, approve: bool, phrase: str = "") -> bool:
        future = self._pending.get(token)
        if future is None or future.done():
            return False
        if approve:
            future.set_result(ConfirmationOutcome.approve(factor="ui", typed_phrase=phrase or None))
        else:
            future.set_result(ConfirmationOutcome.decline("declined in the browser"))
        return True


class WebApp:
    """State shared by every request handler."""

    def __init__(self, runtime: Any, *, token: str) -> None:
        self.runtime = runtime
        self.token = token
        self.loop = asyncio.get_running_loop()
        self.confirmations = WebConfirmationHandler()
        self._turn_lock = asyncio.Lock()
        self._outbox: queue.Queue[dict[str, Any]] | None = None

        runtime.confirmations.set_handler(self.confirmations)

    # -- auth ----------------------------------------------------------
    def authorised(self, header: str | None, query_token: str | None) -> bool:
        supplied = ""
        if header and header.lower().startswith("bearer "):
            supplied = header[7:].strip()
        elif query_token:
            supplied = query_token
        return bool(supplied) and hmac.compare_digest(supplied, self.token)

    # -- chat ----------------------------------------------------------
    def start_turn(self, utterance: str, outbox: queue.Queue[dict[str, Any]]) -> asyncio.Future[Any]:
        """Schedule one turn on the owned loop and return its future."""
        return asyncio.run_coroutine_threadsafe(self._run_turn(utterance, outbox), self.loop)

    async def _run_turn(self, utterance: str, outbox: queue.Queue[dict[str, Any]]) -> None:
        agent = self.runtime.agent
        async with self._turn_lock:
            self._outbox = outbox
            self.confirmations.attach(outbox)
            try:
                history = [
                    (entry["role"], entry["content"])
                    for entry in agent.session.history[-12:]
                    if entry.get("role") in {"user", "assistant"}
                ]

                async def on_text(delta: str) -> None:
                    outbox.put({"event": "delta", "data": {"text": delta}})

                result = await agent.dispatch_deterministic(
                    utterance, source=TurnSource.API
                )
                if result is None:
                    result = await agent.run(
                        utterance,
                        source=TurnSource.API,
                        modality=Modality.API,
                        history=history,
                        on_text=on_text,
                    )
                outbox.put({"event": "final", "data": result.to_dict()})
            except Exception as exc:  # the stream must always end
                log.exception("web turn failed")
                outbox.put({"event": "error", "data": {"message": str(exc)}})
            finally:
                self.confirmations.attach(None)
                self._outbox = None
                outbox.put({"event": "done", "data": {}})

    # -- reads ---------------------------------------------------------
    def status(self) -> dict[str, Any]:
        agent = self.runtime.agent
        settings = self.runtime.settings
        memory = self.runtime.memory
        knowledge = self.runtime.knowledge
        audit_ok = None
        if self.runtime.audit is not None:
            try:
                audit_ok = self.runtime.audit.verify()[0]
            except Exception:  # noqa: BLE001 - status must render even if audit is corrupt
                audit_ok = False
        return {
            "session_id": agent.session.session_id,
            "state": agent.session.state.value,
            "turns": agent.session.turn_count,
            "provider": getattr(settings, "llm_provider", "unknown"),
            "model": getattr(settings, "llm_model", "unknown"),
            "memory": memory.stats() if memory is not None else {"enabled": False, "count": 0},
            "knowledge": knowledge.stats() if knowledge is not None else {},
            "tools": len(agent.registry),
            "audit_ok": audit_ok,
        }

    def memory_list(self) -> dict[str, Any]:
        store = self.runtime.memory
        if store is None:
            return {"enabled": False, "memories": [], "stats": {}}
        return {
            "enabled": store.enabled,
            "memories": [record.to_dict() for record in store.list_memories(limit=200)],
            "stats": store.stats(),
        }

    def knowledge_list(self) -> dict[str, Any]:
        kb = self.runtime.knowledge
        if kb is None:
            return {"documents": [], "stats": {}}
        repo = kb.ingestor.repo
        documents = []
        for document in kb.documents(limit=500):
            item = document.to_dict()
            item["chunk_count"] = len(repo.list_chunks(document.id))
            documents.append(item)
        return {"documents": documents, "stats": kb.stats()}

    def tools(self) -> dict[str, Any]:
        return {"tools": self.runtime.agent.registry.summary()}

    def sessions(self) -> dict[str, Any]:
        repo = self.runtime.sessions
        if repo is None:
            return {"sessions": []}
        return {"sessions": [record.to_dict() for record in repo.list_sessions(limit=30)]}

    # -- writes --------------------------------------------------------
    async def add_memory(self, payload: dict[str, Any]) -> dict[str, Any]:
        store = self.runtime.memory
        if store is None or not store.enabled:
            raise ValueError("memory is disabled")
        text = str(payload.get("text", "")).strip()
        if not text:
            raise ValueError("text is required")
        from ..tools.memory_tools import _classify

        memory_type, derived_key, value = _classify(text)
        key = str(payload.get("key") or derived_key)
        record = store.remember(
            key=key,
            text=text,
            type=str(payload.get("type") or memory_type),
            value=value,
            source="user_stated",
            confidence=1.0,
            pinned=bool(payload.get("pinned", memory_type in {"preference", "profile"})),
        )
        return record.to_dict()

    async def forget_memory(self, key: str) -> bool:
        store = self.runtime.memory
        if store is None:
            return False
        return store.forget(key)

    async def toggle_memory(self, enabled: bool) -> dict[str, Any]:
        store = self.runtime.memory
        if store is None:
            raise ValueError("memory is not available")
        store.set_enabled(enabled)
        if self.runtime.settings_repo is not None:
            self.runtime.settings_repo.set("memory.enabled", enabled)
        registry = self.runtime.agent.registry
        for spec in build_memory_registry():
            if enabled and spec.name not in registry:
                registry.register(spec)
            elif not enabled:
                registry.unregister(spec.name)
        return {"enabled": enabled}

    async def ingest(self, payload: dict[str, Any]) -> dict[str, Any]:
        kb = self.runtime.knowledge
        if kb is None:
            raise ValueError("the knowledge base is not available")

        def work() -> Any:
            title = payload.get("title") or None
            text = str(payload.get("text") or "").strip()
            if text:
                return kb.add_text(
                    text,
                    uri=str(payload.get("uri") or f"note://{secrets.token_hex(8)}"),
                    title=title or "Pasted note",
                    source_type="note",
                )
            path = str(payload.get("path") or "").strip()
            if not path:
                raise ValueError("provide a file path or some text")
            return kb.add_file(path, title=title)

        document = await asyncio.to_thread(work)
        result = document.to_dict()
        result["duplicate"] = document.duplicate
        return result

    async def delete_document(self, document_id: str) -> bool:
        kb = self.runtime.knowledge
        if kb is None:
            return False
        return await asyncio.to_thread(kb.delete, document_id)

    async def resolve_confirmation(self, payload: dict[str, Any]) -> bool:
        return self.confirmations.resolve(
            str(payload.get("token") or ""),
            approve=bool(payload.get("approve")),
            phrase=str(payload.get("phrase") or ""),
        )



def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")


def make_handler(app: WebApp) -> type[BaseHTTPRequestHandler]:
    """Bind the app into a request-handler class (stdlib has no DI)."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "Prithivi"
        sys_version = ""
        protocol_version = "HTTP/1.1"

        # -- plumbing --------------------------------------------------
        def log_message(self, fmt: str, *args: Any) -> None:
            log.debug("web %s - %s", self.address_string(), fmt % args)

        def _security_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'none'",
            )
            self.send_header("Cache-Control", "no-store")

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self._security_headers()
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _json(self, status: int, payload: Any) -> None:
            self._send(status, _json_bytes(payload), "application/json; charset=utf-8")

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": message})

        def _body(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                raise ValueError("invalid Content-Length")
            if length <= 0:
                return {}
            if length > _MAX_BODY:
                raise ValueError("request body too large")
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise ValueError("body must be JSON")
            if not isinstance(data, dict):
                raise TypeError("body must be a JSON object")
            return data

        def _auth(self, query: dict[str, list[str]]) -> bool:
            if app.authorised(self.headers.get("Authorization"), query.get("token", [None])[0]):
                return True
            self._error(HTTPStatus.UNAUTHORIZED, "a valid bearer token is required")
            return False

        # -- routes ----------------------------------------------------
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            query = _parse_query(parsed.query)

            if path == "/health":
                self._send(HTTPStatus.OK, b"prithivi ok\n", "text/plain; charset=utf-8")
                return
            if path in _STATIC_FILES:
                self._static(path)
                return
            if path == "/favicon.ico":
                self._send(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
                return
            if not path.startswith("/api/"):
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            if not self._auth(query):
                return

            if path == "/api/status":
                self._json(HTTPStatus.OK, app.status())
            elif path == "/api/memory":
                self._json(HTTPStatus.OK, app.memory_list())
            elif path == "/api/knowledge":
                self._json(HTTPStatus.OK, app.knowledge_list())
            elif path == "/api/tools":
                self._json(HTTPStatus.OK, app.tools())
            elif path == "/api/sessions":
                self._json(HTTPStatus.OK, app.sessions())
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            query = _parse_query(parsed.query)
            if not self._auth(query):
                return
            try:
                payload = self._body()
            except (ValueError, TypeError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return

            if path == "/api/chat":
                self._chat(payload)
                return

            try:
                if path == "/api/confirm":
                    ok = self._call(app.resolve_confirmation(payload))
                    self._json(HTTPStatus.OK, {"resolved": ok})
                elif path == "/api/memory":
                    self._json(HTTPStatus.CREATED, self._call(app.add_memory(payload)))
                elif path == "/api/memory/toggle":
                    self._json(HTTPStatus.OK, self._call(app.toggle_memory(bool(payload.get("enabled")))))
                elif path == "/api/knowledge/ingest":
                    self._json(HTTPStatus.CREATED, self._call(app.ingest(payload)))
                else:
                    self._error(HTTPStatus.NOT_FOUND, "not found")
            except (ValueError, TypeError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
            except Exception as exc:  # report, never leak a traceback
                log.exception("web write failed")
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

        def do_DELETE(self) -> None:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            query = _parse_query(parsed.query)
            if not self._auth(query):
                return
            if path.startswith("/api/memory/"):
                key = path[len("/api/memory/"):]
                self._json(HTTPStatus.OK, {"deleted": self._call(app.forget_memory(key))})
            elif path.startswith("/api/knowledge/"):
                document_id = path[len("/api/knowledge/"):]
                self._json(HTTPStatus.OK, {"deleted": self._call(app.delete_document(document_id))})
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")

        # -- helpers ---------------------------------------------------
        def _call(self, coro: Any) -> Any:
            future = asyncio.run_coroutine_threadsafe(coro, app.loop)
            return future.result(timeout=90)

        def _static(self, path: str) -> None:
            name, content_type = _STATIC_FILES[path]
            try:
                body = (_STATIC / name).read_bytes()
            except OSError:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "static asset missing")
                return
            self._send(HTTPStatus.OK, body, content_type)

        def _chat(self, payload: dict[str, Any]) -> None:
            message = str(payload.get("message") or "").strip()
            if not message:
                self._error(HTTPStatus.BAD_REQUEST, "message is required")
                return
            if len(message) > 16_000:
                self._error(HTTPStatus.BAD_REQUEST, "message is too long")
                return

            outbox: queue.Queue[dict[str, Any]] = queue.Queue()
            future = app.start_turn(message, outbox)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self._security_headers()
            self.end_headers()
            self.close_connection = True

            try:
                while True:
                    try:
                        item = outbox.get(timeout=300)
                    except queue.Empty:
                        self._write_event("error", {"message": "the turn timed out"})
                        break
                    self._write_event(item["event"], item["data"])
                    if item["event"] == "done":
                        break
            except (BrokenPipeError, ConnectionResetError):
                future.cancel()
            finally:
                try:
                    self.wfile.flush()
                except OSError:
                    pass

        def _write_event(self, event: str, data: Any) -> None:
            chunk = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
            self.wfile.write(chunk.encode("utf-8"))
            self.wfile.flush()

    return Handler


def _parse_query(query: str) -> dict[str, list[str]]:
    from urllib.parse import parse_qs

    return parse_qs(query, keep_blank_values=False)


async def serve(settings: Any, runtime: Any, *, ready: Callable[[str], None] | None = None) -> int:
    """Run the web UI until interrupted. Returns a process exit code."""

    host = str(getattr(settings, "api_host", "127.0.0.1"))
    port = int(getattr(settings, "api_port", 8000))
    allow_remote = bool(getattr(settings, "api_allow_remote", False))
    configured = getattr(settings, "api_token", None)
    configured_value = configured.reveal() if configured is not None else ""

    if not _is_loopback(host) and not allow_remote:
        print(
            f"prithivi: refusing to bind {host}:{port}. Set PRITHIVI_API_ALLOW_REMOTE=true "
            "and PRITHIVI_API_TOKEN if you really want a routable interface.",
            flush=True,
        )
        return 2
    if not _is_loopback(host) and not configured_value:
        print(
            "prithivi: a routable bind needs a fixed PRITHIVI_API_TOKEN; refusing to "
            "generate an ephemeral one.",
            flush=True,
        )
        return 2

    token = configured_value or secrets.token_urlsafe(24)
    generated = not bool(configured_value)
    app = WebApp(runtime, token=token)

    try:
        server = ThreadingHTTPServer((host, port), make_handler(app))
    except OSError as exc:
        print(f"prithivi: could not bind {host}:{port}: {exc}", flush=True)
        return 1
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, name="prithivi-web", daemon=True)
    thread.start()

    shown_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{shown_host}:{port}/"
    if generated:
        url += f"#token={token}"
    line = f"Prithivi web UI → {url}"
    if ready is not None:
        ready(line)
    print(line, flush=True)
    if generated:
        print("This token is shown once and lives only in this process.", flush=True)
    elif not _is_loopback(host):
        print("Use PRITHIVI_API_TOKEN as the bearer token.", flush=True)

    started = time.monotonic()
    try:
        while True:
            await asyncio.sleep(3600)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        server.shutdown()
        server.server_close()
        log.info("web server stopped after %.1fs", time.monotonic() - started)
    return 0
