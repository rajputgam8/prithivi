"""The local web interface (Phase 10).

Stdlib only: ``http.server`` and a hand-written JSON/SSE layer, because the
target device is a phone in Termux and pulling in a full ASGI stack would cost
more RAM than the assistant itself.

The server binds to loopback by default and requires a bearer token for every
API call. It is a *view* over the same agent, permissions, audit log and memory
as the terminal — never a second, weaker path.
"""

from .server import WebApp, serve

__all__ = ["WebApp", "serve"]
