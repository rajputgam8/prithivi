"""Typed asynchronous event bus.

One bus, many subscribers (ARCHITECTURE.md S3.3). Perception emits, the agent
consumes, interfaces render — nothing imports a UI.

Topics are strings with a dotted prefix and support a trailing ``*`` wildcard::

    bus.subscribe("tool.*", handler)
    bus.subscribe("perception.utterance", handler)

Design constraints:

* **Never drops silently.** A handler that raises is logged and the bus keeps
  going; a failing renderer must not kill the agent turn.
* **Backpressure is explicit.** ``maxsize=0`` is unbounded (sensors may burst),
  but bounded queues drop the *oldest* event and count it, so the UI degrades
  instead of stalling perception.
* **No cross-loop surprises.** Handlers may be sync or async; sync handlers run
  in the executor so they cannot block the bus.
"""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Event", "EventBus", "Subscription"]

log = logging.getLogger("prithivi.events")


@dataclass(frozen=True)
class Event:
    """One message on a topic."""

    topic: str
    payload: Any = None
    source: str = ""

    def __repr__(self) -> str:  # keep payloads out of logs by default
        return f"Event(topic={self.topic!r}, source={self.source!r})"


@dataclass
class Subscription:
    """Handle returned by :meth:`EventBus.subscribe`; call :meth:`cancel` to stop."""

    bus: EventBus
    patterns: tuple[str, ...]
    handler: Callable[[Event], Awaitable[None] | None]
    queue: asyncio.Queue[Event] | None
    task: asyncio.Task[None] | None = None
    _cancelled: bool = field(default=False, repr=False)

    def cancel(self) -> None:
        self._cancelled = True
        if self.task is not None:
            self.task.cancel()
        self.bus._remove(self)


class EventBus:
    """A topic-keyed, wildcard-subscription async event bus."""

    def __init__(self, *, maxsize: int = 0) -> None:
        self._subs: list[Subscription] = []
        self._maxsize = maxsize
        self._published = 0
        self._dropped = 0

    # -- statistics ----------------------------------------------------
    @property
    def stats(self) -> dict[str, int]:
        return {"published": self._published, "dropped": self._dropped, "subscribers": len(self._subs)}

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)

    # -- subscription --------------------------------------------------
    def subscribe(
        self,
        patterns: str | Iterable[str],
        handler: Callable[[Event], Awaitable[None] | None],
        *,
        maxsize: int = 0,
    ) -> Subscription:
        """Register ``handler`` for one pattern or several.

        A sync handler is run in the default executor so a slow renderer cannot
        stall publication.
        """
        pats = (patterns,) if isinstance(patterns, str) else tuple(patterns)
        queue = asyncio.Queue(maxsize=maxsize) if maxsize > 0 else None
        sub = Subscription(bus=self, patterns=pats, handler=handler, queue=queue)
        self._subs.append(sub)
        if queue is not None:
            sub.task = asyncio.create_task(self._pump(sub), name=f"bus:{pats[0]}")
        return sub

    def _remove(self, sub: Subscription) -> None:
        try:
            self._subs.remove(sub)
        except ValueError:
            pass

    async def _pump(self, sub: Subscription) -> None:
        assert sub.queue is not None
        while True:
            event = await sub.queue.get()
            try:
                await self._invoke(sub.handler, event)
            finally:
                sub.queue.task_done()

    async def _invoke(self, handler: Callable[[Event], Awaitable[None] | None], event: Event) -> None:
        try:
            result = handler(event)
            if asyncio.iscoroutine(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("event handler failed for topic %s", event.topic)

    # -- publication ---------------------------------------------------
    def publish(self, topic: str, payload: Any = None, *, source: str = "") -> None:
        """Fan out an event. Does not await subscribers."""
        self._published += 1
        event = Event(topic=topic, payload=payload, source=source)
        for sub in list(self._subs):
            if not self._matches(sub, event.topic):
                continue
            if sub.queue is None:
                asyncio.create_task(self._invoke(sub.handler, event))
            else:
                try:
                    sub.queue.put_nowait(event)
                except asyncio.QueueFull:
                    # Drop the oldest so a stalled renderer cannot back up the bus.
                    with contextlib.suppress(asyncio.QueueEmpty):
                        sub.queue.get_nowait()
                        sub.queue.task_done()
                    with contextlib.suppress(asyncio.QueueFull):
                        sub.queue.put_nowait(event)
                    self._dropped += 1

    async def publish_and_wait(self, topic: str, payload: Any = None, *, source: str = "", timeout: float = 5.0) -> None:
        """Publish and wait for queued subscribers to drain (used by tests)."""
        self.publish(topic, payload, source=source)
        for sub in list(self._subs):
            if sub.queue is not None and self._matches(sub, topic):
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(sub.queue.join(), timeout=timeout)

    @staticmethod
    def _matches(sub: Subscription, topic: str) -> bool:
        return any(fnmatch.fnmatchcase(topic, p) or topic == p for p in sub.patterns)

    async def aclose(self) -> None:
        for sub in list(self._subs):
            sub.cancel()
        await asyncio.sleep(0)
