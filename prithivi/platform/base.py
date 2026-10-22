"""Platform abstraction.

All OS access lives behind :class:`PlatformAdapter` (ARCHITECTURE.md S5). This is
the one deliberate exception to the import rules, and it is what keeps
``tools/`` free of ``pm``/``osascript``/PowerShell specifics.

Adapters **probe their own capabilities at construction** and expose them, so a
tool can return ``BACKEND_UNAVAILABLE`` with a precise reason instead of failing
later with an opaque subprocess error.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

__all__ = ["PlatformAdapter", "PlatformCapabilities", "PlatformError"]


class PlatformError(RuntimeError):
    """An OS-level operation failed. Message must be user-readable."""


@dataclass(frozen=True)
class PlatformCapabilities:
    """What this device can actually do right now.

    Attributes:
        open_urls: can hand a URL to the system browser.
        launch_apps: can start an installed application by package.
        verify_foreground: can confirm an app actually reached the screen
            (requires a privileged permission most sandboxes lack).
        speak / listen: TTS and microphone capture.
        battery / sensors: hardware telemetry.
        notes: human-readable gaps, printed by ``prithivi doctor``.
    """

    open_urls: bool = False
    launch_apps: bool = False
    verify_foreground: bool = False
    speak: bool = False
    listen: bool = False
    battery: bool = False
    sensors: bool = False
    os_name: str = "unknown"
    notes: tuple[str, ...] = ()

    def missing(self, feature: str) -> str:
        """Explain in one sentence why ``feature`` is unavailable."""
        return (
            f"{feature} is unavailable on this device. "
            + ("; ".join(self.notes) if self.notes else "")
        ).strip()


class PlatformAdapter(abc.ABC):
    """Base class for OS-specific behaviour."""

    name: str = "abstract"

    def __init__(self) -> None:
        self._capabilities: PlatformCapabilities | None = None

    @property
    def capabilities(self) -> PlatformCapabilities:
        if self._capabilities is None:
            self._capabilities = self.probe()
        return self._capabilities

    @abc.abstractmethod
    def probe(self) -> PlatformCapabilities:
        """Detect what is available. Must be cheap and must not raise."""

    @abc.abstractmethod
    def open_url(self, url: str, *, package: str | None = None) -> dict[str, object]:
        """Open ``url`` in a browser (optionally a specific one)."""

    @abc.abstractmethod
    def launch_app(self, package: str, *, component: str | None = None) -> dict[str, object]:
        """Start an installed app."""

    def installed_packages(self) -> tuple[str, ...]:
        """Package names currently installed, best effort."""
        return ()

    def is_installed(self, package: str) -> bool:
        return package in self.installed_packages()

    def resolve_launcher(self, package: str) -> str | None:
        """The ``pkg/activity`` component that starts ``package``, if resolvable."""
        return None

    def describe(self) -> str:
        caps = self.capabilities
        flags = [
            ("urls", caps.open_urls),
            ("apps", caps.launch_apps),
            ("tts", caps.speak),
            ("stt", caps.listen),
        ]
        on = [n for n, ok in flags if ok]
        return f"{self.name}: {'+'.join(on) if on else 'no integrations'}"
