"""Desktop platform adapter (Linux/macOS/Windows).

Included so the project runs and is testable on a normal computer as well as on
the phone. It shells out to allowlisted platform binaries with argument arrays —
never a shell string (SECURITY.md S9).
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess

from .base import PlatformAdapter, PlatformCapabilities, PlatformError

__all__ = ["DesktopAdapter"]

#: Friendly app name -> (linux binary, macOS binary, windows binary).
_APP_BINARIES: dict[str, tuple[str, str, str]] = {
    "chrome": ("google-chrome", "Google Chrome", "chrome"),
    "firefox": ("firefox", "Firefox", "firefox"),
    "spotify": ("spotify", "Spotify", "spotify"),
    "terminal": ("gnome-terminal", "Terminal", "wt"),
    "files": ("nautilus", "Finder", "explorer"),
    "calculator": ("gnome-calculator", "Calculator", "calc"),
    "code": ("code", "Visual Studio Code", "Code"),
}


class DesktopAdapter(PlatformAdapter):
    """Best-effort desktop integration."""

    name = "desktop"

    def probe(self) -> PlatformCapabilities:
        notes: list[str] = []
        has_xdg = shutil.which("xdg-open") is not None
        has_open = shutil.which("open") is not None
        start = shutil.which("start") is not None or platform.system() == "Windows"
        open_urls = has_xdg or has_open or start
        launch = any(
            shutil.which(binary)
            for binaries in _APP_BINARIES.values()
            for binary in (binaries[0] if platform.system() == "Linux" else binaries[1],)
        )
        if not open_urls:
            notes.append("No xdg-open/open available.")
        if not os.environ.get("DISPLAY") and platform.system() == "Linux":
            notes.append("No DISPLAY set: this is a headless session, so GUI apps cannot be seen.")
        return PlatformCapabilities(
            open_urls=open_urls,
            launch_apps=launch,
            verify_foreground=False,
            speak=shutil.which("espeak") is not None or platform.system() == "Darwin",
            listen=bool(os.environ.get("PULSE_SERVER") or os.environ.get("ALSA_CARD")),
            battery=shutil.which("upower") is not None,
            sensors=False,
            os_name=f"{platform.system()} {platform.release()}",
            notes=tuple(notes),
        )

    def open_url(self, url: str, *, package: str | None = None) -> dict[str, object]:
        opener = {"Darwin": "open", "Windows": "start"}.get(platform.system(), "xdg-open")
        if shutil.which(opener) is None and platform.system() == "Linux":
            raise PlatformError("xdg-open is not installed")
        try:
            result = subprocess.run([opener, url], capture_output=True, text=True, timeout=20, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            raise PlatformError(str(exc)) from exc
        if result.returncode != 0:
            raise PlatformError(f"{opener} failed: {(result.stderr or result.stdout)[:200]}")
        return {"ok": True, "method": opener, "url": url, "foreground_verified": False}

    def launch_app(self, package: str, *, component: str | None = None) -> dict[str, object]:
        key = package.lower().strip()
        binaries = _APP_BINARIES.get(key)
        if binaries is None:
            raise PlatformError(f"I do not know how to launch {package!r} on {platform.system()}")
        index = {"Linux": 0, "Darwin": 1, "Windows": 2}[platform.system()]
        binary = binaries[index]
        if shutil.which(binary) is None:
            raise PlatformError(f"{binary} is not installed")
        try:
            subprocess.Popen([binary], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError) as exc:
            raise PlatformError(str(exc)) from exc
        return {"ok": True, "method": binary, "package": key, "foreground_verified": False}

    def installed_packages(self) -> tuple[str, ...]:
        return tuple(_APP_BINARIES)
