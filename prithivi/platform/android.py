"""Android/Termux platform adapter.

Findings that shaped this file, all verified on the target device:

* ``pm list packages`` works from Termux (uid 10272) and lists 350+ packages.
* ``pm resolve-activity --user 0 --brief -a android.intent.action.MAIN
  -c android.intent.category.LAUNCHER <pkg>`` returns the launcher component.
  **The ``--user 0`` is mandatory** — without it ActivityManager raises
  ``INTERACT_CROSS_USERS`` because the default user is -2.
* ``am start`` / ``cmd activity start-activity`` **cannot** start activities from
  Termux: ActivityTaskManagerService requires the calling package to be
  ``com.android.shell``, which does not belong to a normal app uid.
* ``monkey -p <pkg> -c android.intent.category.LAUNCHER 1`` is the documented
  Termux escape hatch and works from a real Termux session; it needs
  ``/system/bin`` on ``PATH`` because ``monkey`` shells out to ``app_process``.
* ``termux-open-url <url> [package]`` routes a URL to a chosen app, but only when
  the **Termux:API companion app** (``com.termux.api``) is installed. Without it
  the command exits 0 and silently does nothing — which is exactly the kind of
  false success this codebase must never report.
* ``dumpsys`` needs ``android.permission.DUMP``, which Termux lacks, so we cannot
  verify that an app reached the foreground. Evidence therefore reports the
  mechanism used and its exit status, and says plainly that foreground state was
  not verifiable.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from functools import lru_cache

from .base import PlatformAdapter, PlatformCapabilities, PlatformError

__all__ = ["AndroidAdapter"]

log = logging.getLogger("prithivi.platform.android")

#: Termux's own launcher scripts live here; the real Android tools in /system/bin.
_TERMUX_BIN = "/data/data/com.termux/files/usr/bin"
_SYSTEM_BIN = "/system/bin"


def _run(
    argv: list[str],
    *,
    timeout: float = 20.0,
) -> subprocess.CompletedProcess[str]:
    """Run an Android CLI, forcing ``/system/bin`` onto PATH.

    ``monkey`` is a shell script that execs ``app_process``, so it fails with
    "app_process: not found" unless PATH contains ``/system/bin``.

    Callers place ``--user 0`` themselves: it is a *subcommand* option, not a
    global one, so it must follow the subcommand name. Injecting it
    automatically (``pm --user 0 resolve-activity``) is silently rejected.
    """
    env = dict(os.environ)
    path_parts = [_SYSTEM_BIN, _TERMUX_BIN]
    if env.get("PATH"):
        path_parts.append(env["PATH"])
    env["PATH"] = os.pathsep.join(path_parts)

    return subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout, env=env, check=False
    )


class AndroidAdapter(PlatformAdapter):
    """Termux-on-Android adapter with graceful degradation."""

    name = "android"

    def __init__(self) -> None:
        super().__init__()
        self._packages: tuple[str, ...] | None = None

    # -- capability probing -------------------------------------------
    def probe(self) -> PlatformCapabilities:
        from .detect import in_proot

        notes: list[str] = []

        has_termux_api = "com.termux.api" in self._list_packages()
        has_termux_open = shutil.which("termux-open-url", path=_TERMUX_BIN) is not None
        has_monkey = os.path.isfile(f"{_SYSTEM_BIN}/monkey")
        can_resolve = self._can_resolve()
        can_verify = self._can_dump_activities()
        proot = in_proot()

        # Actually handing work to Android needs a bridge to the system: either
        # Termux:API (``termux-open-url``) or a shell tool that reaches the window
        # manager (``monkey``). Resolving a component is not enough on its own.
        #
        # Under PRoot those bridges exist on disk but cannot function: PRoot
        # intercepts the binder/window-manager syscalls they rely on, so
        # ``monkey`` dies with a mangled exec and ``termux-open-url`` raises a
        # uid/``com.android.shell`` SecurityException. Reporting the capability as
        # available (the old ``has_termux_api`` shortcut did) makes every call
        # fail late and cryptically; the probe must own that gap instead.
        bridged = not proot
        open_urls = bridged and has_termux_api
        launch_apps = bridged and (has_monkey or has_termux_api)

        if proot:
            notes.append(
                "Running under PRoot: Android binder/window-manager calls are intercepted, so "
                "apps cannot be launched and the browser cannot be reached. Run Prithivi in "
                "native Termux (`/data/data/com.termux/files/usr/bin/python main.py ...`) for "
                "app control."
            )
        if not has_termux_api:
            notes.append(
                "Termux:API is not installed. Install com.termux.api from F-Droid, or "
                "termux-open-url will silently do nothing."
            )
        if not proot and not (open_urls or launch_apps):
            notes.append("No usable bridge to Android: app launching and URL opening are unavailable.")
        if not can_verify:
            notes.append(
                "Foreground app cannot be verified: 'dumpsys' needs "
                "android.permission.DUMP, which Termux does not have."
            )

        caps = PlatformCapabilities(
            open_urls=open_urls,
            launch_apps=launch_apps,
            verify_foreground=can_verify,
            speak=shutil.which("termux-tts-speak", path=_TERMUX_BIN) is not None,
            listen=shutil.which("termux-microphone-record", path=_TERMUX_BIN) is not None,
            battery=shutil.which("termux-battery-status", path=_TERMUX_BIN) is not None,
            sensors=shutil.which("termux-sensor", path=_TERMUX_BIN) is not None,
            os_name=f"android (termux uid {os.getuid()}, proot={proot})",
            notes=tuple(notes),
        )
        return caps

    def _can_resolve(self) -> bool:
        try:
            result = _run(["pm", "resolve-activity", "--user", "0", "--brief",
                           "-a", "android.intent.action.MAIN",
                           "-c", "android.intent.category.LAUNCHER",
                           "com.android.chrome"], timeout=20.0)
        except (OSError, subprocess.SubprocessError):
            return False
        return "package:" in result.stdout or "/" in result.stdout

    @staticmethod
    def _can_dump_activities() -> bool:
        try:
            result = _run(["dumpsys", "activity", "activities"], timeout=15.0)
        except (OSError, subprocess.SubprocessError):
            return False
        # dumpsys reports the denial on whichever stream it likes; check both.
        combined = (result.stdout + result.stderr)
        return "Permission Denial" not in combined and bool(result.stdout.strip())

    # -- packages ------------------------------------------------------
    @lru_cache(maxsize=1)
    def _list_packages(self) -> frozenset[str]:
        try:
            result = _run(["pm", "list", "packages"], timeout=30.0)
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("cannot list packages: %s", exc)
            return frozenset()
        names = {
            line.split(":", 1)[1].strip()
            for line in result.stdout.splitlines()
            if line.startswith("package:")
        }
        return frozenset(names)

    def installed_packages(self) -> tuple[str, ...]:
        if self._packages is None:
            self._packages = tuple(sorted(self._list_packages()))
        return self._packages

    def is_installed(self, package: str) -> bool:
        return package in self._list_packages()

    def resolve_launcher(self, package: str) -> str | None:
        """``pkg/activity`` for ``package``'s launcher entry, via pm."""
        try:
            result = _run([
                "pm", "resolve-activity", "--user", "0", "--brief",
                "-a", "android.intent.action.MAIN",
                "-c", "android.intent.category.LAUNCHER",
                package,
            ], timeout=20.0)
        except (OSError, subprocess.SubprocessError):
            return None
        for line in reversed(result.stdout.strip().splitlines()):
            candidate = line.strip()
            if "/" in candidate and candidate.split("/")[0] == package:
                return candidate
        return None

    # -- actions -------------------------------------------------------
    def open_url(self, url: str, *, package: str | None = None) -> dict[str, object]:
        """Open ``url``, preferring an explicit browser package."""
        attempts: list[dict[str, object]] = []

        if package:
            argv = [f"{_TERMUX_BIN}/termux-open-url", url, package]
            attempts.append({"mechanism": "termux-open-url", "argv": argv[1:]})
            outcome = self._try(argv, timeout=30.0)
            attempts[-1].update(outcome)
            if outcome["ok"]:
                return {"ok": True, "method": "termux-open-url", "url": url,
                        "package": package, "attempts": attempts,
                        "foreground_verified": False}

        # Let Android pick the default handler.
        try:
            result = _run([
                "pm", "resolve-activity", "--user", "0", "--brief",
                "-a", "android.intent.action.VIEW", "-d", url,
            ], timeout=20.0)
            component = result.stdout.strip().splitlines()[-1].strip() if result.stdout.strip() else ""
        except (OSError, subprocess.SubprocessError) as exc:
            component = ""
            attempts.append({"mechanism": "pm resolve-activity", "error": str(exc)})

        if "/" in component:
            pkg = component.split("/")[0]
            argv = [f"{_TERMUX_BIN}/termux-open-url", url, pkg]
            attempts.append({"mechanism": "termux-open-url", "argv": argv[1:]})
            outcome = self._try(argv, timeout=30.0)
            attempts[-1].update(outcome)
            if outcome["ok"]:
                return {"ok": True, "method": "termux-open-url", "url": url,
                        "package": pkg, "attempts": attempts, "foreground_verified": False}
            raise PlatformError(_explain(attempts, f"could not open {url}"))

        raise PlatformError(_explain(attempts, f"no app on this device handles {url}"))

    def launch_app(self, package: str, *, component: str | None = None) -> dict[str, object]:
        """Start ``package``.

        Cascade, cheapest and most reliable first:

        1. ``monkey`` with the LAUNCHER category — the documented Termux method.
        2. ``termux-open-url`` on the app's deep link with an explicit package
           (needs Termux:API).
        """
        if not self.is_installed(package):
            raise PlatformError(f"{package} is not installed on this device")

        attempts: list[dict[str, object]] = []

        if component is None:
            component = self.resolve_launcher(package)

        argv = [f"{_SYSTEM_BIN}/monkey", "-p", package, "-c", "android.intent.category.LAUNCHER", "1"]
        attempts.append({"mechanism": "monkey", "argv": argv[1:]})
        outcome = self._try(argv, timeout=30.0)
        attempts[-1].update(outcome)
        if outcome["ok"] and not _monkey_failed(outcome):
            return {"ok": True, "method": "monkey", "package": package,
                    "component": component, "attempts": attempts, "foreground_verified": False}

        deep_link = _DEEP_LINKS.get(package)
        if deep_link:
            argv = [f"{_TERMUX_BIN}/termux-open-url", deep_link, package]
            attempts.append({"mechanism": "termux-open-url", "argv": argv[1:]})
            outcome = self._try(argv, timeout=30.0)
            attempts[-1].update(outcome)
            if outcome["ok"]:
                return {"ok": True, "method": "termux-open-url", "package": package,
                        "component": component, "attempts": attempts, "foreground_verified": False}

        raise PlatformError(_explain(attempts, f"could not start {package}"))

    @staticmethod
    def _try(argv: list[str], *, timeout: float) -> dict[str, object]:
        if not os.path.isfile(argv[0]):
            return {"ok": False, "error": f"{os.path.basename(argv[0])} not found at {argv[0]}"}
        try:
            result = _run(argv, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timed out after {timeout:g}s"}
        except (OSError, subprocess.SubprocessError) as exc:
            return {"ok": False, "error": str(exc)}
        combined = (result.stdout + result.stderr).strip()
        return {"ok": result.returncode == 0, "rc": result.returncode, "output": combined[:400]}


#: Deep links used when a package has no ``monkey`` path. Verified to exist for
#: the common apps; an unknown package simply gets no fallback.
_DEEP_LINKS: dict[str, str] = {
    "com.instagram.android": "https://instagram.com",
    "com.facebook.katana": "https://m.facebook.com",
    "com.facebook.system": "https://m.facebook.com",
    "com.whatsapp": "https://wa.me",
    "com.google.android.youtube": "https://m.youtube.com",
    "com.spotify.music": "https://open.spotify.com",
    "com.netflix.mediaclient": "https://www.netflix.com",
    "com.linkedin.android": "https://www.linkedin.com",
    "com.twitter.android": "https://twitter.com",
    "com.android.chrome": "https://www.google.com",
}


def _monkey_failed(outcome: dict[str, object]) -> bool:
    """``monkey`` often exits 0 while printing an error — never trust rc alone."""
    output = str(outcome.get("output") or "").lower()
    return any(
        marker in output
        for marker in ("error", "unable to connect", "not found", "exception", "denial")
    )


def _explain(attempts: list[dict[str, object]], summary: str) -> str:
    """Build a message that names every mechanism tried and why it failed."""
    lines = [summary]
    for attempt in attempts:
        mechanism = attempt.get("mechanism", "?")
        if attempt.get("ok") and not _monkey_failed(attempt):
            lines.append(f"  - {mechanism}: ok")
        else:
            reason = attempt.get("error") or attempt.get("output") or "no detail"
            lines.append(f"  - {mechanism}: {str(reason)[:160]}")
    from .detect import in_proot

    if in_proot():
        lines.append(
            "Prithivi is running under PRoot, where Android app launching is blocked. "
            "Run it in native Termux to open apps and URLs."
        )
    else:
        lines.append(
            "Install the Termux:API app (com.termux.api) from F-Droid, then run "
            "`pkg install termux-api`, to enable this."
        )
    return "\n".join(lines)
