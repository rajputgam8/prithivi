"""Platform adapter selection.

Kept separate from :mod:`prithivi.platform.base` so that ``tools/`` imports only
this factory and never a concrete adapter.
"""

from __future__ import annotations

import os
import platform
import sys

from .android import AndroidAdapter
from .base import PlatformAdapter, PlatformCapabilities, PlatformError

__all__ = ["PlatformAdapter", "PlatformCapabilities", "PlatformError", "detect", "in_proot", "is_termux"]

_ANDROID_MARKERS = ("ANDROID_ROOT", "ANDROID_DATA", "PREFIX", "TERMUX_VERSION")


def in_proot() -> bool:
    """True when running under PRoot's ptrace-based syscall interception.

    This matters a great deal on Android. PRoot intercepts every syscall and
    rewrites paths, which breaks the binder connections that Android's window
    manager and ``monkey`` need. Verified on this device: ``TracerPid`` is
    non-zero and ``proot`` appears 16 times in ``/proc/self/maps``.

    Consequence: app launching works from **native Termux** but not from inside
    a PRoot distribution. :func:`~prithivi.platform.android.AndroidAdapter.probe`
    uses this to report the gap instead of claiming a capability it lacks.
    """
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("TracerPid:"):
                    return int(line.split(":", 1)[1].strip()) != 0
    except (OSError, ValueError):
        pass
    try:
        with open("/proc/self/maps", encoding="utf-8") as handle:
            return "proot" in handle.read(65536)
    except OSError:
        return False


def is_termux() -> bool:
    """True inside Termux, which is how we know we are on Android."""
    if os.environ.get("TERMUX_VERSION"):
        return True
    if "com.termux" in os.environ.get("PREFIX", ""):
        return True
    return any(marker in os.environ for marker in ("ANDROID_ROOT", "ANDROID_DATA"))


def is_android() -> bool:
    return is_termux() or "ANDROID" in platform.release().upper() or os.path.isdir("/system/app")


def detect() -> PlatformAdapter:
    """Return the adapter for the current machine.

    Falls back to :class:`AndroidAdapter` on any unknown POSIX system rather than
    raising, so ``prithivi doctor`` can always report what is missing.
    """
    if is_android():
        return AndroidAdapter()
    from .desktop import DesktopAdapter
    return DesktopAdapter()


def describe_environment() -> str:
    """One-line environment summary for startup logs."""
    return (
        f"python={sys.version.split()[0]} os={platform.system()}/{platform.release()} "
        f"termux={is_termux()} proot={in_proot()} arch={platform.machine()}"
    )
