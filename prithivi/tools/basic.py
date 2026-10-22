"""Tool implementations.

Phase 1 ships four: two information tools, and the two that make Prithivi feel
like an assistant on a phone — opening a URL and starting an app.

Every tool is registered through :func:`prithivi.tools.registry.build_registry`
rather than at import time, so a test can construct an isolated registry.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from urllib.parse import urlparse

from ..core.schema import Model
from ..platform.base import PlatformError
from .base import (
    Evidence,
    SideEffect,
    ToolContext,
    ToolErrorCode,
    ToolResult,
    tool,
)
from .safe_math import CalcError, calculate_expression

__all__ = ["APP_CATALOG", "build_registry", "resolve_app"]


def _probe_open_urls(ctx: ToolContext) -> str | None:
    """Why URL opening is unavailable, or ``None`` if it works."""
    return _missing_capability(ctx, "open_urls", "hand a URL to the system browser")


def _probe_launch_apps(ctx: ToolContext) -> str | None:
    """Why app launching is unavailable, or ``None`` if it works."""
    return _missing_capability(ctx, "launch_apps", "start an installed application")


def _missing_capability(ctx: ToolContext, attribute: str, action: str) -> str | None:
    """Report the first concrete reason ``attribute`` is false.

    Without a probe, ``doctor`` would claim ``open_url`` works on a device where
    every call is guaranteed to fail a uid check. Returns just the first reason:
    a wall of text is harder to act on than the one thing to fix.
    """
    platform = _platform(ctx)
    caps = getattr(platform, "capabilities", None)
    if caps is None or getattr(caps, attribute, False):
        return None
    for note in getattr(caps, "notes", ()) or ():
        if note:
            return str(note).rstrip(".") + "."
    return f"This platform cannot {action} in the current environment."


def _platform(ctx: ToolContext):  # type: ignore[no-untyped-def]
    """Get the platform adapter, caching it on the context for the session."""
    if getattr(ctx, "scratch", None) is not None and "platform" in ctx.scratch:
        return ctx.scratch["platform"]
    from ..platform.detect import detect
    adapter = detect()
    ctx.scratch["platform"] = adapter
    return adapter



# ---------------------------------------------------------------------------
# get_time
# ---------------------------------------------------------------------------


class GetTimeArgs(Model):
    """Tool arguments model."""

    timezone: str = "local"
    """IANA timezone name such as 'Asia/Kolkata', or 'local' / 'utc'."""


@tool(
    name="get_time",
    description=(
        "Get the current date and time. Use this whenever the answer depends on "
        "'now' — today's date, the current time, what day it is, or how long "
        "ago something happened. Do NOT use it for historical dates or for "
        "converting between timezones unless a timezone is given."
    ),
    input_model=GetTimeArgs,
    permission_level=0,
    side_effects=SideEffect.NONE,
    idempotent=False,
    timeout_s=5.0,
)
async def _get_time(ctx: ToolContext, timezone: str = "local") -> ToolResult:
    try:
        if timezone.lower() in {"local", "", "device"}:
            now = dt.datetime.now().astimezone()
        elif timezone.lower() == "utc":
            now = dt.datetime.now(dt.UTC)
        else:
            from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
            try:
                now = dt.datetime.now(ZoneInfo(timezone))
            except ZoneInfoNotFoundError:
                return ToolResult.fail(
                    ToolErrorCode.INVALID_INPUT,
                    f"unknown timezone {timezone!r}; use 'local', 'utc', or an IANA name like 'Asia/Kolkata'",
                    remedy="Retry with a valid IANA timezone name.",
                )
    except Exception as exc:  # noqa: BLE001
        return ToolResult.fail(ToolErrorCode.INTERNAL, f"could not read the clock: {exc}")

    return ToolResult.ok(
        {
            "iso": now.isoformat(),
            "date": now.date().isoformat(),
            "time": now.strftime("%H:%M:%S"),
            "weekday": now.strftime("%A"),
            "timezone": timezone,
            "offset": now.strftime("%z"),
        },
        evidence=Evidence("system_clock", {"iso": now.isoformat()}),
    )


# ---------------------------------------------------------------------------
# calculate
# ---------------------------------------------------------------------------


class CalculateArgs(Model):
    """Tool arguments model."""

    expression: str
    """A mathematical expression, e.g. '(125 * 0.18) + 3400 - 500'."""


@tool(
    name="calculate",
    description=(
        "Evaluate an arithmetic expression exactly, using a safe parser rather "
        "than mental arithmetic. Supports + - * / // % ** parentheses, "
        "constants (pi, e), and functions (sqrt, abs, round, min, max, log, ln, "
        "sin, cos, tan, fact, gcd, mean...). Use this for any multi-step numeric "
        "computation, percentages, and unit-free maths. Do NOT use it for "
        "algebra, calculus, symbolic manipulation, or statistics, which it cannot do."
    ),
    input_model=CalculateArgs,
    permission_level=0,
    side_effects=SideEffect.NONE,
    idempotent=True,
    timeout_s=5.0,
)
async def _calculate(ctx: ToolContext, expression: str) -> ToolResult:
    try:
        value = calculate_expression(expression)
    except CalcError as exc:
        return ToolResult.fail(
            ToolErrorCode.INVALID_INPUT, f"cannot evaluate {expression!r}: {exc}",
            remedy="Check the syntax and only use supported operators, constants, and functions.",
        )
    # Render ints as ints; the model reasons about "42" more reliably than "42.0".
    rendered = int(value) if float(value).is_integer() and abs(value) < 1e15 else round(value, 10)
    return ToolResult.ok(
        {"expression": expression, "result": rendered, "as_float": value},
        evidence=Evidence("computation", {"expression": expression, "result": rendered}),
    )


# ---------------------------------------------------------------------------
# open_url
# ---------------------------------------------------------------------------


class OpenUrlArgs(Model):
    """Tool arguments model."""

    url: str
    """An absolute http:// or https:// URL."""
    browser: str | None = None
    """Optional package name of a specific browser, e.g. 'com.android.chrome'."""


@tool(
    name="open_url",
    description=(
        "Open a web page in the device's browser. Use this for 'open <site>', "
        "to visit a link, or to show a page you have researched. Do NOT use it "
        "to open an installed app (use open_application) or to read a page's "
        "contents (use web_fetch)."
    ),
    input_model=OpenUrlArgs,
    permission_level=1,
    side_effects=SideEffect.REVERSIBLE,
    idempotent=True,
    timeout_s=30.0,
    availability_probe=_probe_open_urls,
)
async def _open_url(ctx: ToolContext, url: str, browser: str | None = None) -> ToolResult:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ToolResult.fail(
            ToolErrorCode.INVALID_INPUT, f"{url!r} is not an http(s) URL",
            remedy="Pass a full URL, for example https://instagram.com",
        )

    platform = _platform(ctx)
    if platform is None:
        return ToolResult.unavailable("platform adapter", remedy="Run `prithivi doctor`.")

    try:
        outcome = platform.open_url(url, package=browser)
    except PlatformError as exc:
        return ToolResult.fail(
            ToolErrorCode.PERMISSION, str(exc),
            remedy="Install Termux:API (com.termux.api) from F-Droid, then run Prithivi in native Termux.",
        )
    except Exception as exc:  # noqa: BLE001
        return ToolResult.fail(ToolErrorCode.INTERNAL, f"could not open {url}: {exc}")

    return ToolResult.ok(
        {"url": url, "method": outcome.get("method")},
        evidence=Evidence(
            "browser_request",
            {
                "method": outcome.get("method"),
                "package": outcome.get("package"),
                "foreground_verified": outcome.get("foreground_verified", False),
            },
        ),
    )


# ---------------------------------------------------------------------------
# open_application
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppEntry:
    """One installable app: friendly names, package, and optional search hint."""

    name: str
    package: str
    keywords: tuple[str, ...] = ()


#: Curated catalog. Package IDs are stable per app; verified present on the
#: target device except where noted. A request for an app that is not installed
#: returns a clear "not installed" error rather than a silent no-op.
APP_CATALOG: tuple[AppEntry, ...] = (
    AppEntry("instagram", "com.instagram.android", ("insta", "ig")),
    AppEntry("whatsapp", "com.whatsapp", ("wa", "chat")),
    AppEntry("youtube", "com.google.android.youtube", ("yt", "tube")),
    AppEntry("chrome", "com.android.chrome", ("browser", "google chrome")),
    AppEntry("firefox", "org.mozilla.firefox", ("browser",)),
    AppEntry("facebook", "com.facebook.katana", ("fb", "meta")),
    AppEntry("spotify", "com.spotify.music", ("music",)),
    AppEntry("netflix", "com.netflix.mediaclient", ("tv", "shows")),
    AppEntry("linkedin", "com.linkedin.android", ("linked in",)),
    AppEntry("twitter", "com.twitter.android", ("x", "tweet")),
    AppEntry("gmail", "com.google.android.gm", ("mail", "email")),
    AppEntry("maps", "com.google.android.apps.maps", ("google maps", "navigation")),
    AppEntry("telegram", "org.telegram.messenger", ("tg",)),
    AppEntry("camera", "com.android.camera", ("photo",)),
    AppEntry("phone", "com.android.dialer", ("calls", "dialer")),
    AppEntry("settings", "com.android.settings", ("preferences", "config")),
    AppEntry("calculator", "com.android.calculator2", ("calc",)),
    AppEntry("clock", "com.android.deskclock", ("alarm", "timer")),
    AppEntry("files", "com.android.documentsui", ("file manager",)),
    AppEntry("playstore", "com.android.vending", ("play store", "apps")),
)


def resolve_app(query: str) -> AppEntry | None:
    """Match a spoken/written app name to a catalog entry.

    Matching order: exact name, exact keyword, then substring. Returns ``None``
    when nothing matches, so the caller can offer the available list rather than
    guessing.
    """
    needle = " ".join(query.lower().split())
    if not needle:
        return None
    for entry in APP_CATALOG:
        if entry.name == needle:
            return entry
    for entry in APP_CATALOG:
        if needle in entry.keywords:
            return entry
    for entry in APP_CATALOG:
        if entry.name in needle or any(k in needle for k in entry.keywords if len(k) > 2):
            return entry
    return None


class OpenApplicationArgs(Model):
    """Tool arguments model."""

    application: str
    """Friendly app name, e.g. 'instagram', 'youtube', 'spotify'."""


@tool(
    name="open_application",
    description=(
        "Start an app installed on the phone, by friendly name — 'instagram', "
        "'youtube', 'spotify', 'whatsapp', 'chrome', 'gmail', 'maps', 'camera'. "
        "Use this when the user says to open or launch an app. Do NOT use it for "
        "websites (use open_url) and do NOT use it to post or send anything "
        "(use the specific messaging tool)."
    ),
    input_model=OpenApplicationArgs,
    permission_level=1,
    side_effects=SideEffect.REVERSIBLE,
    idempotent=True,
    timeout_s=40.0,
    availability_probe=_probe_launch_apps,
)
async def _open_application(ctx: ToolContext, application: str) -> ToolResult:
    entry = resolve_app(application)
    if entry is None:
        return ToolResult.fail(
            ToolErrorCode.NOT_FOUND,
            f"I do not know an app called {application!r}",
            remedy=f"Known apps: {', '.join(e.name for e in APP_CATALOG)}",
            data={"available": [e.name for e in APP_CATALOG]},
        )

    platform = _platform(ctx)
    if platform is None:
        return ToolResult.unavailable("platform adapter", remedy="Run `prithivi doctor`.")

    if not platform.is_installed(entry.package):
        return ToolResult.fail(
            ToolErrorCode.NOT_FOUND,
            f"{entry.name} ({entry.package}) is not installed on this phone",
            remedy="Install it from the Play Store, or ask me for something else.",
        )

    try:
        outcome = platform.launch_app(entry.package, component=platform.resolve_launcher(entry.package))
    except PlatformError as exc:
        return ToolResult.fail(
            ToolErrorCode.PERMISSION, str(exc),
            remedy="Install Termux:API from F-Droid and run Prithivi in native Termux.",
        )
    except Exception as exc:  # noqa: BLE001
        return ToolResult.fail(ToolErrorCode.INTERNAL, f"could not start {entry.name}: {exc}")

    return ToolResult.ok(
        {"application": entry.name, "package": entry.package, "method": outcome.get("method")},
        evidence=Evidence(
            "activity_started",
            {
                "package": entry.package,
                "component": outcome.get("component"),
                "method": outcome.get("method"),
                "foreground_verified": outcome.get("foreground_verified", False),
            },
        ),
    )


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def build_registry() -> list:
    """Return the Phase 1 tool specs."""
    return [_get_time, _calculate, _open_url, _open_application]
