"""Secret loading, masking, and redaction.

This is the **only** module permitted to read secret-valued environment
variables. Everything else receives a :class:`Secret` and cannot print it by
accident.

Rationale (SECURITY.md T4): an assistant that transcribes the user's screen,
reads web pages, and logs every tool call is exactly the kind of program that
leaks an API key into a log file. Masking is therefore a type-level property,
not a convention.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path

__all__ = ["Redactor", "Secret", "get_secret", "load_env_file", "redactor"]


class Secret:
    """A string that never reveals itself in logs or tracebacks.

    Call :meth:`reveal` at the exact point the plaintext is needed, e.g. when
    building an ``Authorization`` header.
    """

    __slots__ = ("_name", "_value")

    def __init__(self, value: str, *, name: str = "secret") -> None:
        self._value = value
        self._name = name

    def reveal(self) -> str:
        return self._value

    @property
    def is_set(self) -> bool:
        return bool(self._value)

    def __bool__(self) -> bool:
        return self.is_set

    def __len__(self) -> int:
        return len(self._value)

    def __repr__(self) -> str:
        return f"Secret({self._name}, {'set' if self.is_set else 'unset'})"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Secret) and other._value == self._value

    def __hash__(self) -> int:
        return hash(("Secret", self._name, self._value))


# Patterns for credentials that must never be written to a log, a transcript,
# or an LLM prompt. Ordered longest/most-specific first.
#
# Two shapes matter: (a) a vendor-issued token with a distinctive prefix, and
# (b) a ``key: value`` pair where the *key* announces a credential. Shape (b)
# needs an optional auth scheme (``Bearer``/``Basic``/``Token``) because those
# spaces stop a bare ``\S+`` from matching.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),                      # OpenAI-style
    re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}\b"),                  # Anthropic
    re.compile(r"\b(?:gh[pousr]|github_pat)_[A-Za-z0-9_]{20,}\b"),  # GitHub
    re.compile(r"\bglpat-[A-Za-z0-9_\-]{16,}\b"),                    # GitLab
    re.compile(r"\bxox[abposr]-[A-Za-z0-9\-]{10,}\b"),               # Slack
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                            # AWS access key
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b"),                     # Google API key
    re.compile(r"\bya29\.[A-Za-z0-9_\-]{20,}\b"),                    # Google OAuth
    # JWT: header.payload.signature. Header must start "eyJ" (base64 of '{');
    # the segment bounds are deliberately loose because signature length varies.
    re.compile(r"\beyJ[A-Za-z0-9_\-]{2,}\.[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,}\b"),
    re.compile(r"(?i)\b(bearer|basic|token)\s+([A-Za-z0-9._\-/+=]{12,})"),  # auth scheme
    # key: value form. Keeps the key, drops the value.
    re.compile(
        r"(?i)\b(api[_-]?key|apikey|authorization|auth[_-]?token|access[_-]?token"
        r"|refresh[_-]?token|client[_-]?secret|private[_-]?key|secret[_-]?key"
        r"|password|passwd|secret|bearer|token)\b\s*[:=]\s*"
        r"[\"']?(?:bearer\s+|basic\s+|token\s+)?([A-Za-z0-9._\-/+=]{8,})"
    ),
)


class Redactor:
    """Scrubs credential-shaped substrings and known literal secrets."""

    def __init__(self, literals: Mapping[str, str] | None = None) -> None:
        self._literals = tuple(sorted({v for v in (literals or {}).values() if v and len(v) >= 8}, key=len, reverse=True))

    def add_literal(self, value: str) -> None:
        """Register a specific value (e.g. the loaded API key) for exact removal."""
        if value and len(value) >= 8:
            self._literals = tuple(sorted({*self._literals, value}, key=len, reverse=True))

    def scrub(self, text: str) -> str:
        """Return ``text`` with every credential-shaped run replaced."""
        if not text:
            return text
        out = text
        for literal in self._literals:
            if literal in out:
                out = out.replace(literal, "[REDACTED]")
        for pattern in _PATTERNS:
            # Group 1 (or 2 when the pattern has an auth-scheme prefix) is the
            # value; the key and scheme are preserved so logs stay readable.
            value_group = pattern.groups
            if value_group:
                out = pattern.sub(
                    lambda m, group=value_group: m.group(0).replace(m.group(group), "[REDACTED]"), out
                )
            else:
                out = pattern.sub("[REDACTED]", out)
        return out

    def scrub_obj(self, value: object, *, _depth: int = 0) -> object:
        """Recursively scrub strings in a JSON-like structure.

        Keys are scrubbed too, so a secret used as a *key* is not leaked.
        """
        if _depth > 12:  # pragma: no cover - guards pathological nesting
            return "[TRUNCATED]"
        if isinstance(value, str):
            return self.scrub(value)
        if isinstance(value, Secret):
            return "[REDACTED]"
        if isinstance(value, Mapping):
            return {self.scrub(str(k)): self.scrub_obj(v, _depth=_depth + 1) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self.scrub_obj(v, _depth=_depth + 1) for v in value]
        return value


redactor = Redactor()


def load_env_file(path: str | Path) -> dict[str, str]:
    """Read a ``.env`` file into a dict without touching ``os.environ``.

    A hand-rolled parser avoids a hard dependency on python-dotenv and, more
    importantly, gives us the ``chmod`` warning that SECURITY.md S6 requires.

    Inline comments are stripped, because the shipped ``.env.example`` documents
    every key as ``KEY=value   # what it means``. Without this, a copy of that
    template yields values like ``"prod  # prod enforces secret rules"``, which
    compare unequal to ``prod`` and therefore quietly disable the prod
    enforcement path. A ``#`` inside a quoted value is left alone.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        return {}
    out: dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = _strip_inline_comment(value.strip())
        if key:
            out[key] = value
    return out


def _strip_inline_comment(value: str) -> str:
    """Remove an unquoted trailing ``# comment``.

    Quoted values are returned with only their quotes stripped, so a user agent
    containing ``#`` survives.
    """
    if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
        return value[1:-1]
    marker = value.find(" #")
    if marker != -1:
        value = value[:marker]
    elif value.startswith("#"):
        value = ""
    return value.rstrip()


def get_secret(env: Mapping[str, str], name: str, *, default: str = "") -> Secret:
    """Wrap ``env[name]`` as a :class:`Secret` and register it for redaction."""
    value = env.get(name, default) or ""
    secret = Secret(value, name=name)
    if secret.is_set:
        redactor.add_literal(value)
    return secret


def warn_if_world_readable(path: str | Path) -> str | None:
    """Return a warning message if a ``.env`` is group/world readable."""
    p = Path(path).expanduser()
    if not p.is_file():
        return None
    mode = p.stat().st_mode & 0o777
    if mode & 0o077:
        return f"{p} is mode {mode:o}; run: chmod 600 {p}"
    return None


def env_flag(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def env_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def env_list(env: Mapping[str, str], name: str) -> tuple[str, ...]:
    raw = env.get(name, "")
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def require_env(env: Mapping[str, str], name: str, *, hint: str = "") -> str:
    """Fetch a required value or fail loudly by name (never a bare KeyError)."""
    value = env.get(name, "")
    if not value:
        message = f"{name} is required but not set"
        if hint:
            message += f" — {hint}"
        raise OSError(message)
    redactor.add_literal(value)
    return value


def default_env_path() -> Path:
    """``$PRITHIVI_ENV_FILE`` if set, else ``./.env``."""
    override = os.environ.get("PRITHIVI_ENV_FILE")
    return Path(override) if override else Path(".env")
