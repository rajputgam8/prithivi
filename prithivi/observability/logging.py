"""Structured logging with secret redaction.

Two formats: ``text`` for a human reading a terminal, ``json`` for log shipping.
Both pass through the global :mod:`~prithivi.security.secrets` redactor, so an API
key cannot reach a log file even if it was pasted into a prompt or returned by a
tool (SECURITY.md T4).

The text format is the one shown in the brief: a timestamp, a channel tag, and
the message.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from ..security.secrets import redactor

__all__ = ["JsonFormatter", "TextFormatter", "configure_logging", "get_logger"]

_CONFIGURED = False


class TextFormatter(logging.Formatter):
    """``[22:41:05] channel   message`` with redaction."""

    def __init__(self, *, debug: bool = False) -> None:
        super().__init__()
        self._debug = debug

    def format(self, record: logging.LogRecord) -> str:
        stamp = __import__("datetime").datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        channel = record.name.replace("prithivi.", "")
        message = record.getMessage()
        line = f"[{stamp}] {channel:<14} {message}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        if self._debug and record.args:
            line += f"\n    args={record.args!r}"
        # Redact last, so an exception's payload cannot leak either.
        return redactor.scrub(line)


class JsonFormatter(logging.Formatter):
    """One JSON object per line, fields scrubbed."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "channel": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(redactor.scrub_obj(payload), default=str)


def configure_logging(
    *,
    level: str = "INFO",
    fmt: str = "text",
    stream: Any = None,
    force: bool = False,
) -> None:
    """Install the root handler. Safe to call more than once."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    resolved = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(
        JsonFormatter() if fmt.lower() == "json" else TextFormatter(debug=resolved <= logging.DEBUG)
    )

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(resolved)

    # Third-party libraries are far too chatty at INFO.
    for noisy in ("httpx", "httpcore", "asyncio", "urllib3", "respx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
