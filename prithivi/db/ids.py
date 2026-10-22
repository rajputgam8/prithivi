"""Sortable, dependency-free identifiers (DATA_MODEL.md "Conventions").

IDs are generated application-side as ULID-like strings: 10 characters of
base32 timestamp followed by 16 characters of base32 randomness. They sort
lexicographically by creation time, which lets ``ORDER BY id`` stand in for
``ORDER BY created_at`` on indexes that already key on the primary key, and
they never collide across processes because the random tail is 80 bits.
"""

from __future__ import annotations

import os
import time

__all__ = ["CROCKFORD", "new_id", "timestamp_of"]

#: Crockford base32: no I, L, O, U, so a hand-copied id cannot be misread.
CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

_TIME_LEN = 10
_RANDOM_LEN = 16


def _encode(value: int, length: int) -> str:
    out = []
    for _ in range(length):
        out.append(CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(out))


def new_id(prefix: str = "") -> str:
    """Return a fresh id, optionally namespaced (``mem_01J8...``)."""
    millis = int(time.time() * 1000)
    tail = int.from_bytes(os.urandom(10), "big")
    return f"{prefix}{_encode(millis, _TIME_LEN)}{_encode(tail, _RANDOM_LEN)}"


def timestamp_of(identifier: str) -> float:
    """Recover the creation time (epoch seconds) encoded in an id.

    Returns ``0.0`` when the id was not produced by :func:`new_id` (for
    example a caller-supplied key), so callers can sort mixed populations
    without branching.
    """
    body = identifier.rsplit("_", 1)[-1]
    if len(body) < _TIME_LEN:
        return 0.0
    value = 0.0
    for char in body[:_TIME_LEN].upper():
        digit = CROCKFORD.find(char)
        if digit < 0:
            return 0.0
        value = value * 32 + digit
    return value / 1000.0
