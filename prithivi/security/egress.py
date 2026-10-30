"""Network egress policy (SECURITY.md T6).

Prithivi can read arbitrary web pages and that text reaches an LLM prompt. The
egress policy is the boundary: outbound requests are restricted to an explicit
host allowlist when one is set, and ``allow_network=False`` makes the assistant
genuinely offline. With no allowlist configured the default is "any public host"
— private/loopback addresses are refused by the SSRF guard, so the default never
reaches the device's own network or services.

Checked at the transport level rather than per-tool, so a new tool cannot
accidentally bypass it.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from urllib.parse import urlparse

__all__ = ["EgressError", "EgressPolicy", "check_url", "is_private_host"]


class EgressError(PermissionError):
    """A request was refused because the destination is not allowed."""


def is_private_host(host: str) -> bool:
    """True for loopback/link-local/private hosts.

    Blocks SSRF via a user-supplied URL reaching the device's own services.
    Deliberately conservative: ``localhost`` and any RFC1918 / unique-local
    address count as private.
    """
    host = host.lower().strip("[]")
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        return True
    if host in {"::1", "0.0.0.0"} or host.startswith(("fc", "fd")):
        return True
    parts = host.split(".")
    if len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
        a, b = int(parts[0]), int(parts[1])
        if a == 127 or a == 10 or (a == 192 and b == 168):
            return True
        if a == 172 and 16 <= b <= 31:
            return True
        if a == 169 and b == 254:  # link-local, incl. cloud metadata
            return True
    return False


@dataclass(frozen=True)
class EgressPolicy:
    """Decides whether a URL may be fetched.

    Args:
        allow_network: master switch. ``False`` denies everything.
        allowed_hosts: glob patterns. Empty means "no explicit host restriction":
            any *public* host is permitted. Private/loopback addresses are still
            refused by the SSRF guard in :meth:`validate` before this is reached,
            so the default cannot reach the device's own network.
        allow_private_hosts: off by default; SSRF guard.
        require_https: refuse plaintext HTTP except for loopback/LAN dev servers.
    """

    allow_network: bool = True
    allowed_hosts: tuple[str, ...] = ()
    allow_private_hosts: bool = False
    require_https: bool = False
    always_allowed: tuple[str, ...] = field(default=())

    def host_permitted(self, host: str) -> bool:
        host = host.lower()
        if any(fnmatch.fnmatchcase(host, p) for p in self.always_allowed):
            return True
        if not self.allowed_hosts:
            # No explicit restriction: any public host is reachable. The SSRF
            # guard in validate() has already refused private/loopback hosts, so
            # this is "any host on the public internet", not "any host at all".
            return True
        return any(fnmatch.fnmatchcase(host, p.lower()) for p in self.allowed_hosts)

    def check(self, url: str) -> None:
        """Raise :class:`EgressError` unless ``url`` may be fetched."""
        self.validate(url)

    def validate(self, url: str) -> str:
        """Return the host if allowed, else raise :class:`EgressError`."""
        if not self.allow_network:
            raise EgressError(
                "Network access is disabled (PRITHIVI_EGRESS_ALLOW_NETWORK=false). "
                "Prithivi is running fully offline."
            )
        try:
            parsed = urlparse(url)
        except ValueError as exc:
            raise EgressError(f"cannot parse URL: {url!r}") from exc
        if parsed.scheme not in {"http", "https"}:
            raise EgressError(f"scheme {parsed.scheme!r} is not allowed; use http or https")
        host = (parsed.hostname or "").lower()
        if not host:
            raise EgressError(f"no host in URL: {url!r}")
        if not self.allow_private_hosts and is_private_host(host):
            raise EgressError(f"refusing to reach private host {host!r}")
        if self.require_https and parsed.scheme != "https" and not is_private_host(host):
            raise EgressError(f"plaintext http is not permitted: {url!r}")
        if not self.host_permitted(host):
            raise EgressError(
                f"host {host!r} is not in PRITHIVI_EGRESS_ALLOWED_HOSTS "
                f"(currently {list(self.allowed_hosts) or 'empty'})"
            )
        return host

    def warnings(self) -> list[str]:
        if not self.allow_network:
            return []
        if not self.allowed_hosts and not self.always_allowed:
            return [
                (
                    "Egress allowlist is empty: any public host may be read. "
                    "Set PRITHIVI_EGRESS_ALLOWED_HOSTS to restrict page reading to "
                    "the hosts you trust."
                )
            ]
        return []


def check_url(url: str, *, allow_network: bool = True, allowed_hosts: tuple[str, ...] = ()) -> str:
    """One-shot helper for callers that do not hold an :class:`EgressPolicy`."""
    return EgressPolicy(allow_network=allow_network, allowed_hosts=allowed_hosts).validate(url)
