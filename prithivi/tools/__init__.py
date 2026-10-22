"""Tool contracts, the allowlist registry, and the tool implementations.

Tools are dumb executors plus permission metadata (ARCHITECTURE.md S9). This
package deliberately has no import-time side effects: nothing is registered until
a registry is built, so a test can assemble exactly the tools it needs.

Import rules (ARCHITECTURE.md S2): nothing here may import ``core.agent``, and
nothing here may import ``llm``. A tool has no authority and no opinions — it
receives services through :class:`~prithivi.tools.base.ToolContext` and returns a
:class:`~prithivi.tools.base.ToolResult`.
"""

from __future__ import annotations

from .base import ToolContext, ToolError, ToolErrorCode, ToolResult, ToolSpec, tool
from .basic import build_registry as build_basic_registry
from .registry import ToolRegistry
from .search import build_search_registry
from .web_fetch import build_web_fetch_registry

__all__ = [
    "ToolContext",
    "ToolError",
    "ToolErrorCode",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "build_basic_registry",
    "build_search_registry",
    "build_tool_specs",
    "build_web_fetch_registry",
    "tool",
]


def build_tool_specs(*, include_web: bool = True) -> list[ToolSpec]:
    """Every tool Prithivi ships, in a stable order.

    Args:
        include_web: the Phase 6 web tools (``search_web``, ``web_fetch``,
            ``research_web``). They are excluded when egress is disabled, because
            a tool that is guaranteed to fail should not be offered to the model
            as if it worked.
    """
    specs: list[ToolSpec] = list(build_basic_registry())
    if include_web:
        specs.extend(build_search_registry())
        specs.extend(build_web_fetch_registry())
    return specs
