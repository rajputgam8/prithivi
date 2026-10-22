"""Web layer: search providers and page fetching.

Kept separate from ``llm/`` because searching is not a model capability — a search
outage degrades Prithivi to answering from parametric knowledge, while an LLM
outage stops it answering at all. Nothing here may import ``tools/`` or call the
agent; both modules return data and never authority (ARCHITECTURE.md S14).
"""

from __future__ import annotations

__all__ = ["fetch", "providers"]
