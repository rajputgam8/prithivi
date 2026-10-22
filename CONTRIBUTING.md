# Contributing to PRITHIVI

## Non-negotiable rules

These come from the project brief and are enforced by CI where possible:

1. **No monolithic files.** If a file approaches ~400 lines, split it by responsibility.
2. **Clean modular architecture.** Respect the import rules in `ARCHITECTURE.md` §2. A test enforces them.
3. **Type hints everywhere.** `mypy --strict` must pass. No bare `dict`/`Any` in public signatures — use Pydantic models.
4. **Pydantic models for all structured data** crossing a module boundary (tool args, results, events, settings, DB rows).
5. **Explicit error handling.** No bare `except:`. No swallowed exceptions. Errors become typed values or are re-raised with context.
6. **Never hard-code secrets.** Environment variables via `prithivi.core.config` / `prithivi.security.secrets` only.
7. **Tests for important functionality.** New behaviour ships with a test in the same change.
8. **Components stay replaceable.** Talk to protocols/interfaces, not concrete classes.
9. **Minimal dependencies.** Justify every new one in the PR description.
10. **No claims without tests.** A feature is "done" only when its acceptance test passes.
11. **No simulated functionality.** Don't stub a feature to make a demo look complete.
12. **Real APIs and official OAuth only.** No scraping around ToS, no bypass of login/CAPTCHA/MFA/rate limits.
13. **Confirm sensitive actions.** Anything L3/L4 needs explicit user confirmation.
14. **Audit trail for important actions.**
15. **Protect user data.** Local-first, encrypt what must be, delete what must be deleted.

## Layering

```
interface/  →  may import anything
core/       →  may import llm/, security/, memory/, rag/, tools/registry, events
tools/      →  may import tools/base, platform/, security/, llm/(schemas only). NOT core/agent
voice/, vision/  →  may import core/events, platform/. NOT tools/
llm/        →  must not import tools/ or core/agent
platform/   →  the only place OS-specific libraries are allowed
```

A tool is a dumb executor plus metadata. It never decides anything.

## Adding a new tool (the important recipe)

A new capability must not require touching `core/`.

```python
# prithivi/tools/weather.py
from pydantic import BaseModel, Field
from prithivi.security.permissions import PermissionLevel
from prithivi.tools.base import ToolContext, ToolResult, tool


class WeatherInput(BaseModel):
    """Input for get_weather."""
    city: str = Field(..., description="City name, e.g. 'Mumbai'")
    units: str = Field("celsius", description="celsius | fahrenheit")


class WeatherOutput(BaseModel):
    city: str
    temp_c: float
    condition: str
    source_url: str


@tool(
    name="get_weather",
    description=(
        "Get current weather for a city. Use for questions about current or forecast "
        "weather. Do NOT use for historical climate research or general climate questions."
    ),
    input_model=WeatherInput,
    output_model=WeatherOutput,
    permission_level=PermissionLevel.L0,
    side_effects="none",
    requires_confirmation=False,
    timeout_s=10.0,
)
async def get_weather(ctx: ToolContext, args: WeatherInput) -> ToolResult:
    data = await ctx.http.get(...)          # egress policy applied by ctx.http
    if not data.ok:
        return ToolResult.failure(code="UPSTREAM_ERROR", message=str(data.status))
    return ToolResult.ok(data=WeatherOutput(...).model_dump(),
                         evidence={"http_status": data.status})
```

Then register it in the tool package's `__init__.py` (`REGISTRY.register(...)`) and add tests:

- happy path (mocked HTTP, no network)
- malformed input rejected before the executor runs
- backend unavailable returns `BACKEND_UNAVAILABLE`, not a fake result
- permission level and confirmation requirement match the intended level
- schema is valid JSON Schema (contract test covers it automatically)

## Adding an LLM provider

Implement `LLMProvider` in `prithivi/llm/`, add it to `llm/registry.py`, and add contract tests against recorded fixtures in `prithivi/tests/fixtures/llm/`. Do not special-case a provider anywhere outside its adapter.

## Style

- `ruff` for lint + format. `mypy --strict` for types.
- Docstrings on public modules, classes, and tool descriptions that name what the tool does **not** do.
- Comments explain *why*, never restate the code. Prefer a clear name.
- Small commits, conventional prefix (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`).
- Update the phase's docs in the same PR.

## Local checks before opening a PR

```bash
ruff check . && ruff format --check .
mypy prithivi
pytest -m "not hardware and not live" --cov=prithivi --cov-fail-under=70
```

## Reporting security issues

Do not open a public issue for a vulnerability. Note it privately with a reproduction path and the affected version; it will be fixed before disclosure.