"""Runtime configuration.

All tunables in one place (ARCHITECTURE.md S16). Built from a ``.env`` file plus
the process environment, with explicit, typed parsing — there is no implicit
coercion of ``"maybe"`` to ``True``.

``Settings.from_env`` is the single entry point. Tests construct ``Settings``
with :meth:`Settings.for_testing` so that no test can be influenced by the
developer's real ``.env``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ..security.secrets import (
    Secret,
    default_env_path,
    env_flag,
    env_float,
    env_int,
    env_list,
    get_secret,
    load_env_file,
    warn_if_world_readable,
)

__all__ = ["ConfigError", "Settings"]

_VALID_ENV = ("dev", "prod")
_VALID_LOG_FORMATS = ("text", "json")
#: Mirrors ``prithivi.web.providers.PROVIDERS``; kept as a literal so importing
#: Settings cannot pull in httpx, which is loaded lazily elsewhere for startup
#: speed on a phone.
_VALID_SEARCH_PROVIDERS = ("duckduckgo", "serper", "brave", "tavily", "searxng")
_KEYED_SEARCH_PROVIDERS = ("serper", "brave", "tavily")


class ConfigError(ValueError):
    """Configuration is missing or inconsistent; the message names the variable."""


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings."""

    # --- core -------------------------------------------------------------
    env: str = "dev"
    data_dir: Path = Path("data")
    log_level: str = "INFO"
    log_format: str = "text"

    # --- llm --------------------------------------------------------------
    llm_provider: str = "openai_compatible"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: Secret = field(default_factory=lambda: Secret("", name="PRITHIVI_LLM_API_KEY"))
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024
    llm_timeout_s: float = 60.0
    llm_stream: bool = True

    # --- agent ------------------------------------------------------------
    permission_default_level: int = 2
    confirmation_timeout_s: int = 120
    agent_max_iterations: int = 6
    agent_turn_budget_s: float = 45.0

    # --- security ---------------------------------------------------------
    master_key: Secret = field(default_factory=lambda: Secret("", name="PRITHIVI_MASTER_KEY"))
    egress_allow_network: bool = True
    egress_allowed_hosts: tuple[str, ...] = ()
    memory_enabled: bool = True
    audit_retention_days: int = 180

    # --- phase 2: voice ---------------------------------------------------
    wake_word: str = "hey prithivi"
    wake_word_enabled: bool = True
    clap_to_wake: bool = True
    clap_threshold: float = 1.6
    clap_cooldown_ms: int = 700
    stt_provider: str = "termux"
    stt_model: str = "tiny.en"
    stt_language: str = "auto"
    tts_provider: str = "termux"
    tts_rate: float = 1.0
    tts_pitch: float = 1.0
    tts_voice: str = ""
    tts_engine: str = ""
    #: Streaming TTS coalescing: if the first spoken chunk takes at least this
    #: many seconds (the Termux:API round trip can), later sentences are merged
    #: into chunks of ``tts_coalesce_chars`` so the fixed per-call cost is paid
    #: once instead of per sentence. Set high (e.g. 999) to always stream.
    tts_slow_after_s: float = 1.5
    tts_coalesce_chars: int = 220
    barge_in_enabled: bool = True

    # --- phase 3: android control ----------------------------------------
    android_app_allowlist: tuple[str, ...] = ()
    browser_package: str = "com.android.chrome"
    fs_allowed_roots: tuple[str, ...] = ()

    # --- phase 6: web intelligence ---------------------------------------
    search_provider: str = "duckduckgo"
    search_api_key: Secret = field(default_factory=lambda: Secret("", name="PRITHIVI_SEARCH_API_KEY"))
    search_endpoint: str = ""
    search_timeout_s: float = 15.0
    web_fetch_max_bytes: int = 2_000_000
    web_fetch_timeout_s: float = 15.0
    web_respect_robots: bool = True
    web_user_agent: str = "Prithivi/0.6 (+https://github.com/prithivi)"
    web_max_chars_per_page: int = 4000
    research_max_sources: int = 8
    research_budget_s: float = 20.0
    remember_research: bool = False

    # --- phase 7: rag + memory -------------------------------------------
    embedding_model: str = "hash-embed-v1"
    embedding_dim: int = 384
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_top_k: int = 8

    # --- phase 10: api ----------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_token: Secret = field(default_factory=lambda: Secret("", name="PRITHIVI_API_TOKEN"))
    api_allow_remote: bool = False

    # ------------------------------------------------------------------
    @property
    def is_prod(self) -> bool:
        return self.env == "prod"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "prithivi.db"

    @property
    def audit_db_path(self) -> Path:
        return self.data_dir / "audit.db"

    @property
    def log_path(self) -> Path:
        return Path("logs") / "prithivi.log"

    def ensure_dirs(self) -> None:
        """Create the writable directories Prithivi needs. Safe to call twice."""
        for path in (self.data_dir, self.data_dir / "vectors", self.data_dir / "audio",
                     self.data_dir / "screenshots", Path("logs")):
            path.mkdir(parents=True, exist_ok=True)

    def validate(self) -> list[str]:
        """Return a list of human-readable problems. Empty list means valid.

        Never raises: the CLI prints these and continues in ``dev`` so a new user
        is not locked out of the one command that tells them what is missing.
        """
        problems: list[str] = []
        if self.env not in _VALID_ENV:
            problems.append(f"PRITHIVI_ENV must be one of {_VALID_ENV}, got {self.env!r}")
        if self.log_format not in _VALID_LOG_FORMATS:
            problems.append(f"PRITHIVI_LOG_FORMAT must be one of {_VALID_LOG_FORMATS}, got {self.log_format!r}")
        if not 0 <= self.permission_default_level <= 4:
            problems.append(f"PRITHIVI_PERMISSION_DEFAULT_LEVEL must be 0..4, got {self.permission_default_level}")
        if self.agent_max_iterations < 1:
            problems.append("PRITHIVI_AGENT_MAX_ITERATIONS must be >= 1")
        if not self.llm_model:
            problems.append("PRITHIVI_LLM_MODEL is required (e.g. gpt-4o-mini)")
        if self.llm_provider in {"openai_compatible", "anthropic"} and not self.llm_api_key.is_set:
            problems.append(
                "PRITHIVI_LLM_API_KEY is not set. Prithivi needs a cloud LLM because a local "
                "model will not fit in this device's available RAM (~750 MB free)."
            )
        if self.is_prod and not self.master_key.is_set:
            problems.append("PRITHIVI_MASTER_KEY is required when PRITHIVI_ENV=prod")
        if self.api_allow_remote and self.api_host in {"127.0.0.1", "localhost", "::1"}:
            problems.append("PRITHIVI_API_ALLOW_REMOTE=true but PRITHIVI_API_HOST is loopback")
        if not self.egress_allow_network and self.search_provider != "duckduckgo":
            problems.append("network egress is disabled but a remote search provider is selected")
        if self.search_provider not in _VALID_SEARCH_PROVIDERS:
            problems.append(
                f"PRITHIVI_SEARCH_PROVIDER must be one of {sorted(_VALID_SEARCH_PROVIDERS)}, "
                f"got {self.search_provider!r}"
            )
        if self.search_provider == "searxng" and not self.search_endpoint:
            problems.append("PRITHIVI_SEARCH_PROVIDER=searxng needs PRITHIVI_SEARCH_ENDPOINT (e.g. https://searx.be)")
        if self.search_provider in _KEYED_SEARCH_PROVIDERS and not self.search_api_key.is_set:
            problems.append(
                f"PRITHIVI_SEARCH_PROVIDER={self.search_provider} needs PRITHIVI_SEARCH_API_KEY "
                "(or use duckduckgo, which needs no key)"
            )
        return problems

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, env_file: str | Path | None = None, *, overrides: Mapping[str, str] | None = None) -> Settings:
        """Build settings from ``.env`` then the process environment.

        Precedence: explicit ``overrides`` > ``os.environ`` > ``.env`` file >
        dataclass defaults.
        """
        merged: dict[str, str] = {}
        merged.update(load_env_file(env_file if env_file is not None else default_env_path()))
        merged.update({k: v for k, v in os.environ.items() if k.startswith("PRITHIVI_")})
        merged.update({k: str(v) for k, v in (overrides or {}).items()})

        try:
            settings = cls(
                env=merged.get("PRITHIVI_ENV", "dev").strip().lower(),
                data_dir=Path(merged.get("PRITHIVI_DATA_DIR", "data")).expanduser(),
                log_level=merged.get("PRITHIVI_LOG_LEVEL", "INFO").strip().upper(),
                log_format=merged.get("PRITHIVI_LOG_FORMAT", "text").strip().lower(),
                llm_provider=merged.get("PRITHIVI_LLM_PROVIDER", "openai_compatible").strip().lower(),
                llm_base_url=merged.get("PRITHIVI_LLM_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/"),
                llm_api_key=get_secret(merged, "PRITHIVI_LLM_API_KEY"),
                llm_model=merged.get("PRITHIVI_LLM_MODEL", "gpt-4o-mini").strip(),
                llm_temperature=env_float(merged, "PRITHIVI_LLM_TEMPERATURE", 0.2),
                llm_max_tokens=env_int(merged, "PRITHIVI_LLM_MAX_TOKENS", 1024),
                llm_timeout_s=env_float(merged, "PRITHIVI_LLM_TIMEOUT_S", 60.0),
                llm_stream=env_flag(merged, "PRITHIVI_LLM_STREAM", True),
                permission_default_level=env_int(merged, "PRITHIVI_PERMISSION_DEFAULT_LEVEL", 2),
                confirmation_timeout_s=env_int(merged, "PRITHIVI_CONFIRMATION_TIMEOUT_S", 120),
                agent_max_iterations=env_int(merged, "PRITHIVI_AGENT_MAX_ITERATIONS", 6),
                agent_turn_budget_s=env_float(merged, "PRITHIVI_AGENT_TURN_BUDGET_S", 45.0),
                master_key=get_secret(merged, "PRITHIVI_MASTER_KEY"),
                egress_allow_network=env_flag(merged, "PRITHIVI_EGRESS_ALLOW_NETWORK", True),
                egress_allowed_hosts=env_list(merged, "PRITHIVI_EGRESS_ALLOWED_HOSTS"),
                memory_enabled=env_flag(merged, "PRITHIVI_MEMORY_ENABLED", True),
                audit_retention_days=env_int(merged, "PRITHIVI_AUDIT_RETENTION_DAYS", 180),
                wake_word=merged.get("PRITHIVI_WAKE_WORD", "hey prithivi").strip(),
                wake_word_enabled=env_flag(merged, "PRITHIVI_WAKE_WORD_ENABLED", True),
                clap_to_wake=env_flag(merged, "PRITHIVI_CLAP_TO_WAKE", True),
                clap_threshold=env_float(merged, "PRITHIVI_CLAP_THRESHOLD", 1.6),
                clap_cooldown_ms=env_int(merged, "PRITHIVI_CLAP_COOLDOWN_MS", 700),
                stt_provider=merged.get("PRITHIVI_STT_PROVIDER", "termux").strip().lower(),
                stt_model=merged.get("PRITHIVI_STT_MODEL", "tiny.en").strip(),
                stt_language=merged.get("PRITHIVI_STT_LANGUAGE", "auto").strip().lower(),
                tts_provider=merged.get("PRITHIVI_TTS_PROVIDER", "termux").strip().lower(),
                tts_rate=env_float(merged, "PRITHIVI_TTS_RATE", 1.0),
                tts_pitch=env_float(merged, "PRITHIVI_TTS_PITCH", 1.0),
                tts_voice=merged.get("PRITHIVI_TTS_VOICE", "").strip(),
                tts_engine=merged.get("PRITHIVI_TTS_ENGINE", "").strip(),
                tts_slow_after_s=env_float(merged, "PRITHIVI_TTS_SLOW_AFTER_S", 1.5),
                tts_coalesce_chars=env_int(merged, "PRITHIVI_TTS_COALESCE_CHARS", 220),
                barge_in_enabled=env_flag(merged, "PRITHIVI_BARGE_IN_ENABLED", True),
                android_app_allowlist=env_list(merged, "PRITHIVI_ANDROID_APP_ALLOWLIST"),
                browser_package=merged.get("PRITHIVI_BROWSER_PACKAGE", "com.android.chrome").strip(),
                fs_allowed_roots=env_list(merged, "PRITHIVI_FS_ALLOWED_ROOTS"),
                search_provider=merged.get("PRITHIVI_SEARCH_PROVIDER", "duckduckgo").strip().lower(),
                search_api_key=get_secret(merged, "PRITHIVI_SEARCH_API_KEY"),
                search_endpoint=merged.get("PRITHIVI_SEARCH_ENDPOINT", "").strip(),
                search_timeout_s=env_float(merged, "PRITHIVI_SEARCH_TIMEOUT_S", 15.0),
                web_fetch_max_bytes=env_int(merged, "PRITHIVI_WEB_FETCH_MAX_BYTES", 2_000_000),
                web_fetch_timeout_s=env_float(merged, "PRITHIVI_WEB_FETCH_TIMEOUT_S", 15.0),
                web_respect_robots=env_flag(merged, "PRITHIVI_WEB_RESPECT_ROBOTS", True),
                web_user_agent=merged.get("PRITHIVI_WEB_USER_AGENT", "Prithivi/0.6 (+https://github.com/prithivi)").strip(),
                web_max_chars_per_page=env_int(merged, "PRITHIVI_WEB_MAX_CHARS_PER_PAGE", 4000),
                research_max_sources=env_int(merged, "PRITHIVI_RESEARCH_MAX_SOURCES", 8),
                research_budget_s=env_float(merged, "PRITHIVI_RESEARCH_BUDGET_S", 20.0),
                remember_research=env_flag(merged, "PRITHIVI_REMEMBER_RESEARCH", False),
                embedding_model=merged.get("PRITHIVI_EMBEDDING_MODEL", "hash-embed-v1").strip(),
                embedding_dim=env_int(merged, "PRITHIVI_EMBEDDING_DIM", 384),
                chunk_size=env_int(merged, "PRITHIVI_CHUNK_SIZE", 512),
                chunk_overlap=env_int(merged, "PRITHIVI_CHUNK_OVERLAP", 64),
                retrieval_top_k=env_int(merged, "PRITHIVI_RETRIEVAL_TOP_K", 8),
                api_host=merged.get("PRITHIVI_API_HOST", "127.0.0.1").strip(),
                api_port=env_int(merged, "PRITHIVI_API_PORT", 8000),
                api_token=get_secret(merged, "PRITHIVI_API_TOKEN"),
                api_allow_remote=env_flag(merged, "PRITHIVI_API_ALLOW_REMOTE", False),
            )
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc

        problems = settings.validate()
        if problems and settings.is_prod:
            raise ConfigError("; ".join(problems))
        return settings

    @classmethod
    def for_testing(cls, **overrides: object) -> Settings:
        """Settings for tests: no network, temp-friendly, no secrets.

        Any attempt to reach the network must fail loudly in a test, so egress is
        off unless a test opts back in.
        """
        base: dict[str, object] = {
            "env": "dev",
            "data_dir": Path("./data"),
            "egress_allow_network": False,
            "llm_api_key": Secret("test-key-not-real", name="PRITHIVI_LLM_API_KEY"),
        }
        base.update(overrides)
        return cls(**base)  # type: ignore[arg-type]

    def warnings(self) -> list[str]:
        """Non-fatal advisories to show the user once at startup."""
        out: list[str] = []
        if self.is_prod:
            problem = warn_if_world_readable(default_env_path())
            if problem:
                out.append(problem)
        if self.egress_allow_network and not self.egress_allowed_hosts:
            out.append("Egress host allowlist is empty: Prithivi may read any public web page. Set PRITHIVI_EGRESS_ALLOWED_HOSTS to restrict this.")
        if self.permission_default_level < 2:
            out.append("PRITHIVI_PERMISSION_DEFAULT_LEVEL < 2 means actions that change your data run without confirmation.")
        return out
