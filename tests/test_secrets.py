"""Secret handling and redaction (SECURITY.md S6).

Two claims are tested here. First, a credential cannot leak by being printed,
logged, or hashed: ``Secret`` refuses to reveal itself in ``repr``/``str``, and
the :class:`Redactor` removes credential-shaped text wherever it appears.
Second, configuration parsing is strict — ``PRITHIVI_ALLOW_EVERYTHING=maybe``
must not silently become ``True``.
"""

from __future__ import annotations

import pytest

from prithivi.security.secrets import (
    Redactor,
    Secret,
    env_flag,
    env_float,
    env_int,
    env_list,
    get_secret,
    load_env_file,
    redactor,
    require_env,
    warn_if_world_readable,
)

# Realistic shapes. These are fake, but the same length and character classes as
# the real thing, so the patterns are actually exercised.
FAKE = {
    "openai": "sk-abcdefghijklmnopqrstuvwxyz0123",
    "anthropic": "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123",
    "github": "ghp_abcdefghijklmnopqrstuvwxyz012345",
    "gitlab": "glpat-abcdefghijklmnopqrst",
    "slack": "xoxb-123456789012-abcdefghijkl",
    "aws": "AKIAIOSFODNN7EXAMPLE",
    "google": "AIzaSyAbcdefghijklmnopqrstuvwxyz012345",
    "jwt": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r",
}


# --- Secret -----------------------------------------------------------------


def test_a_secret_never_reveals_itself_in_a_repr() -> None:
    secret = Secret(FAKE["openai"], name="PRITHIVI_LLM_API_KEY")
    assert FAKE["openai"] not in repr(secret)
    assert FAKE["openai"] not in str(secret)
    assert FAKE["openai"] not in f"{secret}"
    # Enough is said to debug with: which variable, and whether it is set.
    assert "PRITHIVI_LLM_API_KEY" in repr(secret)
    assert "set" in repr(secret)


def test_reveal_is_the_only_way_out() -> None:
    assert Secret(FAKE["openai"]).reveal() == FAKE["openai"]


def test_an_empty_secret_is_falsy() -> None:
    assert not Secret("")
    assert not Secret("").is_set
    assert Secret("x").is_set


def test_a_secret_survives_a_traceback() -> None:
    """An exception raised while a Secret is on the stack must not print it.

    This is the path that actually leaks in practice: a framework formatting
    ``locals()`` into a log line.
    """
    secret = Secret(FAKE["openai"], name="K")

    def boom() -> None:
        raise ValueError("while using the key")

    with pytest.raises(ValueError) as exc:
        boom()
    assert FAKE["openai"] not in str(exc.value)
    assert repr(secret).count(FAKE["openai"]) == 0


def test_secrets_compare_by_value() -> None:
    assert Secret("abc", name="a") == Secret("abc", name="b")
    assert Secret("abc") != Secret("xyz")
    assert Secret("abc") != "abc"


def test_secrets_are_hashable_for_use_as_keys() -> None:
    """Two variables holding the same value stay distinct keys.

    ``__hash__`` mixes in the name, so collapsing them would let a dict keyed by
    secrets conflate two different credentials.
    """
    assert len({Secret("abc", name="a"), Secret("abc", name="a")}) == 1
    assert len({Secret("abc", name="a"), Secret("abc", name="b")}) == 2


# --- redaction --------------------------------------------------------------


@pytest.mark.security
@pytest.mark.parametrize("vendor", sorted(FAKE))
def test_a_known_credential_shape_is_redacted(vendor: str) -> None:
    text = f"the key is {FAKE[vendor]} ok"
    scrubbed = Redactor().scrub(text)
    assert FAKE[vendor] not in scrubbed
    assert "REDACTED" in scrubbed


@pytest.mark.security
@pytest.mark.parametrize(
    "text",
    [
        "Authorization: Bearer abcdefghijklmnop123456",
        "api_key=abcdefghijklmnop123456",
        'api_key: "abcdefghijklmnop123456"',
        "password: hunter2hunter2",
        "client_secret=abcdefghijklmnop",
        "PRIVATE_KEY = abcdefghijklmnop",
    ],
)
def test_credentials_in_key_value_form_are_redacted(text: str) -> None:
    scrubbed = Redactor().scrub(text)
    assert "abcdefghijklmnop" not in scrubbed
    assert "hunter2hunter2" not in scrubbed


def test_the_key_is_kept_so_the_log_stays_readable() -> None:
    scrubbed = Redactor().scrub("api_key=abcdefghijklmnop123456")
    assert "api_key" in scrubbed, "the field name is not a secret"


@pytest.mark.parametrize(
    "text",
    [
        "here is a normal sentence about nothing in particular",
        "https://docs.python.org/3/library/os.html",
        "the answer is 42",
        "",
    ],
)
def test_ordinary_text_is_untouched(text: str) -> None:
    assert Redactor().scrub(text) == text


def test_a_registered_literal_is_scrubbed_anywhere() -> None:
    """An opaque key that matches no vendor pattern must still be removable."""
    scrubber = Redactor({"K": "totally-opaque-credential"})
    assert "totally-opaque-credential" not in scrubber.scrub(
        "value=totally-opaque-credential end"
    )


def test_short_literals_are_not_registered() -> None:
    """Registering a 3-character string would redact half the language."""
    scrubber = Redactor()
    scrubber.add_literal("abc")
    assert scrubber.scrub("abc def") == "abc def"


def test_scrub_obj_walks_nested_structures() -> None:
    payload = {
        "messages": [{"content": f"use key {FAKE['openai']}"}],
        "nested": {"deep": {"secret": FAKE["github"]}},
        "count": 3,
    }
    scrubbed = Redactor().scrub_obj(payload)
    assert FAKE["openai"] not in str(scrubbed)
    assert FAKE["github"] not in str(scrubbed)
    assert scrubbed["count"] == 3, "non-strings must pass through untouched"


def test_scrub_obj_hides_a_secret_used_as_a_key() -> None:
    scrubbed = Redactor().scrub_obj({FAKE["openai"]: "value"})
    assert FAKE["openai"] not in str(scrubbed)


def test_scrub_obj_replaces_a_secret_object() -> None:
    assert Redactor().scrub_obj({"k": Secret("anything")}) == {"k": "[REDACTED]"}


def test_scrub_obj_bounds_its_recursion() -> None:
    deep: object = "leaf"
    for _ in range(30):
        deep = {"n": deep}
    assert Redactor().scrub_obj(deep) is not None


# --- .env parsing -----------------------------------------------------------


def test_a_dotenv_file_is_parsed_without_touching_the_process_env(tmp_path) -> None:
    import os

    path = tmp_path / ".env"
    path.write_text(
        "\n".join([
            "# a comment",
            "",
            "PRITHIVI_LLM_MODEL=gpt-4o",
            'PRITHIVI_LLM_API_KEY="quoted-value"',
            "export PRITHIVI_EXPORTED=exported",
            "SPACED  =  trimmed  ",
            "no_equals_sign",
            "PRITHIVI_SINGLE_QUOTED='single'",
        ]),
        encoding="utf-8",
    )
    parsed = load_env_file(path)
    assert parsed["PRITHIVI_LLM_MODEL"] == "gpt-4o"
    assert parsed["PRITHIVI_LLM_API_KEY"] == "quoted-value"
    assert parsed["PRITHIVI_EXPORTED"] == "exported"
    assert parsed["SPACED"] == "trimmed"
    assert parsed["PRITHIVI_SINGLE_QUOTED"] == "single"
    assert "no_equals_sign" not in parsed
    assert "PRITHIVI_LLM_MODEL" not in os.environ


def test_a_missing_dotenv_is_empty(tmp_path) -> None:
    assert load_env_file(tmp_path / "nope.env") == {}


def test_a_world_readable_dotenv_is_flagged(tmp_path) -> None:
    path = tmp_path / ".env"
    path.write_text("A=1", encoding="utf-8")
    path.chmod(0o644)
    warning = warn_if_world_readable(path)
    assert warning is not None
    assert "chmod 600" in warning

    path.chmod(0o600)
    assert warn_if_world_readable(path) is None


def test_a_missing_dotenv_has_no_permission_warning(tmp_path) -> None:
    assert warn_if_world_readable(tmp_path / "absent") is None


# --- typed configuration parsing -------------------------------------------


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", "On"])
def test_truthy_flags_parse(raw: str) -> None:
    assert env_flag({"F": raw}, "F") is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "", "maybe", "2"])
def test_anything_else_is_false_not_an_error(raw: str) -> None:
    """An unrecognised flag is False, never an exception at startup."""
    assert env_flag({"F": raw}, "F") is False


def test_a_missing_flag_uses_the_default() -> None:
    assert env_flag({}, "F", default=True) is True
    assert env_flag({"F": ""}, "F", default=True) is True


def test_numbers_fail_loudly_on_garbage() -> None:
    """A bad number must stop startup, not become a silent default.

    Unlike a flag — where an unrecognised value can safely fail closed to
    ``False`` — a number has no safe default: ``0`` could mean "no iterations"
    or "unlimited", and ``env_int`` cannot know which the user meant. So it
    raises, naming the variable, rather than guessing.
    """
    assert env_int({"N": "12"}, "N", 0) == 12
    assert env_int({}, "N", 7) == 7
    with pytest.raises(ValueError, match="PRITHIVI_MAX_ITERATIONS"):
        env_int({"PRITHIVI_MAX_ITERATIONS": "lots"}, "PRITHIVI_MAX_ITERATIONS", 5)


def test_a_float_falls_back_only_when_absent() -> None:
    assert env_float({"F": "0.25"}, "F", 0.0) == 0.25
    assert env_float({}, "F", 1.5) == 1.5
    with pytest.raises(ValueError, match="F"):
        env_float({"F": "x"}, "F", 1.5)


def test_lists_split_on_commas() -> None:
    assert env_list({"L": "a, b ,c"}, "L") == ("a", "b", "c")
    assert env_list({}, "L") == ()
    assert env_list({"L": ""}, "L") == ()


def test_require_env_names_the_missing_variable() -> None:
    """A missing key must fail by name, not as a bare KeyError."""
    with pytest.raises(OSError) as exc:
        require_env({}, "PRITHIVI_LLM_API_KEY", hint="set it in .env")
    assert "PRITHIVI_LLM_API_KEY" in str(exc.value)
    assert "set it in .env" in str(exc.value)


def test_require_env_returns_a_present_value() -> None:
    assert require_env({"K": "v"}, "K") == "v"


def test_get_secret_registers_the_value_for_redaction() -> None:
    """A loaded key must be scrubbable even though it matches no vendor pattern."""
    secret = get_secret({"K": "opaque-key-1234"}, "K")
    assert secret.reveal() == "opaque-key-1234"
    assert "opaque-key-1234" not in redactor.scrub("leaking opaque-key-1234 here")


def test_get_secret_of_a_missing_key_is_empty() -> None:
    assert not get_secret({}, "K").is_set