"""Pure-Python schema layer.

Replaces pydantic v2, which cannot be installed on aarch64-Android: pydantic-core
is a Rust extension with no ``manylinux``/Android wheel, and the Android target
triple cannot be built without the NDK Rust toolchain. See ARCHITECTURE.md S19.

Provides the three things the assistant actually needs from pydantic:

1. typed records, via stdlib :mod:`dataclasses`;
2. validation of untrusted JSON (tool arguments produced by an LLM), with
   field-path error reporting;
3. JSON Schema generation, which is what turns a tool's input type into the
   ``tools=[...]`` payload sent to the model.

Validation is deliberately strict: JSON from a language model is already the
right shape, and silently coercing ``"5"`` into ``5`` hides real bugs.

Only stdlib types are supported: ``str int float bool None``, ``list``/``tuple``/
``set``, ``dict``, ``Optional``, ``Union``, ``Literal``, ``Enum``, ``datetime``,
``Any``, and nested :class:`Model` subclasses.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import sys
import types
import typing
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "FieldError",
    "FieldSpec",
    "Model",
    "SchemaError",
    "ValidationError",
]

#: Sentinel for "no default", distinct from ``None`` which is a real value.
_UNSET = object()


@dataclasses.dataclass(frozen=True)
class FieldSpec:
    """Resolved metadata for one model field."""

    name: str
    type: object
    default: object
    required: bool
    optional: bool


class SchemaError(TypeError):
    """A :class:`Model` subclass is declared incorrectly."""


@dataclasses.dataclass(frozen=True)
class FieldError:
    """One reason a payload failed to validate."""

    path: str
    message: str
    expected: str | None = None

    def __str__(self) -> str:
        if self.expected:
            return f"{self.path or '<root>'}: expected {self.expected}, got {self.message}"
        return f"{self.path or '<root>'}: {self.message}"


class ValidationError(ValueError):
    """Aggregate of every :class:`FieldError` found in one payload."""

    def __init__(self, errors: Sequence[FieldError]) -> None:
        self.errors = list(errors)
        joined = "; ".join(str(e) for e in self.errors)
        super().__init__(f"{len(self.errors)} validation error(s): {joined}")

    def to_dict(self) -> dict[str, object]:
        return {
            "error": "VALIDATION_ERROR",
            "message": str(self),
            "fields": [dataclasses.asdict(e) for e in self.errors],
        }


# ---------------------------------------------------------------------------
# type inspection helpers
# ---------------------------------------------------------------------------


def _type_name(tp: object) -> str:
    if tp is type(None):
        return "null"
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return "enum(" + "|".join(str(m.value) for m in tp) + ")"
    origin = typing.get_origin(tp)
    if origin is not None:
        args = typing.get_args(tp)
        inner = ",".join(_type_name(a) for a in args)
        name = {list: "array", tuple: "array", set: "array", dict: "object"}.get(origin)
        return f"{name or _type_name(origin)}[{inner}]" if name else f"{origin}[{inner}]"
    if isinstance(tp, type):
        return {str: "string", int: "integer", float: "number", bool: "boolean", dt.datetime: "string"}.get(tp, tp.__name__)
    return str(tp).replace("typing.", "")


def _is_model(tp: object) -> bool:
    return isinstance(tp, type) and issubclass(tp, Model) and tp is not Model


def _strip_optional(tp: object) -> tuple[object, bool]:
    """Return ``(inner, optional)`` for ``Optional[X]`` / ``X | None``."""
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(tp)
        if type(None) in args:
            rest = tuple(a for a in args if a is not type(None))
            if len(rest) == 1:
                return rest[0], True
            return rest[0], True  # heterogeneous unions are not supported; first wins
        return tp, False
    return tp, False


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


class _Ctx:
    __slots__ = ("errors",)

    def __init__(self) -> None:
        self.errors: list[FieldError] = []

    def fail(self, path: str, message: str, expected: str | None = None) -> None:
        self.errors.append(FieldError(path=path, message=message, expected=expected))


def _validate(tp: object, value: object, path: str, ctx: _Ctx) -> object:
    """Validate ``value`` against ``tp``, returning a coerced value.

    Returns ``_UNSET``-free output: raises nothing, records into ``ctx`` instead so
    that a single pass reports every problem rather than only the first.
    """
    tp, optional = _strip_optional(tp)
    if value is None:
        if optional:
            return None
        ctx.fail(path, "null is not allowed here", "non-null")
        return None

    origin = typing.get_origin(tp)

    # Literal
    if origin is typing.Literal:
        allowed = typing.get_args(tp)
        if value not in allowed:
            ctx.fail(path, repr(value), "one of " + ", ".join(repr(a) for a in allowed))
        return value

    # Any / no annotation
    if tp is typing.Any or tp is None or tp is object:
        return value

    # Enum
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        try:
            return tp(value)
        except ValueError:
            ctx.fail(path, repr(value), _type_name(tp))
            return value

    # Nested model
    if _is_model(tp):
        assert isinstance(tp, type)
        if not isinstance(value, Mapping):
            ctx.fail(path, f"{type(value).__name__} is not an object", _type_name(tp))
            return value
        return tp.from_dict(dict(value), _path=path, _ctx=ctx)

    # Containers
    if origin in (list, tuple, set, frozenset):
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            ctx.fail(path, f"{type(value).__name__} is not an array", _type_name(tp))
            return value
        args = typing.get_args(tp)
        item_tp = args[0] if args else typing.Any
        items = [_validate(item_tp, v, f"{path}[{i}]", ctx) for i, v in enumerate(value)]
        if origin is list:
            return items
        if origin is set:
            return set(items)
        if origin is frozenset:
            return frozenset(items)
        return tuple(items)

    if origin is dict:
        if not isinstance(value, Mapping):
            ctx.fail(path, f"{type(value).__name__} is not an object", _type_name(tp))
            return value
        args = typing.get_args(tp)
        val_tp = args[1] if len(args) == 2 else typing.Any
        return {str(k): _validate(val_tp, v, f"{path}.{k}", ctx) for k, v in value.items()}

    if origin is not None:
        # Unsupported parametric alias, e.g. Sequence[str] from abc.
        ctx.fail(path, f"unsupported annotation {tp!r}", "simple type")
        return value

    # Scalars
    if tp is bool:
        if not isinstance(value, bool):
            ctx.fail(path, f"{type(value).__name__} is not a boolean", "boolean")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            ctx.fail(path, f"{type(value).__name__} is not an integer", "integer")
            return value
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            ctx.fail(path, f"{type(value).__name__} is not a number", "number")
            return value
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            ctx.fail(path, f"{type(value).__name__} is not a string", "string")
            return value
        return value
    if tp is dt.datetime:
        if not isinstance(value, str):
            ctx.fail(path, "datetime must be an ISO-8601 string", "string")
            return value
        try:
            return _parse_datetime(value)
        except ValueError:
            ctx.fail(path, f"cannot parse {value!r} as ISO-8601", "ISO-8601 timestamp")
            return value

    ctx.fail(path, f"unsupported annotation {tp!r}", "supported type")
    return value


def _parse_datetime(raw: str) -> dt.datetime:
    """Parse ISO-8601, accepting the ``Z`` suffix that stdlib rejects."""
    text = raw[:-1] + "+00:00" if raw.endswith(("Z", "z")) else raw
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed


# ---------------------------------------------------------------------------
# JSON Schema generation
# ---------------------------------------------------------------------------

_SCHEMA_OF: dict[object, dict[str, object]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    dt.datetime: {"type": "string", "format": "date-time"},
    typing.Any: {},
}


def _json_schema(tp: object) -> dict[str, object]:
    tp, optional = _strip_optional(tp)
    if origin := typing.get_origin(tp):
        if origin is typing.Literal:
            values = list(typing.get_args(tp))
            out: dict[str, object] = {"enum": values}
            if optional:
                out = {"anyOf": [out, {"type": "null"}]}
            return out
        if origin in (list, tuple, set, frozenset):
            args = typing.get_args(tp)
            out = {"type": "array", "items": _json_schema(args[0]) if args else {}}
            return out
        if origin is dict:
            args = typing.get_args(tp)
            out = {"type": "object"}
            if len(args) == 2:
                out["additionalProperties"] = _json_schema(args[1])
            return out
        return {}
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        out = {"type": "string", "enum": [m.value for m in tp]}
        return out
    if _is_model(tp):
        assert isinstance(tp, type)
        return tp.json_schema()
    base = dict(_SCHEMA_OF.get(tp, {}))
    if not base:
        base = {"type": "string"}  # pragma: no cover - guarded by SchemaError elsewhere
    return base


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Model:
    """Base class for every validated record in Prithivi.

    Subclasses declare type-annotated fields and are turned into dataclasses
    automatically by :meth:`__init_subclass__` — writing ``@dataclass`` yourself
    is optional and mostly unnecessary.

    Do not use a mutable default; write ``field(default_factory=...)``.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # A subclass of a dataclass is not itself a dataclass, so
        # dataclasses.fields() would return () and every field would look
        # unknown. Apply the decorator here so the mistake is impossible.
        if "__dataclass_fields__" not in cls.__dict__:
            dataclasses.dataclass(cls)  # type: ignore[call-overload]

    @classmethod
    def model_fields(cls) -> dict[str, FieldSpec]:
        """Resolved per-field metadata: type, default, and whether it is required.

        Computed once and cached. This is the single source of truth for
        validation, construction, and JSON Schema generation, so those three can
        never disagree about what a model looks like.

        Raises:
            SchemaError: if the class is not a dataclass, or an annotation
                cannot be resolved (see :meth:`_hints`).
        """
        cached = cls.__dict__.get("__prithivi_spec__")
        if cached is not None:
            return typing.cast("dict[str, FieldSpec]", cached)

        if not dataclasses.is_dataclass(cls):
            raise SchemaError(
                f"{cls.__qualname__} must be decorated with @dataclasses.dataclass "
                "to inherit from prithivi.core.schema.Model"
            )

        hints = cls._hints()
        specs: dict[str, FieldSpec] = {}
        for f in dataclasses.fields(cls):  # type: ignore[arg-type]
            if f.name.startswith("_"):
                continue  # private attributes are not part of the wire format
            tp = hints.get(f.name, typing.Any)
            _, optional = _strip_optional(tp)
            if f.default is not dataclasses.MISSING:
                default: object = f.default
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                default = f.default_factory()  # type: ignore[misc]
            else:
                default = _UNSET
            specs[f.name] = FieldSpec(
                name=f.name,
                type=tp,
                default=default,
                required=default is _UNSET and not optional,
                optional=optional,
            )
        cls.__prithivi_spec__ = specs  # type: ignore[attr-defined]
        return specs

    @classmethod
    def _hints(cls) -> dict[str, object]:
        """Resolved type hints, cached per class.

        Must resolve against the *defining* module's globals, not this one:
        ``typing.get_type_hints`` evaluates the string annotations produced by
        ``from __future__ import annotations`` and fails on any name it cannot
        see. That is exactly what happens to a subclass defined in a different
        module than its base.
        """
        cached = cls.__dict__.get("__prithivi_hints__")
        if cached is not None:
            return typing.cast("dict[str, object]", cached)
        module = sys.modules.get(cls.__module__)
        if module is None:  # pragma: no cover - only if __module__ is bogus
            raise SchemaError(f"{cls.__qualname__}: cannot locate defining module {cls.__module__!r}")
        try:
            hints = typing.get_type_hints(cls, vars(module))
        except NameError as exc:
            raise SchemaError(
                f"{cls.__qualname__}: cannot resolve type hints ({exc}). "
                "Any type named in an annotation must be importable in the defining module."
            ) from exc
        cls.__prithivi_hints__ = hints  # type: ignore[attr-defined]
        return hints

    @classmethod
    def from_dict(cls, data: object, *, _path: str = "", _ctx: _Ctx | None = None) -> Model:
        """Validate ``data`` and build the model.

        Raises:
            ValidationError: with one :class:`FieldError` per problem.
        """
        ctx = _ctx if _ctx is not None else _Ctx()
        if not isinstance(data, Mapping):
            ctx.fail(_path or cls.__name__, f"{type(data).__name__} is not an object", "object")
            raise ValidationError(ctx.errors)

        specs = cls.model_fields()
        for key in sorted(set(map(str, data)) - set(specs)):
            ctx.fail(f"{_path}.{key}" if _path else key, "unknown field (not accepted)")

        kwargs: dict[str, object] = {}
        for name, spec in specs.items():
            child = f"{_path}.{name}" if _path else name
            if name in data:
                kwargs[name] = _validate(spec.type, data[name], child, ctx)
                continue
            if spec.default is not _UNSET:
                continue  # let __init__ apply the default
            if spec.optional:
                kwargs[name] = None
                continue
            ctx.fail(child, "required field is missing", _type_name(spec.type))

        if ctx.errors:
            raise ValidationError(ctx.errors)
        return cls(**kwargs)  # type: ignore[call-arg]

    def to_dict(self) -> dict[str, object]:
        """Plain JSON-compatible projection (enums to values, datetimes to ISO)."""
        out: dict[str, object] = {}
        for name in self.model_fields():
            out[name] = _encode(getattr(self, name))
        return out

    @classmethod
    def json_schema(cls) -> dict[str, object]:
        """JSON Schema (draft 2020-12) describing an instance of this model."""
        properties: dict[str, object] = {}
        required: list[str] = []
        for name, spec in cls.model_fields().items():
            properties[name] = _json_schema(spec.type)
            if spec.required:
                required.append(name)
        return {
            "type": "object",
            "title": cls.__name__,
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }


def _encode(value: object) -> object:
    if isinstance(value, Model):
        return value.to_dict()
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, dt.datetime):
        return value.astimezone(dt.UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if isinstance(value, set):
        return sorted(_encode(v) for v in value)
    if isinstance(value, Mapping):
        return {str(k): _encode(v) for k, v in value.items()}
    return value
