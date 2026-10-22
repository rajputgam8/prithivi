"""Calculator with a hand-written parser.

There is **no** ``eval``/``exec`` anywhere in Prithivi (SECURITY.md S9), and this
module is why that is affordable rather than merely stated: "what is 12% of 3400"
needs an expression evaluator, and the model will keep asking for one.

Implemented as a recursive-descent parser:

    expr    := term (('+' | '-') term)*
    term    := power (('*' | '/' | '//' | '%') power)*
    power   := unary ('**' power)?          # right-associative
    unary   := ('-' | '+') unary | primary
    primary := number | name | name '(' args ')' | '(' expr ')'

Unknown names are a parse error, never a name lookup. The only reachable
callables are the constants and functions in :data:`FUNCTIONS`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

__all__ = ["CONSTANTS", "FUNCTIONS", "CalcError", "calculate_expression"]


class CalcError(ValueError):
    """A syntax, name, domain, or arithmetic error in an expression."""


CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "π": math.pi,
    "e": math.e,
    "tau": math.tau,
    "phi": (1 + 5**0.5) / 2,
}

#: Degrees-based trig, because "sin(90)" should be 1 for a human.
_FUNCTIONS: dict[str, object] = {
    "sqrt": lambda x: math.sqrt(x),
    "cbrt": lambda x: math.copysign(abs(x) ** (1 / 3), x),
    "abs": abs,
    "round": lambda x, n=0: round(x, int(n)),
    "floor": math.floor,
    "ceil": math.ceil,
    "min": min,
    "max": max,
    "pow": pow,
    "log": lambda x, base=10: math.log(x, base) if base != 10 else math.log10(x),
    "log10": math.log10,
    "log2": math.log2,
    "ln": math.log,
    "exp": math.exp,
    "sin": lambda x: math.sin(math.radians(x)),
    "cos": lambda x: math.cos(math.radians(x)),
    "tan": lambda x: math.tan(math.radians(x)),
    "asin": lambda x: math.degrees(math.asin(x)),
    "acos": lambda x: math.degrees(math.acos(x)),
    "atan": lambda x: math.degrees(math.atan(x)),
    "atan2": lambda y, x: math.degrees(math.atan2(y, x)),
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "hypot": math.hypot,
    "degrees": math.degrees,
    "radians": math.radians,
    # Wrapped because the tokenizer yields floats and these need exact ints.
    "fact": lambda x: math.factorial(int(x)),
    "gcd": lambda *xs: math.gcd(*(int(x) for x in xs)),
    "sum": lambda *xs: sum(xs),
    "mean": lambda *xs: sum(xs) / len(xs),
}

#: Public aliases for docs and tests.
FUNCTIONS = _FUNCTIONS

_TOKEN_RE = re.compile(
    r"""
    \s*(?:
        # Thousands separators: 1,250 is one number, not a tuple argument.
        (?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)
      | (?P<name>[A-Za-z_][A-Za-z_0-9]*)
      | (?P<op>\*\*|\^|//|[+\-*/%(),])
    )
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class _Token:
    kind: str  # 'number' | 'name' | 'op' | 'end'
    text: str
    pos: int


def _tokenize(source: str) -> list[_Token]:
    tokens: list[_Token] = []
    index = 0
    while index < len(source):
        match = _TOKEN_RE.match(source, index)
        if match is None or match.end() == match.start():
            if source[index].isspace():
                index += 1
                continue
            raise CalcError(f"unexpected character {source[index]!r} at position {index}")
        index = match.end()
        kind = match.lastgroup or ""
        tokens.append(_Token(kind=kind, text=match.group(kind), pos=match.start(kind)))
    tokens.append(_Token(kind="end", text="", pos=len(source)))
    return tokens


class _Parser:
    """Recursive-descent evaluator over a fixed token stream.

    Recursion depth is bounded explicitly. Deeply nested input ("((((1))))") is
    attacker-influenced — the expression comes from a model — and an unbounded
    parser turns a crafted string into a ``RecursionError`` that escapes as an
    unrelated crash instead of a clean refusal.
    """

    #: Every level of ``_expr -> _term -> _power -> _unary -> _primary`` costs a
    #: few frames, so this leaves generous headroom under CPython's limit while
    #: capping the damage from nesting.
    MAX_DEPTH = 48

    def __init__(self, tokens: list[_Token]) -> None:
        self._tokens = tokens
        self._index = 0
        self._depth = 0

    # -- token helpers -------------------------------------------------
    @property
    def _current(self) -> _Token:
        return self._tokens[self._index]

    def _advance(self) -> _Token:
        token = self._tokens[self._index]
        self._index += 1
        return token

    def _accept(self, text: str) -> bool:
        if self._current.kind == "op" and self._current.text == text:
            self._index += 1
            return True
        return False

    def _expect(self, text: str) -> None:
        if not self._accept(text):
            raise CalcError(f"expected {text!r} at position {self._current.pos}, found {self._current.text!r}")

    # -- grammar -------------------------------------------------------
    def parse(self) -> float:
        value = self._expr()
        if self._current.kind != "end":
            raise CalcError(f"unexpected {self._current.text!r} at position {self._current.pos}")
        return value

    def _expr(self) -> float:
        self._enter()
        try:
            return self._expr_inner()
        finally:
            self._depth -= 1

    def _enter(self) -> None:
        self._depth += 1
        if self._depth > self.MAX_DEPTH:
            raise CalcError(
                f"expression is nested more than {self.MAX_DEPTH} levels deep"
            )

    def _expr_inner(self) -> float:
        value = self._term()
        while self._current.kind == "op" and self._current.text in {"+", "-"}:
            operator = self._advance().text
            right = self._term()
            value = value + right if operator == "+" else value - right
        return value

    def _term(self) -> float:
        value = self._power()
        while self._current.kind == "op" and self._current.text in {"*", "/", "//", "%"}:
            operator = self._advance().text
            right = self._power()
            if operator in {"/", "//", "%"} and right == 0:
                raise CalcError("division by zero")
            if operator == "*":
                value *= right
            elif operator == "/":
                value = value / right
            elif operator == "//":
                value = value // right
            else:
                value = value % right
        return value

    def _power(self) -> float:
        base = self._unary()
        if self._accept("**") or self._accept("^"):
            exponent = self._power()  # right-associative: 2 ** 3 ** 2 == 512
            try:
                return base**exponent
            except (OverflowError, ValueError) as exc:
                raise CalcError(str(exc)) from exc
        return base

    def _unary(self) -> float:
        # Bounded like _expr: "-----1" is unbounded input for a recursive rule.
        self._enter()
        try:
            if self._accept("-"):
                return -self._unary()
            if self._accept("+"):
                return self._unary()
            return self._primary()
        finally:
            self._depth -= 1

    def _primary(self) -> float:
        token = self._current
        if token.kind == "number":
            self._advance()
            return float(token.text.replace(",", ""))
        if token.kind == "op" and token.text == "(":
            self._advance()
            value = self._expr()
            self._expect(")")
            return value
        if token.kind == "name":
            self._advance()
            lowered = token.text.lower()
            if self._current.kind == "op" and self._current.text == "(":
                self._advance()
                args: list[float] = []
                if not (self._current.kind == "op" and self._current.text == ")"):
                    args.append(self._expr())
                    while self._accept(","):
                        args.append(self._expr())
                self._expect(")")
                return self._call(lowered, args, token.pos)
            if lowered in CONSTANTS:
                return CONSTANTS[lowered]
            raise CalcError(f"unknown name {token.text!r} at position {token.pos}")
        raise CalcError(f"unexpected {token.text!r} at position {token.pos}")

    @staticmethod
    def _call(name: str, args: list[float], pos: int) -> float:
        function = _FUNCTIONS.get(name)
        if function is None:
            raise CalcError(f"unknown function {name!r} at position {pos}")
        try:
            return float(function(*args))  # type: ignore[operator]
        except CalcError:
            raise
        except TypeError as exc:
            raise CalcError(f"{name}() called with the wrong number of arguments: {exc}") from exc
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            raise CalcError(f"{name}(): {exc}") from exc


def calculate_expression(source: str) -> float:
    """Evaluate ``source`` and return a float.

    Raises:
        CalcError: on any syntax error, unknown name, or domain error. Callers
            convert this into a ``ToolResult`` so the agent can retry.
    """
    if not source or not source.strip():
        raise CalcError("empty expression")
    if len(source) > 512:
        raise CalcError("expression is too long (limit 512 characters)")
    return _Parser(_tokenize(source)).parse()
