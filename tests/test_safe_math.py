"""The arithmetic parser.

Security-relevant: this replaces ``eval`` for anything the model can influence,
so the tests care about both correctness and what it refuses.
"""

from __future__ import annotations

import pytest

from prithivi.tools.safe_math import CalcError, calculate_expression


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1+1", 2),
        ("2 * 3", 6),
        ("10/4", 2.5),
        ("10//3", 3),
        ("2**10", 1024),
        ("-5 + 3", -2),
        ("(125*0.18)+3400-500", 2922.5),
        ("1,250 + 500", 1750),
        ("  7 -  9  ", -2),
        ("2^-2", 0.25),
    ],
)
def test_arithmetic(expression: str, expected: float) -> None:
    assert calculate_expression(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("sqrt(16)", 4.0),
        ("abs(-7)", 7),
        ("round(3.14159, 2)", 3.14),
        ("min(3, 1, 2)", 1),
        ("max(3, 1, 2)", 3),
        ("fact(5)", 120),
        ("gcd(12, 18)", 6),
        ("mean(2, 4, 6)", 4),
        ("pi", 3.141592653589793),
    ],
)
def test_functions(expression: str, expected: float) -> None:
    assert calculate_expression(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('rm -rf /')",
        "().__class__.__bases__[0].__subclasses__()",
        "open('/etc/passwd')",
        "eval('1+1')",
        "exec('x=1')",
        "1; import os",
        "lambda: 1",
        "[1,2,3]",
        "{'a': 1}",
        "1 if 1 else 2",
    ],
)
def test_rejects_code(expression: str) -> None:
    """Anything that is not arithmetic must be refused, not evaluated."""
    with pytest.raises(CalcError):
        calculate_expression(expression)


@pytest.mark.parametrize(
    "expression",
    ["1 +", "(1 + 2", "2 **", "sqrt(-1)", "1/0", "fact(-1)", "unknown_fn(1)", ""],
)
def test_rejects_bad_input(expression: str) -> None:
    with pytest.raises(CalcError):
        calculate_expression(expression)


def test_division_by_zero_is_an_error_not_an_exception() -> None:
    """Callers rely on CalcError; a bare ZeroDivisionError would escape the tool."""
    with pytest.raises(CalcError):
        calculate_expression("1/0")


def test_result_is_a_real_number() -> None:
    assert isinstance(calculate_expression("1+1"), (int, float))


@pytest.mark.parametrize("expression", ["1+1", "sqrt(4)", "3*3-1", "fact(4)"])
def test_is_deterministic(expression: str) -> None:
    assert calculate_expression(expression) == calculate_expression(expression)


def test_deep_nesting_does_not_recurse_forever() -> None:
    """Bounded recursion: a pathological input must fail, not smash the stack."""
    with pytest.raises(CalcError):
        calculate_expression("(" * 200 + "1" + ")" * 200)


def test_long_input_is_rejected() -> None:
    with pytest.raises(CalcError):
        calculate_expression("1+" * 5000 + "1")


def test_integer_preference_is_exact() -> None:
    """A whole number must not pick up a float tail: 1250*1.18 -> 1475.0 is fine,
    but 3/3 must not be 1.0000000000000002."""
    assert calculate_expression("3/3") == 1
    assert calculate_expression("10/2") == 5