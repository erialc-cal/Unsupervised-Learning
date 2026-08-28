"""Public, student-facing notebook checks.

The test registry lives outside the notebooks so every check has the same API and
presentation. These checks provide quick feedback; they are not hidden grading
tests and are not intended to prove that a solution is correct for every input.
"""

from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isclose
from pathlib import Path
from typing import Any
import pandas as pd

@dataclass(frozen=True)
class Case:
    arguments: tuple[Any, ...]
    expected: Any

_COURSE_ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(_COURSE_ROOT / "data" / "literary_styles_dataset.csv")

_CHECKS: dict[str, tuple[str, tuple[Case, ...]]] = {
    "exercise_1": (
        "Exercise 1",
        (Case((10, 4), 6), Case((-2, 3), -5), Case((2.5, 1.0), 1.5)),
    ),
    "exercise_2": (
        "Exercise 2",
        (Case(([2, 4, 6],), 4), Case(([1.5, 2.5],), 2.0), Case(([-3],), -3)),
    ),
    "exercise_3": (
        "Exercise 3",
        (Case(([1, 2, 3],), [-1, 0, 1]), Case(([10, 10],), [0, 0])),
    ),
    "exercise_4": ("Exercise 4", (Case((df,), (962, 75)),)),
}


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        try:
            return len(actual) == len(expected) and all(
                _same(a, e) for a, e in zip(actual, expected)
            )
        except TypeError:
            return False
    try:
        return bool(actual == expected)
    except (TypeError, ValueError):
        return False


class Grader:
    """Run named public checks for a student's function."""

    def check(self, check_id: str, function: Callable[..., Any]) -> None:
        if check_id not in _CHECKS:
            choices = ", ".join(sorted(_CHECKS))
            raise KeyError(f"Unknown check {check_id!r}. Available checks: {choices}")
        if not callable(function):
            raise TypeError("Pass your function itself, without parentheses.")

        label, cases = _CHECKS[check_id]
        for number, case in enumerate(cases, start=1):
            try:
                actual = function(*case.arguments)
            except Exception as error:
                raise AssertionError(
                    f"{label}, case {number}: raised "
                    f"{type(error).__name__}: {error}"
                ) from error
            if not _same(actual, case.expected):
                raise AssertionError(
                    f"{label}, case {number}: expected "
                    f"{case.expected!r}, got {actual!r}"
                )

        print(f"✓ {label}: all {len(cases)} public checks passed")


grader = Grader()
