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
import numpy as np
from copy import deepcopy

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


# Three unequal, tie-free points keep the public checks fast and interpretable.
_X = np.array([[0.], [1.], [4.]])
_Z = np.array([[0., 0.], [1., 0.], [3., 0.]])
_Q = np.array([[0., 5/16, 1/16], [5/16, 0., 1/8], [1/16, 1/8, 0.]])
_P_uniform = (np.ones((3, 3)) - np.eye(3)) / 6
_GRAD = np.array([[1/6, 0.], [-43/120, 0.], [23/120, 0.]])

def _original_expected(X, sigma):
    # Explicit row-by-row conditional probabilities, then symmetrization.
    weights = np.array([
        [0. if i == j else np.exp(-sum((x-y)**2)/(2*sigma**2))
         for j, y in enumerate(X)] for i, x in enumerate(X)
    ])
    conditional = np.array([row / sum(row) for row in weights])
    return (conditional + conditional.T) / (2 * len(X))

_P = _original_expected(_X, 2.)
# Independent scalar-loop reference for one descent step.
_STEP_GRAD = np.array([
    [sum(4 * (_P[i, j] - _Q[i, j]) * (_Z[i, d] - _Z[j, d])
         / (1 + sum((_Z[i] - _Z[j])**2)) for j in range(3))
     for d in range(2)] for i in range(3)
])
_CHECKS.update({
    "lab1_original_similarities": ("Q1: original similarities", (
        Case((_X, 2.), _P),
        Case((np.zeros((3, 2)), 1.), _P_uniform),
    )),
    "lab1_embedding_similarities": ("Q1: embedding similarities", (
        Case((_Z,), _Q),
        Case((np.zeros((3, 2)),), _P_uniform),
    )),
    "lab1_compute_gradient": ("Q1: gradient", (
        Case((_Q, _Q, _Z), np.zeros_like(_Z)),
        Case((_P_uniform, _Q, _Z), _GRAD),
    )),
    "lab1_gradient_descent": ("Q1: one gradient descent step", (
        Case((_X, 2., _Z, 0.1, 1), _Z - 0.1 * _STEP_GRAD),
    )),
    "lab1_global_preservation": ("Q5: global distance preservation", (
        Case((_X, 3 * _X + 7), 1.),
        # Condensed distances are [1, 4, 3] and [4, 1, 3].
        Case((_X, np.array([[0.], [4.], [1.]])), -13/14),
    )),
    "lab1_local_preservation": ("Q5: local neighborhood preservation", (
        Case((_X, _X, 1), 1.),
        Case((_X, np.array([[0.], [4.], [1.]]), 1), 0.),
        Case((_X, np.array([[0.], [4.], [1.]]), 2), 1.),
    )),
})


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, np.ndarray):
        try:
            actual = np.asarray(actual)
            return actual.shape == expected.shape and bool(
                np.allclose(actual, expected, rtol=1e-7, atol=1e-10)
            )
        except (TypeError, ValueError):
            return False
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

    def check_embedding(self, X: Any, Z: Any) -> None:
        """Check an already-computed 2D embedding without fitting it again."""
        Z = np.asarray(Z)
        assert Z.shape == (len(X), 2), "Expected one 2D point per input row."
        assert np.issubdtype(Z.dtype, np.number), "Embedding must be numeric."
        assert np.isfinite(Z).all(), "Embedding contains NaN or infinity."
        print("✓ Embedding: shape and finite values passed")

    def check(self, check_id: str, function: Callable[..., Any]) -> None:
        if check_id not in _CHECKS:
            choices = ", ".join(sorted(_CHECKS))
            raise KeyError(f"Unknown check {check_id!r}. Available checks: {choices}")
        if not callable(function):
            raise TypeError("Pass your function itself, without parentheses.")

        label, cases = _CHECKS[check_id]
        for number, case in enumerate(cases, start=1):
            try:
                actual = function(*deepcopy(case.arguments))
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
