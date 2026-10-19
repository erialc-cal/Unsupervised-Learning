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
    validator: Callable[[Any, Any], bool] | None = None

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


#### CHECKS FOR LAB 01

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


#### CHECKS FOR LAB 02
# Graphs exclude self-edges. kNN columns identify each point's k nearest
# OTHER observations, following the question's S[i, j] convention.
_CLUSTER_X = np.array([[0., 0.], [1., 0.], [4., 0.]])
_EPS = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])
_KNN = np.array([[0., 1., 0.], [1., 0., 1.], [0., 0., 0.]])
_GAUSS = np.array([[0., np.exp(-1/8), np.exp(-2)],
                   [np.exp(-1/8), 0., np.exp(-9/8)],
                   [np.exp(-2), np.exp(-9/8), 0.]])
_LAPLACIAN = np.array([[1., -1., 0.], [-1., 3., -2.], [0., -2., 2.]])


def _eigenvectors_valid(actual, expected):
    L, k = expected
    U = np.asarray(actual)
    if U.shape != (len(L), k) or not np.isfinite(U).all():
        return False
    # Eigenvectors can change sign or rotate within a repeated eigenspace.
    values = np.linalg.eigvalsh(L)[:k]
    return (np.allclose(U.T @ U, np.eye(k)) and
            np.allclose(L @ U, U * values, atol=1e-8))


def _partition_valid(actual, expected):
    # The notebook returns a fitted KMeans object, rather than labels alone.
    labels = np.asarray(getattr(actual, "labels_", None))
    expected = np.asarray(expected)
    return (labels.shape == expected.shape and
            np.array_equal(labels[:, None] == labels[None, :],
                           expected[:, None] == expected[None, :]))


_CHECKS.update({
    "epsilon_neighbor": ("Lab 2, Ex1 Q1: epsilon graph", (
        Case((_CLUSTER_X, 1.), _EPS),
        # Distinguishes Euclidean distance from squared Euclidean distance.
        Case((_CLUSTER_X, 2.), _EPS),
        Case((np.zeros((2, 2)), 0.), np.array([[0., 1.], [1., 0.]])),
    )),
    "k_nearest_neighbor": ("Lab 2, Ex1 Q1: directed kNN graph", (
        Case((_CLUSTER_X, 1), _KNN),
        Case((_CLUSTER_X, 2), np.ones((3, 3)) - np.eye(3)),
    )),
    "gaussian_kernel_neighbor": ("Lab 2, Ex1 Q1: Gaussian graph", (
        Case((_CLUSTER_X, 2.), _GAUSS),
        Case((np.zeros((3, 2)), 1.), np.ones((3, 3)) - np.eye(3)),
    )),
    "compute_similarity": ("Lab 2, Ex1 Q1: graph dispatch", (
        Case((_CLUSTER_X, "epsilon", 1, 2., 2.), _EPS),
        Case((_CLUSTER_X, "knn", 1, 2., 2.), _KNN),
        Case((_CLUSTER_X, "gaussian", 1, 2., 2.), _GAUSS),
    )),
    "graph_laplacian": ("Lab 2, Ex1 Q2: unnormalized Laplacian", (
        Case((np.array([[0., 1., 0.], [1., 0., 2.], [0., 2., 0.]]),), _LAPLACIAN),
        Case((np.zeros((3, 3)),), np.zeros((3, 3))),
        Case((_KNN,), np.diag(_KNN.sum(axis=1)) - _KNN),
    )),
    "get_eigenvectors": ("Lab 2, Ex1 Q3: smallest eigenvectors", (
        Case((_LAPLACIAN, 2), (_LAPLACIAN, 2), _eigenvectors_valid),
        Case((np.diag([3., 0., 0., 5.]), 2),
             (np.diag([3., 0., 0., 5.]), 2), _eigenvectors_valid),
    )),
    "spectral_clustering": ("Lab 2, Ex1 Q4: spectral clustering", (
        Case((np.array([[0.], [0.1], [10.], [10.1]]), 2, "epsilon", 0.5),
             [0, 0, 1, 1], _partition_valid),
        Case((np.array([[0.], [0.1], [10.], [10.1], [20.], [20.1]]),
              3, "epsilon", 0.5), [0, 0, 1, 1, 2, 2], _partition_valid),
    )),
})



# - **Ex1 Q1–Q4:** Run the `grader.check(...)` cells beside the function definitions. Eigenvector signs and repeated-eigenspace bases may vary; cluster IDs may be permuted.
# - **Ex1 Q5:** `grader.check_clusterings(X, {"implementation": sp, "spectral": sklearn_labels, "kmeans": km_labels})` after the toy-data comparison.
# - **Ex2 Q1:** `grader.check_datasets(dataset_dict)` checks the three datasets' sizes and labels and the uniform rectangles. Review Gaussian means, variances and mixture probabilities against the prompt as well.
# - **Ex2 Q2:** Within the loop over datasets, run `grader.check_clusterings(X, {"spectral": sp_labels, "kmeans": km_labels, "hierarchical": hc_labels, "dbscan": db_labels}, expected_methods=4)`. Review the 3 by 5 comparison plot manually.
# - **Ex2 Q3:** Save the three SciPy partitions and the three sklearn partitions in lists, then run `grader.check_hierarchy(X, scipy_hc, dist_thresh, scipy_partitions, sklearn_partitions)`. Use the same linkage method and thresholds in both libraries. To validate the reconstructed sklearn linkage and its leaf counts, repeat with `linkage_matrix(AC)` as the second argument. Review both dendrograms manually.
# - **Ex3 Q1:** Save the predicted labels for each parameter in `labels_by_parameter`, then run `grader.check_silhouette(X, labels_by_parameter, silhouette)`. Keep only parameters producing between 2 and N-1 clusters, and use the same ordering for both lists.
# - **Ex3 Q2.1:** `grader.check_external_metrics(y, labels_by_parameter, {"RI": RI_list, "ARI": ARI_list, "NMI": NMI_list, "AMI": AMI_list})`. Save partitions during the corresponding metric loop.
# - **Ex3 Q2.2–Q2.3:** Review the proposed validation approach and its applications manually; `my_approach` has no specified return contract.
# - **Ex3 Q2.4:** Run `grader.check_stability(stability)` beside the function. This public fixture controls `np.random.choice` to check duplicate handling and intersection alignment; implementations using a local Generator need an equivalent injectable sampling fixture. Review the full K sweep and error-bar plots separately.
# - **Ex3 Q2.5 and Bonus:** Review conclusions, cluster descriptions and package exploration manually.



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

    def check_clusterings(self, X, clusterings, expected_methods=3):
        """Ex1 Q5 / Ex2 Q2: pass {method_name: labels_or_fitted_model}."""
        assert len(clusterings) == expected_methods, "Unexpected number of methods."
        X = np.asarray(X)
        assert X.ndim == 2 and np.isfinite(X).all(), "Expected finite data matrix."
        for name, result in clusterings.items():
            labels = np.asarray(getattr(result, "labels_", result))
            assert labels.shape == (len(X),), f"{name}: expected one label per row."
            assert np.issubdtype(labels.dtype, np.number) and np.isfinite(labels).all(), (
                f"{name}: labels must be finite numbers (DBSCAN noise -1 is allowed)."
            )
        print("✓ Clustering outputs: shapes and labels passed; review plots manually")

    def check_datasets(self, datasets):
        """Ex2 Q1: check dataset_dict structure and specified uniform support.

        These are structural checks, not a statistical test of the simulation.
        """
        bounds = np.array([[[-1, -.5], [-.5, 1]],
                           [[-.1, .2], [.8, 1]], [[.3, -1], [.7, -.2]]])
        for name, k in (("GMM", 3), ("moons", 2), ("uniform", 3)):
            X = np.asarray(datasets[name]["data"])
            y = np.asarray(datasets[name]["label"])
            assert X.shape == (1000, 2) and np.isfinite(X).all(), name
            assert y.shape == (1000,) and set(np.unique(y)) == set(range(k)), name
            if name == "uniform":
                for j in range(k):
                    assert ((X[y == j] >= bounds[j, 0]) &
                            (X[y == j] <= bounds[j, 1])).all(), (
                        f"Uniform component {j}: points outside specified rectangle."
                    )
        print("✓ Ex2 Q1: dataset sizes, labels, finite values and uniform support passed")

    def check_hierarchy(self, X, Z, thresholds, scipy_labels, sklearn_labels):
        """Ex2 Q3: validate linkage and comparable partitions at 3 thresholds.

        Use matching linkage methods and Euclidean distance in both libraries.
        Choose thresholds away from merge distances (boundary conventions differ).
        """
        from scipy.cluster.hierarchy import is_valid_linkage, fcluster
        from sklearn.metrics import adjusted_rand_score

        Z = np.asarray(Z)
        assert Z.shape == (len(X) - 1, 4), "Incorrect linkage shape."
        is_valid_linkage(Z, throw=True)
        counts = {}
        for i, row in enumerate(Z):
            count = sum(1 if child < len(X) else counts[int(child)] for child in row[:2])
            assert row[3] == count, "Linkage column 4 must contain leaf counts."
            counts[len(X) + i] = count
        assert len(thresholds) == len(scipy_labels) == len(sklearn_labels) == 3
        for t, left, right in zip(thresholds, scipy_labels, sklearn_labels):
            expected = fcluster(Z, t=t, criterion="distance")
            for labels in (left, right):
                assert np.asarray(labels).shape == (len(X),)
                assert np.isclose(adjusted_rand_score(expected, labels), 1.), (
                    "Partitions disagree with linkage at the given threshold."
                )
        print("✓ Ex2 Q3: linkage counts and threshold partitions passed; review dendrograms manually")

    def check_silhouette(self, X, labels_by_parameter, scores):
        """Ex3 Q1: compare each reported score to the associated partition."""
        from sklearn.metrics import silhouette_score

        assert len(labels_by_parameter) == len(scores) >= 2, "Provide a parameter sweep."
        for labels, score in zip(labels_by_parameter, scores):
            assert np.asarray(labels).shape == (len(X),)
            assert _same(score, float(silhouette_score(X, labels))), "Incorrect silhouette score."
        print("✓ Ex3 Q1: silhouette scores passed; review parameter choices and plot manually")

    def check_external_metrics(self, y, labels_by_parameter, scores):
        """Ex3 Q2.1: scores maps RI, ARI, NMI, AMI to sequences."""
        from sklearn.metrics import (rand_score, adjusted_rand_score,
                                     normalized_mutual_info_score, adjusted_mutual_info_score)

        assert len(labels_by_parameter) >= 2, "Provide a parameter sweep."
        for name, metric in zip(("RI", "ARI", "NMI", "AMI"),
                                (rand_score, adjusted_rand_score,
                                 normalized_mutual_info_score, adjusted_mutual_info_score)):
            assert len(scores[name]) == len(labels_by_parameter), name
            for labels, score in zip(labels_by_parameter, scores[name]):
                assert np.asarray(labels).shape == np.asarray(y).shape
                assert _same(score, float(metric(y, labels))), f"Incorrect {name}."
        print("✓ Ex3 Q2.1: external metrics passed")

    def check_stability(self, function):
        """Ex3 Q2.4: check intersection alignment, mean and standard error.

        Fixed subsamples include duplicate indices and different row orders.
        The fixture model gives each observation its own stable identity label.
        """
        from unittest.mock import patch
        from sklearn.base import BaseEstimator

        class IdentityClusters(BaseEstimator):
            def fit_predict(self, X):
                return np.asarray(X)[:, 0].astype(int)

        draws = [np.array([4, 1, 1, 3]), np.array([3, 4, 0, 0]),
                 np.array([2, 0, 0, 5]), np.array([5, 2, 1, 1])]
        seen = []

        def criterion(left, right):
            assert np.array_equal(np.sort(left), [3, 4] if not seen else [2, 5]), (
                "Compare only the shared original observations, once each."
            )
            assert np.array_equal(left, right), "Align labels by original row index."
            seen.append(True)
            return 0.25 if len(seen) == 1 else 0.75

        with patch("numpy.random.choice", side_effect=draws):
            result = function(np.arange(6.)[:, None], 2, IdentityClusters(),
                              M=2, sub=2/3, criterion=criterion)
        assert len(seen) == 2, "Evaluate the criterion once per repetition."
        # Accept population or sample standard deviation for the estimated SE.
        assert any(_same(result, expected) for expected in
                   ((0.5, 0.25 / np.sqrt(2)), (0.5, 0.25))), (
            "Return (mean validity, standard deviation / sqrt(M))."
        )
        print("✓ Ex3 Q2.4: stability alignment, averaging and standard error passed")

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
            matches = case.validator or _same
            if not matches(actual, case.expected):
                raise AssertionError(
                    f"{label}, case {number}: expected "
                    f"{case.expected!r}, got {actual!r}"
                )

        print(f"✓ {label}: all {len(cases)} public checks passed")


grader = Grader()
