from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from multiscat.lobatto import get_lobatto_points


def test_lobatto_points_known_results() -> None:
    known_results = {
        2: (np.array([-1.0, 1.0]), np.array([1.0, 1.0])),
        3: (np.array([-1.0, 0.0, 1.0]), np.array([1 / 3, 4 / 3, 1 / 3])),
        4: (
            np.array([-1.0, -np.sqrt(1 / 5), np.sqrt(1 / 5), 1.0]),
            np.array([1 / 6, 5 / 6, 5 / 6, 1 / 6]),
        ),
    }

    for n, (expected_points, expected_weights) in known_results.items():
        result = get_lobatto_points(n)
        np.testing.assert_allclose(result.points, expected_points, rtol=1e-5)
        np.testing.assert_allclose(result.weights, expected_weights, rtol=1e-5)


@pytest.fixture()
def random_n() -> int:
    """Fixture to generate a random n between 2 and 20."""
    rng = np.random.default_rng()
    return rng.integers(2, 20)


def test_lobatto_points_symmetry(random_n: int) -> None:
    result = get_lobatto_points(random_n)
    np.testing.assert_allclose(
        result.points,
        -result.points[::-1],
        err_msg=f"Points not symmetric for n={random_n}",
    )
    np.testing.assert_allclose(
        result.weights,
        result.weights[::-1],
        err_msg=f"Weights not symmetric for n={random_n}",
    )


def _lobatto_from_fortran(
    a: float,
    b: float,
    n: int,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    """Calculate lobatto weights based on the fortran approach.

    Coppied directly from the code provided in
    https://doi.org/10.1007/978-94-015-8240-7_4

    Parameters
    ----------
    a : float
        _description_
    b : float
        _description_
    n : int
        _description_

    Returns
    -------
    _type_
        _description_

    """
    n_unique = (n + 1) // 2
    pi = np.arccos(-1.0)
    shift = 0.5 * (b + a)
    scale = 0.5 * (b - a)
    weight = (b - a) / (n * (n - 1))
    w = np.zeros(n)
    x = np.zeros(n)

    x[0] = a
    w[0] = weight

    for k in range(2, n_unique + 1):
        z = np.cos(pi * (4 * k - 3) / (4 * n - 2))
        p2 = 0.0
        p1 = 1.0
        for _ in range(7):
            p2 = 0.0
            p1 = 1.0

            for j in range(1, n):
                p3 = p2
                p2 = p1
                p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j

            p2 = (n - 1) * (p2 - z * p1) / (1.0 - z * z)
            p3 = (2 * z * p2 - n * (n - 1) * p1) / (1.0 - z * z)
            z = z - p2 / p3

        x[k - 1] = shift - scale * z
        x[n - k] = shift + scale * z
        w[k - 1] = weight / (p1 * p1)
        w[n - k] = w[k - 1]

    x[n - 1] = b
    w[n - 1] = weight

    return x, w


def test_lobatto_points_against_fortran(random_n: int) -> None:
    result = get_lobatto_points(random_n)
    fortran_result = _lobatto_from_fortran(-1, 1, random_n)
    np.testing.assert_allclose(result.points, fortran_result[0], atol=1e-8)
    np.testing.assert_allclose(result.weights, fortran_result[1], atol=1e-8)
