from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from multiscat.lobatto import LobattoMetadata, get_derivative_polynomials


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
        result = LobattoMetadata(n, 2)
        np.testing.assert_allclose(result.values - 1, expected_points, rtol=1e-5)
        np.testing.assert_allclose(result.weights, expected_weights, rtol=1e-5)


@pytest.fixture
def random_n() -> int:
    """Fixture to generate a random n between 2 and 20."""
    rng = np.random.default_rng()
    return rng.integers(2, 28).item()


def test_lobatto_points_symmetry(random_n: int) -> None:
    result = LobattoMetadata(random_n, 2.0)
    np.testing.assert_allclose(
        result.values - 1.0,
        -(result.values[::-1] - 1.0),
        err_msg=f"Points not symmetric for n={random_n}",
        atol=2e-7,
    )
    np.testing.assert_allclose(
        result.weights,
        result.weights[::-1],
        err_msg=f"Weights not symmetric for n={random_n}",
        atol=1e-10,
    )


def _lobatto_from_fortran(
    a: float,
    b: float,
    n: int,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    """
    Calculate lobatto weights based on the fortran approach.

    Coppied directly from the code provided by <https://doi.org/10.1007/978-94-015-8240-7_4>

    Parameters
    ----------
    a : float
    b : float
    n : int

    Returns
    -------
    tuple[
        np.ndarray[Any, np.dtype[np.float64]],
        np.ndarray[Any, np.dtype[np.float64]],
    ]

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
    result = LobattoMetadata(random_n, 1)
    fortran_result = _lobatto_from_fortran(0, 1, random_n)
    np.testing.assert_allclose(result.values, fortran_result[0], atol=1e-8)
    np.testing.assert_allclose(result.weights, fortran_result[1], atol=1e-8)


def get_lobatto_derivatives_explicit(
    points: LobattoMetadata,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """
    Calculate the derivative matrix u_i'(R_j) for the lobatto basis.

    Parameters
    ----------
    points : LobattoPoints

    Returns
    -------
    np.ndarray[Any, np.dtype[np.float64]]

    """
    # The derivatives can be evaluated analytically
    # u_i'(R_i) = \sum_j=0 M+1 (R_i - R_j)^-1
    # or for j=\=i
    # u_i'(R_j) = (R_i - R_j)^-1 product_0^M+1 (R_j-R_k) / (R_i - R_k)
    # Where the product excludes k=j and k=i
    n_points = points.values.size

    # Calculate the reciprocal of differences (R_i - R_j)^-1, ignoring the diagonal
    diff = points.values[:, np.newaxis] - points.values[np.newaxis, :]
    reciprocal_diff = np.where(diff != 0, 1.0 / diff, 0)

    # Calculate product_k=0^M+1 (R_j-R_k) / (R_i - R_k)
    mask = np.eye(n_points, dtype=bool)
    products = np.prod(
        np.where(
            # Ignoring the zero elements from the product
            mask[np.newaxis, :, :] | mask[:, np.newaxis, :],
            1,
            (diff[np.newaxis, :, :] * reciprocal_diff[:, np.newaxis, :]),
        ),
        axis=2,
    )

    u_derivative = reciprocal_diff * products
    # Calculate diagonal elements seperately
    # u_i'(R_i) = \sum_j=0 M+1 (R_i - R_j)^-1
    u_derivative[np.arange(n_points), np.arange(n_points)] = np.sum(
        reciprocal_diff,
        axis=1,
    )
    return u_derivative


def test_lobatto_derivatives_against_explicit(random_n: int) -> None:
    lobatto_points = LobattoMetadata(random_n, 2)

    polynomial_derivatives = np.array(
        [p(lobatto_points.values) for p in get_derivative_polynomials(lobatto_points)],
        dtype=np.float64,
    )

    derivatives = get_lobatto_derivatives_explicit(lobatto_points)
    np.testing.assert_allclose(
        derivatives,
        polynomial_derivatives,
        equal_nan=True,
        atol=1e-7,
    )
