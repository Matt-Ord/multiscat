from __future__ import annotations

import numpy as np
import pytest
from slate_core.metadata import LabelSpacing, LobattoSpacedMetadata

from multiscat.polynomial import get_derivative_polynomials


@pytest.fixture
def random_n() -> int:
    """Fixture to generate a random n between 2 and 20."""
    rng = np.random.default_rng()
    return rng.integers(2, 20).item()


def get_lobatto_derivatives_explicit(
    points: LobattoSpacedMetadata,
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
    with np.errstate(divide="ignore"):
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
    # u_i'(R_j) = \sum_j=0 M+1 (R_i - R_j)^-1
    u_derivative[np.arange(n_points), np.arange(n_points)] = np.sum(
        reciprocal_diff,
        axis=1,
    )
    return u_derivative


def test_lobatto_derivatives_against_explicit(random_n: int) -> None:
    lobatto_points = LobattoSpacedMetadata(
        random_n,
        spacing=LabelSpacing(delta=2.0),
    )

    polynomial_derivatives = np.array(
        [p(lobatto_points.values) for p in get_derivative_polynomials(lobatto_points)],
        dtype=np.float64,
    )

    derivatives = get_lobatto_derivatives_explicit(lobatto_points)
    # Our polynomials are normalized such that
    # u_i(R_j) = delta_{i,j} / sqrt(w_j)
    # so we need to divide the usual fortran derivatives by
    # the square root of the weights
    np.testing.assert_allclose(
        derivatives / np.sqrt(lobatto_points.quadrature_weights[:, np.newaxis]),
        polynomial_derivatives,
        equal_nan=True,
        atol=1e-7,
    )
