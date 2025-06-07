from __future__ import annotations

import numpy as np
from slate_core.metadata import LabelSpacing, LobattoSpacedMetadata

from multiscat.polynomial import get_barycentric_derivatives, get_derivative_polynomials


def test_lobatto_derivatives_against_explicit() -> None:
    rng = np.random.default_rng()
    random_n = rng.integers(2, 20).item()
    lobatto_points = LobattoSpacedMetadata(
        random_n,
        spacing=LabelSpacing(delta=2.0),
    )

    polynomial_derivatives = np.array(
        [p(lobatto_points.values) for p in get_derivative_polynomials(lobatto_points)],
        dtype=np.float64,
    )

    derivatives = get_barycentric_derivatives(lobatto_points)
    # Our polynomials are normalized such that
    # u_i(R_j) = delta_{i,j} / sqrt(w_j)
    # so we need to divide the usual fortran derivatives by
    # the square root of the weights
    np.testing.assert_allclose(
        polynomial_derivatives,
        derivatives,
        atol=1e-7,
    )
