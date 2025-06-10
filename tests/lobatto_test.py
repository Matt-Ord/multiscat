from __future__ import annotations

import numpy as np
from slate_core.metadata import Domain, LobattoSpacedMetadata

from multiscat.polynomial import get_barycentric_derivatives, get_derivative_polynomials


def test_lobatto_derivatives_against_explicit() -> None:
    rng = np.random.default_rng()
    random_n = rng.integers(2, 20).item()
    lobatto_points = LobattoSpacedMetadata(
        random_n,
        domain=Domain(delta=2.0),
    )

    polynomial_derivatives = np.array(
        [p(lobatto_points.values) for p in get_derivative_polynomials(lobatto_points)],
        dtype=np.float64,
    )

    derivatives = get_barycentric_derivatives(lobatto_points)
    np.testing.assert_allclose(polynomial_derivatives, derivatives, atol=1e-7)
