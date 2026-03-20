from __future__ import annotations

import numpy as np
from multiscat_fortran import get_lobatto_weights
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
    # Our polynomials are normalized such that
    # u_i(R_j) = delta_{i,j} / sqrt(w_j)
    # so we need to divide the usual fortran derivatives by
    # the square root of the weights
    np.testing.assert_allclose(
        polynomial_derivatives,
        derivatives,
        atol=1e-7,
    )


def test_lobatto_points_and_weights_match_fortran() -> None:
    random_n = np.random.default_rng().integers(50, 100).item()
    zmin = -0.4
    zmax = 1.7
    lobatto_points = LobattoSpacedMetadata(
        random_n,
        domain=Domain(start=zmin, delta=(zmax - zmin)),
    )

    weights_raw, points_raw, ierr_raw = get_lobatto_weights(
        zmin=float(zmin),
        zmax=float(zmax),
        node_count=int(random_n),
    )
    weights = np.asarray(weights_raw, dtype=np.float64)
    points = np.asarray(points_raw, dtype=np.float64)
    ierr = int(ierr_raw)

    if ierr != 0:
        msg = f"Fortran get_lobatto_weights failed with error code {ierr}"
        raise RuntimeError(msg)

    np.testing.assert_allclose(points, lobatto_points.values, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        weights,
        1.0 / lobatto_points.basis_weights,
        rtol=1e-12,
        atol=1e-12,
    )
