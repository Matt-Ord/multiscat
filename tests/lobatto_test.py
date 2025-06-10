from __future__ import annotations

import numpy as np
from slate_core.metadata import Domain, LobattoSpacedMetadata

from multiscat.polynomial import (
    get_barycentric_derivatives,
    get_barycentric_kinetic_operator,
    get_derivative_polynomials,
    get_polynomials,
)


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


def test_derivatives_scaling() -> None:
    rng = np.random.default_rng()
    random_n = rng.integers(2, 20).item()
    random_n = 3
    metadata_0 = LobattoSpacedMetadata(
        random_n,
        domain=Domain(delta=2.0),
    )
    metadata_1 = LobattoSpacedMetadata(
        random_n,
        domain=Domain(delta=1.0),
    )

    # If we halve the domain delta, the derivatives should double
    # becuase the basis functions are halved in width.
    # I'm not so sure that this is working
    np.testing.assert_allclose(
        4 * (get_barycentric_kinetic_operator(metadata_0).raw_data),
        (get_barycentric_kinetic_operator(metadata_1).raw_data),
        atol=1e-7,
    )


def test_polynomial_points() -> None:
    rng = np.random.default_rng()
    random_n = rng.integers(2, 30).item()
    lobatto_points = LobattoSpacedMetadata(
        random_n,
        domain=Domain(delta=2.0),
    )

    polynomial_values = np.array(
        [p(lobatto_points.values) for p in get_polynomials(lobatto_points)],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        polynomial_values,
        np.diag(lobatto_points.basis_weights),
        atol=1e-5,
    )
