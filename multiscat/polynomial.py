from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from slate_core.metadata import SpacedMetadata


def get_unnormalized_polynomials(
    metadata: SpacedMetadata[np.dtype[np.floating]],
) -> list[np.polynomial.Polynomial]:
    """Get the lobatto polynomials for the lobatto points."""
    if metadata.is_periodic:
        msg = "Currently we do not support periodic metadata."
        raise NotImplementedError(msg)
    domain = np.array([metadata.values[0], metadata.values[-1]])
    polynomials = [
        cast(
            "np.polynomial.Polynomial",
            np.polynomial.Polynomial.fromroots(np.delete(metadata.values, i), domain),  # type: ignore bad library type
        )
        for i in range(metadata.values.size)
    ]
    return [
        polynomial / polynomial(point)
        for (polynomial, point) in zip(
            polynomials,
            metadata.values,
            strict=True,
        )
    ]


def get_polynomials(
    metadata: SpacedMetadata[np.dtype[np.floating]],
) -> list[np.polynomial.Polynomial]:
    """Get the weighted lobatto polynomials for the lobatto points."""
    return [
        p / np.sqrt(w)
        for (p, w) in zip(
            get_unnormalized_polynomials(metadata),
            metadata.quadrature_weights,
            strict=True,
        )
    ]


def get_derivative_polynomials(
    metadata: SpacedMetadata[np.dtype[np.floating]],
) -> list[np.polynomial.Polynomial]:
    """Get the derivative polynomials for the lobatto points."""
    return [p.deriv() for p in get_polynomials(metadata)]


def get_derivative_matrix(
    metadata: SpacedMetadata[np.dtype[np.floating]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    return np.array(  # type: ignore shape-mismatch
        [p(metadata.values) for p in get_derivative_polynomials(metadata)],
    )
