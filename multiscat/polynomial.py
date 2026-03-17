from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from slate_core.metadata import PERIODIC_FEATURE

if TYPE_CHECKING:
    from slate_core.metadata import SpacedMetadata


def get_unnormalized_polynomials(
    metadata: SpacedMetadata[np.dtype[np.floating]],
) -> list[np.polynomial.Polynomial]:
    """Get the lobatto polynomials for the lobatto points."""
    if PERIODIC_FEATURE in metadata.features:
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
            metadata.basis_weights,
            strict=True,
        )
    ]


def get_derivative_polynomials(
    metadata: SpacedMetadata[np.dtype[np.floating]],
) -> list[np.polynomial.Polynomial]:
    """Get the derivative polynomials for the lobatto points."""
    return [p.deriv() for p in get_polynomials(metadata)]


# TODO: we should specify the interpolation type, e.g. lagrange  # noqa: FIX002
# vs fourier in the metadata.
def get_barycentric_derivatives(
    metadata: SpacedMetadata[np.dtype[np.floating]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    r"""
    Compute the derivative matrix M_ij = u_i'(x_j) using barycentric interpolation.

    For basis functions defined by lagrange interpolation, the interpolating
    polynomial is given by:
    .. math::
        u_i(x) = \frac{1}{\sqrt{w_i}} \prod_{k=0}^{M+1} \frac{x - x_k}{x_i - x_k}

    This function computes the derivative matrix for these basis functions.

    """
    # We scale the values to the range [0, 1] to avoid numerical issues
    values = metadata.values
    scale_factor = 1 / (values[-1] - values[0])
    scaled_values = (values - values[0]) * scale_factor

    difference = scaled_values[:, None] - scaled_values[None, :]
    np.fill_diagonal(difference, 1.0)  # avoid divide by zero
    # λ_i = 1 / Π_{k ≠ i} (x_i - x_k)
    barycentric_weights = 1.0 / np.prod(difference, axis=1)
    scaled_derivatives = barycentric_weights[None, :] / (
        difference * barycentric_weights[:, None]
    )

    diag_mask = np.diag_indices(metadata.fundamental_size)
    scaled_derivatives[diag_mask] = 0
    scaled_derivatives[diag_mask] = -np.sum(scaled_derivatives, axis=1)

    derivatives = scaled_derivatives * scale_factor

    return derivatives.T / np.sqrt(metadata.basis_weights[:, np.newaxis])
