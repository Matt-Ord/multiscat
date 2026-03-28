from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from slate_core import FundamentalBasis
from slate_quantum import Operator
from slate_quantum.operator import OperatorBasis, operator_basis

if TYPE_CHECKING:
    from slate_core.metadata import BarycentricMetadata


def get_unnormalized_polynomials(
    metadata: BarycentricMetadata,
) -> list[np.polynomial.Polynomial]:
    """Get the lobatto polynomials for the lobatto points."""
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
    metadata: BarycentricMetadata,
) -> list[np.polynomial.Polynomial]:
    """Get the weighted lobatto polynomials for the lobatto points."""
    return [
        p * w
        for (p, w) in zip(
            get_unnormalized_polynomials(metadata),
            metadata.basis_weights,
            strict=True,
        )
    ]


def get_derivative_polynomials(
    metadata: BarycentricMetadata,
) -> list[np.polynomial.Polynomial]:
    """Get the derivative polynomials for the lobatto points."""
    return [p.deriv() for p in get_polynomials(metadata)]


def get_barycentric_derivatives(
    metadata: BarycentricMetadata,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    r"""
    Compute the derivative matrix M_ij = u_i'(x_j) using barycentric interpolation.

    For basis functions defined by lagrange interpolation, the interpolating
    polynomial is given by:
    .. math::
        u_i(x) = \frac{1}{\sqrt{w_i}} \prod_{k=0}^{M+1} \frac{x - x_k}{x_i - x_k}

    This function computes the derivative matrix for these basis functions.
    """
    values = metadata.values
    # Scale values to [0, 1] to keep numbers well-behaved
    scale_factor = 1.0 / (values[-1] - values[0])
    scaled_values = (values - values[0]) * scale_factor

    difference = scaled_values[:, None] - scaled_values[None, :]
    # Fill diagonal with 1.0 to avoid log(0) and division by zero.
    np.fill_diagonal(difference, 1.0)

    # Compute log|w_i| = sum_{k!=i} log|x_i - x_k|
    log_diff = np.log(np.abs(difference))
    log_w = np.sum(log_diff, axis=1)

    # Track the signs sign(w_i) = prod_{k!=i} sign(x_i - x_k)
    sign_w = np.prod(np.sign(difference), axis=1)

    # Assemble the ratio w_i / w_j = (sign_i / sign_j) * exp(log_w_i - log_w_j)
    # Note: 1 / sign_j is mathematically identical to sign_j
    log_ratio = log_w[:, None] - log_w[None, :]
    sign_ratio = sign_w[:, None] * sign_w[None, :]
    weight_ratio = sign_ratio * np.exp(log_ratio)

    # Off-diagonal elements: D_ij = (w_i / w_j) / (x_i - x_j)
    scaled_derivatives = weight_ratio / difference

    # Diagonal elements: D_ii = sum_{k!=i} 1 / (x_i - x_k)
    inv_diff = 1.0 / difference
    np.fill_diagonal(inv_diff, 0.0)  # Ensure we don't sum the dummy 1.0s

    diag_mask = np.diag_indices(metadata.fundamental_size)
    scaled_derivatives[diag_mask] = np.sum(inv_diff, axis=1)

    derivatives = scaled_derivatives * scale_factor

    return derivatives.T * metadata.basis_weights[:, np.newaxis]


def get_barycentric_kinetic_operator[M1: BarycentricMetadata](
    metadata: M1,
) -> Operator[OperatorBasis[M1], np.dtype[np.float64]]:
    """
    Get the kinetic operator grad squared in a barycentric basis.

    Formula for this are taken from:
    "QUANTUM SCATTERING VIA THE LOG DERIVATIVE OF THE KOHN VARIATIONAL PRINCIPLE"
    D. E. Manolopoulos and R. E. Wyatt, Chem. Phys. Lett., 1988, 152,23
    """
    # We use the barycentric metadata to get the lobatto points
    # and the weights.
    # We make use of the formula
    # T_ij = \sum_k=0 M+1 \omega_k u_i'(R_k) u'_j(R_k)
    # to calculate the kinetic matrix T_ij
    derivatives = get_barycentric_derivatives(metadata)
    return Operator(
        operator_basis(FundamentalBasis(metadata)).upcast(),
        -np.einsum(
            "k,ik,jk->ij",
            1.0 / np.square(metadata.basis_weights),
            derivatives,
            derivatives,
        ),
    )
