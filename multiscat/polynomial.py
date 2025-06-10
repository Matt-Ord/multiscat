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

    return derivatives.T * metadata.basis_weights[:, np.newaxis]


def get_barycentric_kinetic_operator[M1: BarycentricMetadata](
    metadata: M1,
) -> Operator[OperatorBasis[M1], np.dtype[np.complex128]]:
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
            1 / np.square(metadata.basis_weights),
            derivatives,
            derivatives,
        ),
    )
