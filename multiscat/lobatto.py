from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Never, Self, cast, override

import numpy as np
import scipy.special  # type: ignore unknown
from slate_core import (
    Basis,
    Ctype,
    FundamentalBasis,
    SimpleMetadata,
    TransformedBasis,
    TupleBasis,
    TupleMetadata,
    basis,
)
from slate_core.basis import AsUpcast
from slate_core.metadata import (
    AxisDirections,
    LengthMetadata,
    SpacedLengthMetadata,
)
from slate_quantum import Operator
from slate_quantum.operator import DiagonalOperatorBasis, OperatorMetadata


class LobattoMetadata(LengthMetadata):
    """Metadata for a Lobatto basis."""

    def __init__(self, n: int, delta: float) -> None:
        self._delta = delta
        super().__init__(n)

    @cached_property
    def weights(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Get the weights of the lobatto points."""
        return get_lobatto_points(self.fundamental_size, (0, self.delta)).weights

    @cached_property
    def points(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Get the points of the lobatto points."""
        return get_lobatto_points(self.fundamental_size, (0, self.delta)).points

    @property
    @override
    def delta(self) -> float:
        return self._delta


type ScatteringBasisMetadata[
    M0: SimpleMetadata = SpacedLengthMetadata,
    M1: SimpleMetadata = LobattoMetadata,
    E: AxisDirections = AxisDirections,
] = TupleMetadata[
    tuple[M0, M0, M1],
    E,
]


def _get_vectors_perpendicular_to(
    vector: np.ndarray[Any, np.dtype[np.floating]],
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.floating]],
    np.ndarray[tuple[int], np.dtype[np.floating]],
]:
    assert vector.size == 3, "Vector must be a 3D vector."  # noqa: PLR2004, S101
    guess = (
        np.array([1, 0, 0]) if abs(vector[0]) > abs(vector[1]) else np.array([0, 1, 0])
    )
    v0 = guess - np.dot(guess, vector) * vector
    v0 /= np.linalg.norm(v0)
    return (v0, np.cross(vector, v0))  # type: ignore bad library type


def _project_x01_axis_directions(
    metadata: AxisDirections,
) -> AxisDirections:
    """
    Project the axis directions from the scattering basis metadata.

    This is used to extract the axis directions from the metadata.
    """
    vx, vy, vz = metadata.vectors
    v0, v1 = _get_vectors_perpendicular_to(vz)

    a_plane = np.array([np.dot(vx, v0), np.dot(vx, v1)])
    b_plane = np.array([np.dot(vy, v0), np.dot(vy, v1)])
    return AxisDirections(vectors=(a_plane, b_plane))


def get_split_scattering_metadata[
    M0: SimpleMetadata,
    M1: SimpleMetadata,
](
    metadata: ScatteringBasisMetadata[M0, M1, AxisDirections],
) -> tuple[
    TupleMetadata[tuple[M0, M0], AxisDirections],
    M1,
]:
    """
    Get the metadata for the split scattering basis.

    This is used to split the metadata into its components.
    """
    directions_x01 = _project_x01_axis_directions(metadata.extra)
    return (
        TupleMetadata(metadata.children[:2], directions_x01),
        metadata.children[2],
    )


type CloseCouplingBasis[
    M0: SimpleMetadata = SpacedLengthMetadata,
    M1: SimpleMetadata = LobattoMetadata,
    E: AxisDirections = AxisDirections,
] = TupleBasis[
    tuple[
        AsUpcast[TransformedBasis[FundamentalBasis[M0]], M0],
        AsUpcast[TransformedBasis[FundamentalBasis[M0]], M0],
        FundamentalBasis[M1],
    ],
    E,
]


def close_coupling_basis[
    M0: SimpleMetadata,
    M1: SimpleMetadata,
    E: AxisDirections,
](
    metadata: ScatteringBasisMetadata[M0, M1, E],
) -> CloseCouplingBasis[M0, M1, E]:
    """
    Get the closed coupling basis from the scattering basis metadata.

    This is used to get the closed coupling basis from the metadata.
    """
    return TupleBasis(
        (
            basis.transformed_from_metadata(metadata.children[0]).upcast(),
            basis.transformed_from_metadata(metadata.children[1]).upcast(),
            basis.from_metadata(metadata.children[2]),
        ),
        metadata.extra,
    )


type ScatteringPotentialBasis[
    M0: SimpleMetadata = SimpleMetadata,
    M1: SimpleMetadata = SimpleMetadata,
    E: AxisDirections = AxisDirections,
    CT: Ctype[Never] = Ctype[Never],
] = DiagonalOperatorBasis[
    basis.AsUpcast[
        TupleBasis[
            tuple[FundamentalBasis[M0], FundamentalBasis[M0], FundamentalBasis[M1]],
            E,
        ],
        TupleMetadata[tuple[M0, M0, M1], E],
    ],
    Basis[TupleMetadata[tuple[M0, M0, M1], E]],
    CT,
    OperatorMetadata[TupleMetadata[tuple[M0, M0, M1], E]],
]
type ScatteringPotential[
    B: ScatteringPotentialBasis,
    DT: np.dtype[np.generic] = np.dtype[np.complexfloating],
] = Operator[B, DT]
type ScatteringPotentialWithMetadata[
    M0: SimpleMetadata = SpacedLengthMetadata,
    M1: SimpleMetadata = LobattoMetadata,
    E: AxisDirections = AxisDirections,
] = ScatteringPotential[ScatteringPotentialBasis[M0, M1, E]]


@dataclass
class LobattoPoints:
    """Represents a lobatto shape function."""

    points: np.ndarray[tuple[int], np.dtype[np.float64]]
    weights: np.ndarray[tuple[int], np.dtype[np.float64]]

    @cached_property
    def polynomials(self: Self) -> list[np.polynomial.Polynomial]:
        """Get the normalized polynomials."""
        domain = np.array([self.points[0], self.points[-1]])
        polynomials = [
            cast(
                "np.polynomial.Polynomial",
                np.polynomial.Polynomial.fromroots(np.delete(self.points, i), domain),  # type: ignore bad library type
            )
            for i in range(self.points.size)
        ]
        return [
            polynomial / polynomial(point)
            for (polynomial, point) in zip(polynomials, self.points, strict=True)
        ]

    @property
    def weighted_polynomials(self: Self) -> list[np.polynomial.Polynomial]:
        """Get the weighted polynomials."""
        return [p * w for (p, w) in zip(self.polynomials, self.weights, strict=True)]

    @cached_property
    def derivative_polynomials(self: Self) -> list[np.polynomial.Polynomial]:
        """Get the polynomial derivatives."""
        return [p.deriv() for p in self.polynomials]  # type: ignore unknown


def _get_fundamental_lobatto(
    n: int,
) -> LobattoPoints:
    """
    Compute the weights and points for the gauss-lobatto quadrature in [-1,1].

    The n-2 free points are the roots of the n-1 Legendre polynomial P(x_i)
    The n-2 free weights are 2 / (n(n-1) * P(x_i)^2)

    The remaining two points are at the end of the interval (-1, 1)
    with weight 2 / (n(n-1))

    For more details see:
    https://mathworld.wolfram.com/LobattoQuadrature.html.

    """
    inner_polynomial = scipy.special.legendre(n - 1)  # type: ignore library types
    # The inner n-2 points are given by the roots of the
    # n-1 th legendre polynomial
    inner_points = np.sort(inner_polynomial.deriv().roots)
    # Weight relative to ends is P(n-1)(point)^-2
    inner_weights = inner_polynomial(inner_points) ** (-2)  # type: ignore library types

    points = np.concat([[-1], inner_points, [1]]).astype(np.float64)
    weights = (2 / (n * (n - 1))) * np.concat([[1], inner_weights, [1]])  # type: ignore library types

    return LobattoPoints(points=points, weights=weights)  # type: ignore library types


def get_lobatto_points(
    n: int,
    limits: tuple[float, float] = (-1, 1),
) -> LobattoPoints:
    points = _get_fundamental_lobatto(n)

    (a, b) = limits
    shift = 0.5 * (b + a)
    scale = 0.5 * (b - a)

    return LobattoPoints(
        points=scale * points.points + shift,  # type: ignore library types
        weights=scale * points.weights,  # type: ignore library types
    )


def get_lobatto_derivative_matrix(
    metadata: LobattoMetadata,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    return np.array(  # type: ignore shape-mismatch
        [
            p(metadata.points)
            for p in LobattoPoints(
                metadata.points,
                metadata.weights,
            ).derivative_polynomials
        ],
    )
