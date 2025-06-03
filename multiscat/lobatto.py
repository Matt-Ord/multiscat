from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import cast, override

import numpy as np
import scipy.special  # type: ignore unknown
from slate_core.metadata import (
    LengthMetadata,
)


@dataclass(frozen=True)
class _LobattoData:
    """Points and weights for Lobatto quadrature."""

    points: np.ndarray[tuple[int], np.dtype[np.float64]]
    weights: np.ndarray[tuple[int], np.dtype[np.float64]]


def _get_fundamental_lobatto_data(
    n: int,
) -> _LobattoData:
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

    return _LobattoData(points=points, weights=weights)  # type: ignore library types


def _get_scaled_lobatto_data(
    n: int,
    limits: tuple[float, float] = (-1, 1),
) -> _LobattoData:
    points = _get_fundamental_lobatto_data(n)

    (a, b) = limits
    shift = 0.5 * (b + a)
    scale = 0.5 * (b - a)

    return _LobattoData(
        points=scale * points.points + shift,  # type: ignore library types
        weights=scale * points.weights,  # type: ignore library types
    )


class LobattoMetadata(LengthMetadata):
    """Metadata for a Lobatto basis."""

    def __init__(self, n: int, delta: float) -> None:
        self._delta = delta
        super().__init__(n)

    @cached_property
    def _lobatto_data(self) -> _LobattoData:
        """Get the fundamental lobatto points and weights."""
        return _get_scaled_lobatto_data(self.fundamental_size, (0, self.delta))

    @property
    def weights(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Get the weights of the lobatto points."""
        return self._lobatto_data.weights

    @property
    @override
    def values(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Get the points of the lobatto points."""
        return self._lobatto_data.points

    @property
    @override
    def delta(self) -> float:
        return self._delta


def get_polynomials(
    metadata: LobattoMetadata,
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
        for (polynomial, point) in zip(polynomials, metadata.values, strict=True)
    ]


def get_weighted_polynomials(
    metadata: LobattoMetadata,
) -> list[np.polynomial.Polynomial]:
    """Get the weighted lobatto polynomials for the lobatto points."""
    return [
        p * w
        for (p, w) in zip(get_polynomials(metadata), metadata.weights, strict=True)
    ]


def get_derivative_polynomials(
    metadata: LobattoMetadata,
) -> list[np.polynomial.Polynomial]:
    """Get the derivative polynomials for the lobatto points."""
    return [p.deriv() for p in get_polynomials(metadata)]


def get_lobatto_derivative_matrix(
    metadata: LobattoMetadata,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    return np.array(  # type: ignore shape-mismatch
        [p(metadata.values) for p in get_derivative_polynomials(metadata)],
    )
