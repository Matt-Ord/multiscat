from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.special


@dataclass
class LobattoPoints:
    points: np.ndarray[tuple[int], np.dtype[np.float64]]
    weights: np.ndarray[tuple[int], np.dtype[np.float64]]


def _get_fundamental_lobatto(
    n: int,
) -> LobattoPoints:
    """Compute the weights and points for the gauss-lobatto quadrature in [-1,1].

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
    inner_weights = inner_polynomial(inner_points) ** (-2)

    points = np.concat([[-1], inner_points, [1]])
    weights = (2 / (n * (n - 1))) * np.concat([[1], inner_weights, [1]])

    return LobattoPoints(points=points, weights=weights)


def get_lobatto_points(
    n: int,
    limits: tuple[float, float] = (-1, 1),
) -> LobattoPoints:
    points = _get_fundamental_lobatto(n)

    (a, b) = limits
    shift = 0.5 * (b + a)
    scale = 0.5 * (b - a)

    return LobattoPoints(
        points=scale * points.points + shift,
        weights=scale * points.weights,
    )
