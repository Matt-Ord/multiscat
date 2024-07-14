from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.special


@dataclass
class LobattoPoints:
    points: np.ndarray[tuple[int], np.dtype[np.float64]]
    weights: np.ndarray[tuple[int], np.dtype[np.float64]]


## TODO write test comparing to this...
def _lobatto_from_fortran(a: float, b: float, n: int):
    """Calculate lobatto weights based on the fortran approach.

    Coppied directly from the code provided in
    https://doi.org/10.1007/978-94-015-8240-7_4

    Parameters
    ----------
    a : float
        _description_
    b : float
        _description_
    n : int
        _description_

    Returns
    -------
    _type_
        _description_

    """
    n_unique = (n + 1) // 2
    pi = np.arccos(-1.0)
    shift = 0.5 * (b + a)
    scale = 0.5 * (b - a)
    weight = (b - a) / (n * (n - 1))
    w = np.zeros(n)
    x = np.zeros(n)

    x[0] = a
    w[0] = weight

    for k in range(2, n_unique + 1):
        z = np.cos(pi * (4 * k - 3) / (4 * n - 2))
        p2 = 0.0
        p1 = 1.0
        for _ in range(7):
            p2 = 0.0
            p1 = 1.0

            for j in range(1, n):
                p3 = p2
                p2 = p1
                p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j

            p2 = (n - 1) * (p2 - z * p1) / (1.0 - z * z)
            p3 = (2 * z * p2 - n * (n - 1) * p1) / (1.0 - z * z)
            z = z - p2 / p3

        x[k - 1] = shift - scale * z
        x[n - k] = shift + scale * z
        w[k - 1] = weight / (p1 * p1)
        w[n - k] = w[k - 1]

    x[n - 1] = b
    w[n - 1] = weight

    return x, w


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
