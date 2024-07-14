from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Self

import numpy as np

from multiscat.lobatto import get_lobatto_points

if TYPE_CHECKING:
    from multiscat.lobatto import LobattoPoints


@dataclass
class XYBasis:
    """Class to store the xy basis vectors."""

    delta_x_stacked: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    """delta_x as a list of delta_x for each axis"""

    shape: tuple[int, int]

    @property
    def dk_stacked(
        self: Self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Dk as a list of dk for each axis."""
        return 2 * np.pi * np.linalg.inv(self.delta_x_stacked).T

    @property
    def nx_points_stacked(
        self: Self,
    ) -> tuple[np.ndarray[tuple[int], np.dtype[np.int_]], ...]:
        """Get the nx points."""
        nx_mesh = np.meshgrid(
            *[np.arange(0, n, dtype=int) for n in self.shape],
            indexing="ij",
        )
        return tuple(nxi.ravel() for nxi in nx_mesh)

    @staticmethod
    def _get_nk_points(n: int) -> np.ndarray[tuple[int], np.dtype[np.int_]]:
        return np.fft.ifftshift(np.arange((-n + 1) // 2, (n + 1) // 2))

    @property
    def nk_points_stacked(
        self: Self,
    ) -> tuple[np.ndarray[tuple[int], np.dtype[np.int_]], ...]:
        """Get the nk points."""
        nx_mesh = np.meshgrid(
            *[self._get_nk_points(n) for n in self.shape],
            indexing="ij",
        )
        return tuple(nxi.ravel() for nxi in nx_mesh)

    @property
    def k_points_stacked(
        self: Self,
    ) -> tuple[np.ndarray[tuple[int], np.dtype[np.int_]], ...]:
        """Get the k points."""
        return np.einsum("ij,il->lj", self.nk_points_stacked, self.dk_stacked)


@dataclass
class LobattoBasis:
    """Represents an n-point lobatto basis."""

    n_points: int
    delta_x: float

    @cached_property
    def _lobatto_points(self: Self) -> LobattoPoints:
        return get_lobatto_points(self.n_points, (0, self.delta_x))

    @property
    def points(self: Self):
        return self._lobatto_points.points

    @property
    def weights(self: Self):
        return self._lobatto_points.weights


def get_lobatto_derivatives(
    points: LobattoBasis,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Calculate the derivative matrix u_i'(R_j) for the lobatto basis.

    Parameters
    ----------
    points : LobattoPoints

    Returns
    -------
    np.ndarray[Any, np.dtype[np.float64]]

    """
    # The derivatives can be evaluated analytically
    # u_i'(R_i) = \sum_j=0 M+1 (R_i - R_j)^-1
    # or for j=\=i
    # u_i'(R_j) = (R_i - R_j)^-1 product_0^M+1 (R_j-R_k) / (R_i - R_k)
    # Where the product excludes k=j and k=i
    n_points = points.points.size

    # Calculate the reciprocal of differences (R_i - R_j)^-1, ignoring the diagonal
    diff = points.points[:, np.newaxis] - points.points[np.newaxis, :]
    reciprocal_diff = np.where(diff != 0, 1.0 / diff, 0)

    # Calculate product_k=0^M+1 (R_j-R_k) / (R_i - R_k)
    mask = np.eye(n_points, dtype=bool)
    products = np.prod(
        np.where(
            # Ignoring the zero elements from the product
            mask[np.newaxis, :, :] | mask[:, np.newaxis, :],
            1,
            (diff[np.newaxis, :, :] * reciprocal_diff[:, np.newaxis, :]),
        ),
        axis=2,
    )

    u_derivative = reciprocal_diff * products
    # Calculate diagonal elements seperately
    # u_i'(R_i) = \sum_j=0 M+1 (R_i - R_j)^-1
    u_derivative[np.arange(n_points), np.arange(n_points)] = np.sum(
        reciprocal_diff,
        axis=1,
    )
    return u_derivative
