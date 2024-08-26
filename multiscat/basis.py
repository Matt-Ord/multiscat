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
    def n(
        self: Self,
    ) -> int:
        return np.prod(self.shape).item()

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
        return np.einsum("ij,il->lj", self.nk_points_stacked, self.dk_stacked)  # type: ignore unknown


@dataclass
class LobattoBasis:
    """Represents an n-point lobatto basis."""

    n_points: int
    delta_x: float

    @cached_property
    def lobatto_points(self: Self) -> LobattoPoints:
        """Get the lobatto points."""
        return get_lobatto_points(self.n_points, (0, self.delta_x))

    @property
    def points(self: Self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Get the points."""
        return self.lobatto_points.points

    @property
    def weights(self: Self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Get the weights."""
        return self.lobatto_points.weights
