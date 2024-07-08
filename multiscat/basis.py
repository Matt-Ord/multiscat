from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np


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
