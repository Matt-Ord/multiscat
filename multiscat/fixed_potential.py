from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from multiscat.config import get_lobatto_points_for_config

if TYPE_CHECKING:
    from multiscat.basis import XYBasis
    from multiscat.config import Config
    from multiscat.lobatto import LobattoPoints


@dataclass
class FixedPotential:
    """Represents a fixed potential, as loaded from the fourier file."""

    xy_basis: XYBasis
    lobatto_basis: LobattoPoints
    data: np.ndarray[tuple[Any, Any], np.dtype[np.complex128]]


def load_fixed_potential_labels(
    config: Config,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    ivx = np.zeros(config.nfc, dtype=np.int_)
    ivy = np.zeros(config.nfc, dtype=np.int_)

    with Path(config.fourier_labels_file).open() as file:
        for i in range(config.nfc):
            line = file.readline()
            ivx[i], ivy[i] = map(int, line.split())

    return ivx, ivy


def load_fixed_potential(
    config: Config,
    fourier_file: Path,
) -> FixedPotential:
    xy_basis = config.xy_basis
    potential = np.zeros(
        (config.nzfixed, np.prod(xy_basis.shape)),
        dtype=np.complex128,
    )

    ivx, ivy = load_fixed_potential_labels(config)

    # Open the data file and read in the fourier components
    with fourier_file.open("r") as file:
        # Discard the first 5 lines
        for _ in range(5):
            next(file)

        # Loop over fourier components
        for kx, ky in zip(ivx, ivy):
            # Loop over z values in fourier components
            for j in range(config.nzfixed):
                line = file.readline().strip()
                real, imag = map(float, line.strip("()").split(","))
                i = np.argwhere(
                    np.logical_and(
                        kx == xy_basis.k_points_stacked[0],
                        ky == xy_basis.k_points_stacked[1],
                    ),
                )
                potential[j, i] = complex(real, imag)

    # Scale to the program units
    potential *= config.rmlmda
    lobatto_points = get_lobatto_points_for_config(config)

    # Interpolate fourier componets at each nfc
    interpolated = np.apply_along_axis(
        lambda d: np.interp(
            lobatto_points.points,
            np.linspace(
                config.stepzmin,
                config.stepzmax,
                config.nzfixed,
                endpoint=True,
            ),
            d,
        ),
        0,
        potential,
    )
    return FixedPotential(
        lobatto_basis=lobatto_points,
        data=interpolated.astype(np.complex128),
        xy_basis=config.xy_basis,
    )
