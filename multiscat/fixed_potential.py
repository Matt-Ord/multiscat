from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from multiscat.config import Config


@dataclass
class FixedPotential:
    """Represents a fixed potential, as loaded from the fourier file."""

    z_points: np.ndarray[tuple[Any], np.dtype[np.float64]]
    data: np.ndarray[tuple[Any, Any], np.dtype[np.complex128]]


def load_fixed_potential(
    config: Config,
    fourier_file: Path,
) -> FixedPotential:
    # Initialize vfcfixed to zeros
    vfcfixed = np.zeros((config.nzfixed, config.nfc), dtype=np.complex128)

    # Open the data file and read in the fourier components
    with fourier_file.open("r") as file:
        # Discard the first 5 lines
        for _ in range(5):
            next(file)

        # Loop over fourier components
        for i in range(config.nfc):
            # Loop over z values in fourier components
            for j in range(config.nzfixed):
                line = file.readline().strip()
                real, imag = map(float, line.strip("()").split(","))
                vfcfixed[j, i] = complex(real, imag)

    # Scale to the program units
    vfcfixed *= config.rmlmda

    return FixedPotential(
        data=vfcfixed,
        z_points=np.linspace(
            config.stepzmin,
            config.stepzmax,
            config.nzfixed,
            endpoint=True,
        ),
    )


def interpolate_potential_z(
    vfcfixed: FixedPotential,
    nfc: int,
    z_points: np.ndarray[Any, Any],
) -> FixedPotential:
    """Interpolate the FixedPotential onto the given z_points."""
    # Interpolate fourier componets at each nfc
    data = np.apply_along_axis(
        lambda d: np.interp(z_points, vfcfixed.z_points, d),
        0,
        vfcfixed.data,
    )
    return FixedPotential(z_points=z_points, data=data[:, :nfc].astype(np.complex128))
