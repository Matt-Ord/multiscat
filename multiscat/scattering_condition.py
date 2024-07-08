from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class ScatteringCondition:
    """Represents a condition for a scattering calculation.

    Theta and phi are given in degrees
    """

    energy: float
    theta: float
    phi: float
    mass: float

    @property
    def theta_radians(self: Self) -> float:
        return self.theta * np.pi / 180.0

    @property
    def phi_radians(self: Self) -> float:
        return self.phi * np.pi / 180.0

    @property
    def _abs_momentum(self: Self) -> float:
        # TODO: what units is this in lol...
        hbar_squared = 4.18020
        return np.sqrt(2 * self.mass * self.energy / hbar_squared)

    @property
    def momentum(self: Self) -> np.ndarray[tuple[Literal[3]], np.dtype[np.float64]]:
        """The momentum as a 3 vector (kx, ky, kz)."""
        k = self._abs_momentum
        return k * np.array(
            [
                np.sin(self.theta_radians) * np.cos(self.phi_radians),
                np.sin(self.theta_radians) * np.sin(self.phi_radians),
                np.cos(self.theta_radians),
            ],
        )


def parse_scattering_condition(line: str, mass: float) -> ScatteringCondition:
    energy, theta, phi = map(float, line.split())
    return ScatteringCondition(energy, theta, phi, mass)


def load_scattering_conditions(file: Path, mass: float) -> list[ScatteringCondition]:
    return [
        parse_scattering_condition(cond, mass) for cond in file.read_text().splitlines()
    ]
