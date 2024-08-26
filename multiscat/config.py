from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Self

import numpy as np

from multiscat.basis import LobattoBasis, XYBasis
from multiscat.scattering_condition import (
    ScatteringCondition,
    load_scattering_conditions,
)

# Hard limits on Multiscat parameters included everywhere at compile time
N_Z_FIXED_MAX = 550  # max no of z points in fixed pot.
N_VFC_FIXED_MAX = 4096  # max no of fourier components (fixed, from file)
N_MAX = 1024  # diffraction channels
M_MAX = 550  # z grid points
# max number of potential fourier components per z slice
N_FOURIER_COMPONENTS_PER_Z = 4096


@dataclass
class GMRESConfig:
    """Represents options for the GMRES solver."""

    precision: float
    preconditioner: int


@dataclass
class Config:
    """Configuration for the scattering calculation."""

    fourier_labels_file: Path
    scattering_conditions_file: Path
    i_test: int
    ipc: int
    eps: float
    nsf: int
    nfc: int
    z_min: float
    z_max: float
    d_max: float
    i_max: int
    a1: float
    a2: float
    b2: float
    n_z_fixed: int
    step_z_min: float
    step_z_max: float
    startindex: int
    endindex: int
    atom_mass: float

    def __post_init__(self: Self) -> None:  # noqa: D105
        if self.nfc > N_FOURIER_COMPONENTS_PER_Z:
            msg = (
                "ERROR: the .conf file needs more fourier components than"
                "allowed by the .inc file (nfc > nfc x)"
            )
            raise ValueError(msg)
        if self.n_z_fixed > N_Z_FIXED_MAX:
            msg = (
                "ERROR: the .conf file needs more z points than"
                "allowed by the .inc file (n_z_fixed > N_Z_FIXED_MAX)"
            )
            raise ValueError(msg)
        if self.nfc > N_VFC_FIXED_MAX:
            msg = (
                "ERROR: the .conf file needs more fourier components than"
                "allowed by the .inc file (nfc > N_VFC_FIXED_MAX)"
            )
            raise ValueError(msg)

    @property
    def mz(self: Self) -> int:
        """Get the number of z points."""
        m = 550
        if m > M_MAX:
            msg = "Mz Too Big!"
            raise ValueError(msg)
        return m

    @property
    def rmlmda(self: Self) -> float:
        """Get the simulation units."""
        return 2 * self.atom_mass / 4.18020

    @cached_property
    def scattering_conditions(self: Self) -> list[ScatteringCondition]:
        """Get the scattering conditions."""
        return load_scattering_conditions(
            self.scattering_conditions_file,
            self.atom_mass,
        )

    @property
    def xy_basis(self: Self) -> XYBasis:
        """Get the scattering conditions."""
        n_points = 1 + 2 * self.i_max
        # TODO: previously this NMax was only for d < d max
        if n_points**2 < N_MAX:
            msg = "ERROR: n too big! (basis)"
            raise ValueError(msg)
        delta_x_stacked = np.array([[self.a1, self.a2], [0, self.b2]])
        return XYBasis(delta_x_stacked, (n_points, n_points))

    @property
    def gmres_config(self: Self) -> GMRESConfig:
        """Get the gmres config."""
        return GMRESConfig(precision=self.eps, preconditioner=self.ipc)

    def __str__(self: Self) -> str:
        """Return the Config as a string."""
        out = ""
        out += f"Fourier labels file = {self.fourier_labels_file}\n"
        out += f"Loading scattering conditions from {self.scattering_conditions_file}\n"
        out += f"Output mode = {self.i_test}\n"
        out += f"GMRES preconditioner flag = {self.ipc}\n"
        out += f"Convergence sig. figures = {self.nsf}\n"
        out += f"Total number of Fourier components to use = {self.nfc}\n"
        out += f"z integration range = ({self.z_min}, {self.z_max})\n"
        out += f"Max energy of closed channels = {self.d_max}\n"
        out += f"Max index of channels = {self.i_max}\n"
        out += f"Unit cell (A) = {self.a1} x {self.b2}\n"
        out += f"Number of z points in Fourier components = {self.n_z_fixed}\n"
        out += f"Calculating for potential input files between {self.startindex}.in"
        out += f"and {self.endindex}.in\n"
        out += f"Atom Mass = {self.atom_mass}"
        return out


def get_lobatto_points_for_config(
    config: Config,
) -> LobattoBasis:
    n_points = config.mz + 1

    return LobattoBasis(n_points, config.z_max - config.z_min)


def _parse_value(line: str) -> str:
    return line.split("!")[0].strip()


def read_config(file_path: Path) -> Config:
    with file_path.open("r") as file:
        lines = file.readlines()

    fourier_labels_file = _parse_value(lines[0])
    scattering_conditions_file = _parse_value(lines[1])
    i_test = int(_parse_value(lines[2]))
    ipc = int(_parse_value(lines[3]))
    ipc = min(max(ipc, 0), 1)
    nsf = int(_parse_value(lines[4]))
    nsf = min(max(nsf, 2), 5)
    eps = 0.5 * (10 ** (-nsf))
    nfc = int(_parse_value(lines[5]))
    z_min, z_max = map(float, _parse_value(lines[6]).split(","))
    float(_parse_value(lines[7]))
    d_max = float(_parse_value(lines[8]))
    i_max = int(_parse_value(lines[9]))
    a1 = float(_parse_value(lines[10]))
    a2 = float(_parse_value(lines[11]))
    b2 = float(_parse_value(lines[12]))
    n_z_fixed = int(_parse_value(lines[13]))
    step_z_min = float(_parse_value(lines[14]))
    step_z_max = float(_parse_value(lines[15]))
    startindex = int(_parse_value(lines[16]))
    endindex = int(_parse_value(lines[17]))
    he_mass = float(_parse_value(lines[18]))

    return Config(
        fourier_labels_file=Path(fourier_labels_file),
        scattering_conditions_file=Path(scattering_conditions_file),
        i_test=i_test,
        ipc=ipc,
        eps=eps,
        nsf=nsf,
        nfc=nfc,
        z_min=z_min,
        z_max=z_max,
        d_max=d_max,
        i_max=i_max,
        a1=a1,
        a2=a2,
        b2=b2,
        n_z_fixed=n_z_fixed,
        step_z_min=step_z_min,
        step_z_max=step_z_max,
        startindex=startindex,
        endindex=endindex,
        atom_mass=he_mass,
    )
