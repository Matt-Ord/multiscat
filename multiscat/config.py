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
NZFIXED_MAX = 550  # max no of z points in fixed pot.
NVFCFIXED_MAX = 4096  # max no of fourier cmpts (fixed, from file)
NMAX = 1024  # diffraction channels
MMAX = 550  # z grid points
NFCX = 4096  # max number of potential fourier components per z slice


@dataclass
class GMRESConfig:
    """Represents options for the GMRES solver."""

    precision: float
    preconditioner: int


@dataclass
class Config:
    """Configuration for the scattering calculation."""

    fourier_labels_file: str
    scatt_cond_file: str
    itest: int
    ipc: int
    eps: float
    nsf: int
    nfc: int
    zmin: float
    zmax: float
    dmax: float
    imax: int
    a1: float
    a2: float
    b2: float
    nzfixed: int
    stepzmin: float
    stepzmax: float
    startindex: int
    endindex: int
    he_mass: float

    def __post_init__(self: Self) -> None:  # noqa: D105
        if self.nfc > NFCX:
            msg = (
                "ERROR: the .conf file needs more fourier components than"
                "allowed by the .inc file (nfc > nfcx)"
            )
            raise ValueError(msg)
        if self.nzfixed > NZFIXED_MAX:
            msg = (
                "ERROR: the .conf file needs more z points than"
                "allowed by the .inc file (nzfixed > NZFIXED_MAX)"
            )
            raise ValueError(msg)
        if self.nfc > NVFCFIXED_MAX:
            msg = (
                "ERROR: the .conf file needs more fourier components than"
                "allowed by the .inc file (nfc > NVFCFIXED_MAX)"
            )
            raise ValueError(msg)

    @property
    def mz(self: Self) -> int:
        """Get the number of z points."""
        m = 550
        if m > MMAX:
            msg = "Mz Too Big!"
            raise ValueError(msg)
        return m

    @property
    def rmlmda(self: Self) -> float:
        """Get the simulation units."""
        return 2 * self.he_mass / 4.18020

    @cached_property
    def scattering_conditions(self: Self) -> list[ScatteringCondition]:
        """Get the scattering conditions."""
        return load_scattering_conditions(Path(self.scatt_cond_file), self.he_mass)

    @property
    def xy_basis(self: Self) -> XYBasis:
        """Get the scattering conditions."""
        n_points = 1 + 2 * self.imax
        # TODO: previously this NMax was only for d < dmax
        if n_points**2 < NMAX:
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
        out += f"Loading scattering conditions from {self.scatt_cond_file}\n"
        out += f"Output mode = {self.itest}\n"
        out += f"GMRES preconditioner flag = {self.ipc}\n"
        out += f"Convergence sig. figures = {self.nsf}\n"
        out += f"Total number of Fourier components to use = {self.nfc}\n"
        out += f"z integration range = ({self.zmin}, {self.zmax})\n"
        out += f"Max energy of closed channels = {self.dmax}\n"
        out += f"Max index of channels = {self.imax}\n"
        out += f"Unit cell (A) = {self.a1} x {self.b2}\n"
        out += f"Number of z points in Fourier components (nzfixed) = {self.nzfixed}\n"
        out += f"Calculating for potential input files between {self.startindex}.in"
        out += f"and {self.endindex}.in\n"
        out += f"Atom Mass = {self.he_mass}"
        return out


def get_lobatto_points_for_config(
    config: Config,
) -> LobattoBasis:
    n_points = config.mz + 1

    return LobattoBasis(n_points, config.zmax - config.zmin)


def _parse_value(line: str) -> str:
    return line.split("!")[0].strip()


def read_config(file_path: Path) -> Config:
    with file_path.open("r") as file:
        lines = file.readlines()

    fourier_labels_file = _parse_value(lines[0])
    scatt_cond_file = _parse_value(lines[1])
    itest = int(_parse_value(lines[2]))
    ipc = int(_parse_value(lines[3]))
    ipc = min(max(ipc, 0), 1)
    nsf = int(_parse_value(lines[4]))
    nsf = min(max(nsf, 2), 5)
    eps = 0.5 * (10 ** (-nsf))
    nfc = int(_parse_value(lines[5]))
    zmin, zmax = map(float, _parse_value(lines[6]).split(","))
    float(_parse_value(lines[7]))
    dmax = float(_parse_value(lines[8]))
    imax = int(_parse_value(lines[9]))
    a1 = float(_parse_value(lines[10]))
    a2 = float(_parse_value(lines[11]))
    b2 = float(_parse_value(lines[12]))
    nzfixed = int(_parse_value(lines[13]))
    stepzmin = float(_parse_value(lines[14]))
    stepzmax = float(_parse_value(lines[15]))
    startindex = int(_parse_value(lines[16]))
    endindex = int(_parse_value(lines[17]))
    he_mass = float(_parse_value(lines[18]))

    return Config(
        fourier_labels_file=fourier_labels_file,
        scatt_cond_file=scatt_cond_file,
        itest=itest,
        ipc=ipc,
        eps=eps,
        nsf=nsf,
        nfc=nfc,
        zmin=zmin,
        zmax=zmax,
        dmax=dmax,
        imax=imax,
        a1=a1,
        a2=a2,
        b2=b2,
        nzfixed=nzfixed,
        stepzmin=stepzmin,
        stepzmax=stepzmax,
        startindex=startindex,
        endindex=endindex,
        he_mass=he_mass,
    )
