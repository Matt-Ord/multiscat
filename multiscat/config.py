from __future__ import annotations

from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path
from typing import Any, Self

import numpy as np

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
    vmin: float
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
    hemass: float

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
        return 2 * self.hemass / 4.18020

    @cache  # noqa: B019
    def label_fourier_components(
        self: Config,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], int]:
        """Label the Fourier components based on the provided file.

        The components are listed in 'fourier_labels_file'
        and appear in the same order as in the potential file.

        Parameters
        ----------
        fourier_labels_file (str): Path to the file containing Fourier labels.
        nfc (int): Number of Fourier components.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, int]:
            Tuple containing arrays of ivx and ivy components and the index of the zero
            Fourier component (nfc00).

        """
        ivx = np.zeros(self.nfc, dtype=np.int_)
        ivy = np.zeros(self.nfc, dtype=np.int_)
        nfc00 = -1

        with Path(self.fourier_labels_file).open() as file:
            for i in range(self.nfc):
                line = file.readline()
                ivx[i], ivy[i] = map(int, line.split())
                if ivx[i] == 0 and ivy[i] == 0:
                    nfc00 = i

        return ivx, ivy, nfc00

    @cached_property
    def scattering_conditions(self: Self) -> list[ScatteringCondition]:
        """Get the scattering conditions."""
        return load_scattering_conditions(Path(self.scatt_cond_file))
