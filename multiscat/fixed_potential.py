from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Never

import numpy as np
from slate_core import TupleMetadata
from slate_quantum import Operator

from multiscat.config import get_lobatto_points_for_config
from multiscat.lobatto import ScatteringPotentialWithMetadata, close_coupling_basis

if TYPE_CHECKING:
    from slate_core.metadata import AxisDirections, SpacedLengthMetadata

    from multiscat.basis import LobattoBasis, XYBasis
    from multiscat.config import Config
    from multiscat.lobatto import LobattoMetadata


@dataclass
class FixedPotential:
    """Represents a fixed potential, as loaded from the fourier file."""

    xy_basis: XYBasis
    z_basis: LobattoBasis
    """data indexed as [n_x, n_k_xy]"""
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
        (config.n_z_fixed, np.prod(xy_basis.shape)),
        dtype=np.complex128,
    )

    ivx, ivy = load_fixed_potential_labels(config)

    # Open the data file and read in the fourier components
    with fourier_file.open("r") as file:
        # Discard the first 5 lines
        for _ in range(5):
            next(file)

        # Loop over fourier components
        for kx, ky in zip(ivx, ivy, strict=False):
            # Loop over z values in fourier components
            for j in range(config.n_z_fixed):
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
    lobatto_basis = get_lobatto_points_for_config(config)

    # Interpolate fourier componets onto the lobatto basis
    interpolated = np.apply_along_axis(
        lambda d: np.interp(
            lobatto_basis.points,
            np.linspace(
                config.step_z_min,
                config.step_z_max,
                config.n_z_fixed,
                endpoint=True,
            )
            - config.z_min,
            d,
        ),
        0,
        potential,
    )
    return FixedPotential(
        z_basis=lobatto_basis,
        data=interpolated.astype(np.complex128),  # type: ignore shape-mismatch
        xy_basis=config.xy_basis,
    )


def interpolate_potential[
    M0: SpacedLengthMetadata,
    M1: LobattoMetadata,
    E: AxisDirections,
](
    metadata: M1,
    potential: ScatteringPotentialWithMetadata[M0, M1, E],
) -> ScatteringPotentialWithMetadata[M0, M1, E]:
    # This function is a placeholder for potential interpolation logic.
    # The actual implementation would depend on the specific requirements
    # and context of the application.
    old_state_metadata = potential.basis.metadata().children[0]
    state_metadata = TupleMetadata(
        (old_state_metadata.children[0], old_state_metadata.children[1], metadata),
        old_state_metadata.extra,
    )
    _state_basis = close_coupling_basis(state_metadata)

    msg = "Potential interpolation is not implemented yet. "
    raise NotImplementedError(msg)
    todo_basis = Never
    z_points = potential.basis.metadata().children[0].children[2].values
    interpolated = np.apply_along_axis(
        lambda d: np.interp(metadata.points, z_points, d),
        0,
        potential.with_basis(todo_basis).raw_data,
    )

    return Operator(todo_basis, data=interpolated)
