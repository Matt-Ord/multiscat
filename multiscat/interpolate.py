from __future__ import annotations

from typing import TYPE_CHECKING, Never

import numpy as np
from slate_core import TupleMetadata
from slate_quantum import Operator

from multiscat.basis import ScatteringPotentialWithMetadata, close_coupling_basis

if TYPE_CHECKING:
    from slate_core.metadata import AxisDirections, SpacedLengthMetadata

    from multiscat.lobatto import LobattoMetadata


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
