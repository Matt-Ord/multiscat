from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from slate_core import SimpleMetadata, TupleBasis, TupleMetadata, basis
from slate_core.basis import AsUpcast, CroppedBasis
from slate_quantum import Operator
from slate_quantum.operator import position_operator_basis

from multiscat.basis import (
    ScatteringBasisMetadata,
    close_coupling_basis,
)

if TYPE_CHECKING:
    from slate_core.metadata import (
        AxisDirections,
        EvenlySpacedLengthMetadata,
        EvenlySpacedVolumeMetadata,
        LobattoSpacedMetadata,
    )
    from slate_quantum.operator import OperatorBasis

type ScatteringOperator[
    M0: SimpleMetadata,
    M1: SimpleMetadata,
    E: AxisDirections,
] = Operator[
    OperatorBasis[ScatteringBasisMetadata[M0, M1, E]],
    np.dtype[np.complexfloating],
]


# TODO: Make this more general, interpolating between  # noqa: FIX002
# LabelledMetadata and move it into slate_quantum.operator.build,
# since this is generally useful.
def interpolate_potential[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedMetadata,
    E: AxisDirections,
](
    metadata: ScatteringBasisMetadata[M0, M1, E],
    potential: Operator[
        OperatorBasis[EvenlySpacedVolumeMetadata],
        np.dtype[np.complexfloating],
    ],
) -> ScatteringOperator[M0, M1, E]:
    old_state_metadata = cast(
        "ScatteringBasisMetadata[EvenlySpacedLengthMetadata]",
        potential.basis.metadata().children[0],
    )
    if metadata.extra != old_state_metadata.extra:
        msg = (
            "The AxisDirections of the potential must match the"
            "interpolated AxisDirections."
        )
        raise ValueError(msg)
    if (
        metadata.children[0].delta != old_state_metadata.children[0].delta
        or metadata.children[1].delta != old_state_metadata.children[1].delta
    ):
        msg = (
            "The repeating unit cell of the potential"
            "must match the repeat cell of the interpolation."
        )
        raise ValueError(msg)

    # Convert the potential to V_{k0, k1, x3} basis
    # This potential is indexed by momentum in the direction parallel to the
    # surface, but is given as a set of evenly spaced points in the
    # perpendicular direction.
    converted = potential.with_basis(
        position_operator_basis(close_coupling_basis(old_state_metadata)),
    )
    # We interpolate the potential to the new lobatto points in the z direction
    # using np.interp.
    interpolated = np.apply_along_axis(
        lambda d: np.interp(
            metadata.children[2].values,
            old_state_metadata.children[2].values,
            d,
        )
        # TODO: We need to add weights to a general LabelledMetadata  # noqa: FIX002
        # if they are not EvenlySpaced
        / np.square(metadata.children[2].basis_weights),
        2,
        converted.raw_data.reshape(old_state_metadata.shape),
    )

    # The basis of the interpolated potential
    # Parralel to the surface, we still have a transformed basis, but we may
    # be interpolating onto a finer grid. We therefore have a cropped basis
    # in the x and y directions.
    # The z direction is still a fundamental basis.
    outer_basis = TupleBasis(
        (
            CroppedBasis(
                old_state_metadata.children[0].fundamental_size,
                basis.transformed_from_metadata(metadata.children[0]),
            ),
            CroppedBasis(
                old_state_metadata.children[1].fundamental_size,
                basis.transformed_from_metadata(metadata.children[1]),
            ),
            basis.from_metadata(metadata.children[2]),
        ),
        metadata.extra,
    )
    out_basis = position_operator_basis(outer_basis)
    return Operator(
        AsUpcast(out_basis, TupleMetadata((metadata, metadata))),
        interpolated.astype(np.complex128),
    )
