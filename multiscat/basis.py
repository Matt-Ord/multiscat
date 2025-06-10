from __future__ import annotations

from typing import Any

import numpy as np
from slate_core import (
    FundamentalBasis,
    SimpleMetadata,
    TransformedBasis,
    TupleBasis,
    TupleMetadata,
    basis,
)
from slate_core.basis import AsUpcast
from slate_core.metadata import (
    AxisDirections,
    Domain,
    EvenlySpacedLengthMetadata,
    LobattoSpacedLengthMetadata,
)
from slate_core.metadata.volume import project_directions_onto_axes

type ScatteringBasisMetadata[
    M0: SimpleMetadata = EvenlySpacedLengthMetadata,
    M1: SimpleMetadata = LobattoSpacedLengthMetadata,
    E: AxisDirections = AxisDirections,
] = TupleMetadata[
    tuple[M0, M0, M1],
    E,
]


def scattering_metadata_from_stacked_delta_x(
    vectors: tuple[np.ndarray[Any, np.dtype[np.floating]], ...],
    shape: tuple[int, int, int],
) -> ScatteringBasisMetadata:
    """Get the metadata for a scattering basis from the vectors and spacing."""
    delta_v = tuple(np.linalg.norm(v).item() for v in vectors)
    normalized_vectors = tuple(v / dv for v, dv in zip(vectors, delta_v, strict=True))
    return TupleMetadata(
        (
            EvenlySpacedLengthMetadata(
                shape[0],
                domain=Domain(delta=delta_v[0]),
                interpolation="Fourier",
            ),
            EvenlySpacedLengthMetadata(
                shape[1],
                domain=Domain(delta=delta_v[1]),
                interpolation="Fourier",
            ),
            LobattoSpacedLengthMetadata(
                shape[2],
                domain=Domain(delta=delta_v[2]),
            ),
        ),
        AxisDirections(vectors=normalized_vectors),
    )


def split_scattering_metadata[
    M0: SimpleMetadata,
    M1: SimpleMetadata,
](
    metadata: ScatteringBasisMetadata[M0, M1, AxisDirections],
) -> tuple[
    TupleMetadata[tuple[M0, M0], AxisDirections],
    M1,
]:
    """Split the scattering basis metadata into parallel and perpendicular parts."""
    return (
        TupleMetadata(
            metadata.children[:2],
            project_directions_onto_axes(metadata.extra, (0, 1)),
        ),
        metadata.children[2],
    )


type CloseCouplingBasis[
    M0: SimpleMetadata = EvenlySpacedLengthMetadata,
    M1: SimpleMetadata = LobattoSpacedLengthMetadata,
    E: AxisDirections = AxisDirections,
] = TupleBasis[
    tuple[
        AsUpcast[TransformedBasis[FundamentalBasis[M0]], M0],
        AsUpcast[TransformedBasis[FundamentalBasis[M0]], M0],
        FundamentalBasis[M1],
    ],
    E,
]


def close_coupling_basis[
    M0: SimpleMetadata,
    M1: SimpleMetadata,
    E: AxisDirections,
](
    metadata: ScatteringBasisMetadata[M0, M1, E],
) -> CloseCouplingBasis[M0, M1, E]:
    """
    Get the close coupling basis from basis metadata.

    This is the basis we use in the close coupling method.
    """
    return TupleBasis(
        (
            basis.transformed_from_metadata(metadata.children[0]).upcast(),
            basis.transformed_from_metadata(metadata.children[1]).upcast(),
            basis.from_metadata(metadata.children[2]),
        ),
        metadata.extra,
    )
