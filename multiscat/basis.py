from __future__ import annotations

from typing import Any, Never

import numpy as np
from slate_core import (
    Basis,
    Ctype,
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
    SpacedLengthMetadata,
)
from slate_quantum import Operator
from slate_quantum.operator import DiagonalOperatorBasis, OperatorMetadata

from multiscat.lobatto import LobattoMetadata

type ScatteringBasisMetadata[
    M0: SimpleMetadata = SpacedLengthMetadata,
    M1: SimpleMetadata = LobattoMetadata,
    E: AxisDirections = AxisDirections,
] = TupleMetadata[
    tuple[M0, M0, M1],
    E,
]


def _get_vectors_perpendicular_to(
    vector: np.ndarray[Any, np.dtype[np.floating]],
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.floating]],
    np.ndarray[tuple[int], np.dtype[np.floating]],
]:
    assert vector.size == 3, "Vector must be a 3D vector."  # noqa: PLR2004, S101
    guess = (
        np.array([1, 0, 0]) if abs(vector[0]) > abs(vector[1]) else np.array([0, 1, 0])
    )
    v0 = guess - np.dot(guess, vector) * vector
    v0 /= np.linalg.norm(v0)
    return (v0, np.cross(vector, v0))  # type: ignore bad library type


def _project_x01_axis_directions(
    metadata: AxisDirections,
) -> AxisDirections:
    """
    Project the axis directions from the scattering basis metadata.

    This is used to extract the axis directions from the metadata.
    """
    vx, vy, vz = metadata.vectors
    v0, v1 = _get_vectors_perpendicular_to(vz)

    a_plane = np.array([np.dot(vx, v0), np.dot(vx, v1)])
    b_plane = np.array([np.dot(vy, v0), np.dot(vy, v1)])
    return AxisDirections(vectors=(a_plane, b_plane))


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
    directions_x01 = _project_x01_axis_directions(metadata.extra)
    return (
        TupleMetadata(metadata.children[:2], directions_x01),
        metadata.children[2],
    )


type CloseCouplingBasis[
    M0: SimpleMetadata = SpacedLengthMetadata,
    M1: SimpleMetadata = LobattoMetadata,
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
    Get the closed coupling basis from the scattering basis metadata.

    This is used to get the closed coupling basis from the metadata.
    """
    return TupleBasis(
        (
            basis.transformed_from_metadata(metadata.children[0]).upcast(),
            basis.transformed_from_metadata(metadata.children[1]).upcast(),
            basis.from_metadata(metadata.children[2]),
        ),
        metadata.extra,
    )


type ScatteringPotentialBasis[
    M0: SimpleMetadata = SimpleMetadata,
    M1: SimpleMetadata = SimpleMetadata,
    E: AxisDirections = AxisDirections,
    CT: Ctype[Never] = Ctype[Never],
] = DiagonalOperatorBasis[
    basis.AsUpcast[
        TupleBasis[
            tuple[FundamentalBasis[M0], FundamentalBasis[M0], FundamentalBasis[M1]],
            E,
        ],
        ScatteringBasisMetadata[M0, M1, E],
    ],
    Basis[ScatteringBasisMetadata[M0, M1, E]],
    CT,
    OperatorMetadata[ScatteringBasisMetadata[M0, M1, E]],
]

type ScatteringPotential[
    B: ScatteringPotentialBasis,
    DT: np.dtype[np.generic] = np.dtype[np.complexfloating],
] = Operator[B, DT]

type ScatteringPotentialWithMetadata[
    M0: SimpleMetadata = SpacedLengthMetadata,
    M1: SimpleMetadata = LobattoMetadata,
    E: AxisDirections = AxisDirections,
] = ScatteringPotential[ScatteringPotentialBasis[M0, M1, E]]
