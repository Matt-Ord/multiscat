from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse.linalg  # type: ignore[import-untyped]
from slate_core.metadata import (
    AxisDirections,
    EvenlySpacedLengthMetadata,
    LobattoSpacedLengthMetadata,
)

from multiscat.basis import (
    close_coupling_basis,
)
from multiscat.multiscat._scipy import (
    ScipyOperatorData,
    apply_diagonal,
    apply_inverse_diagonal,
    build_scipy_operator_data,
)

if TYPE_CHECKING:
    from scipy.sparse.linalg import LinearOperator  # type: ignore[untyped]

    from multiscat.config import OptimizationConfig, ScatteringCondition


def _apply_scattering_v(
    state_vector: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    potential_pairs: np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    """Apply full inter-channel scattering potential V = V^lower + V^upper."""
    return np.einsum("aik,ik->ak", potential_pairs, state_vector)


def _apply_inverse_von_neumann(
    state_vector: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    operator_data: ScipyOperatorData,
    *,
    order: int,
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    """Approximate (D + V^lower)^(-1) by a truncated von Neumann series."""
    term = apply_inverse_diagonal(state_vector, operator_data)
    out = term.copy()

    for _ in range(order):
        # term_{k+1} = -D^{-1} V term_k
        term = -apply_inverse_diagonal(
            _apply_scattering_v(term, operator_data.potential_pairs),
            operator_data,
        )
        out += term

    return out


def _build_scipy_von_neumann_operators[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    condition: ScatteringCondition[M0, M1, E],
    *,
    order: int,
    n_channels: int | None = None,
) -> tuple[LinearOperator, LinearOperator]:

    operator_data = build_scipy_operator_data(
        condition=condition,
        n_channels=n_channels,
    )

    nkx, nky, nz = operator_data.shape
    state_size = int(np.prod((nkx, nky, nz)))
    channel_idx = operator_data.channel_idx

    def _matvec_a(
        flat_state: np.ndarray[tuple[int], np.dtype[np.complex128]],
    ) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
        reshaped = flat_state.reshape((-1, nz))
        out = reshaped.copy()

        diagonal_out = apply_diagonal(
            reshaped[channel_idx].copy(),
            operator_data,
        )
        scatter_out = _apply_scattering_v(
            reshaped[channel_idx].copy(),
            operator_data.potential_pairs,
        )
        out[channel_idx] = diagonal_out + scatter_out
        return out.ravel()

    operator_a = scipy.sparse.linalg.LinearOperator(  # type: ignore[call-arg,unknown]
        (state_size, state_size),
        _matvec_a,
        dtype=np.complex128,  # type: ignore[assignment]
    )

    def _matvec_preconditioner(
        flat_state: np.ndarray[tuple[int], np.dtype[np.complex128]],
    ) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
        reshaped = flat_state.reshape((-1, nz))
        out = reshaped.copy()
        out[channel_idx] = _apply_inverse_von_neumann(
            reshaped[channel_idx],
            operator_data,
            order=order,
        )
        return out.ravel()

    preconditioner = scipy.sparse.linalg.LinearOperator(  # type: ignore[call-arg,unknown]
        (state_size, state_size),
        _matvec_preconditioner,
        dtype=np.complex128,  # type: ignore[assignment]
    )

    return operator_a, preconditioner


def run_multiscat_scipy_von_neumann[
    M0: EvenlySpacedLengthMetadata,
    M1: LobattoSpacedLengthMetadata,
    E: AxisDirections,
](
    condition: ScatteringCondition[M0, M1, E],
    config: OptimizationConfig,
    *,
    order: int = 1,
) -> np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]:
    """
    Solves the Multiscat problem using a von-neumann approximation.

    We approximate the inverse of the lower-block operator (D + V_scatter)
    by a truncated von Neumann series,
    (D + V_scatter)^(-1) ~ sum_{k=0}^order (-D^(-1) V_scatter)^k D^(-1).

    """
    if order < 0:
        msg = f"Invalid von Neumann order {order}. Expected a non-negative integer."
        raise ValueError(msg)

    operator_a, preconditioner = _build_scipy_von_neumann_operators(
        condition,
        order=order,
        n_channels=config.n_channels,
    )
    initial_state = condition.initial_state.with_basis(
        close_coupling_basis(condition.metadata),
    ).raw_data.ravel()

    restart = min(config.max_iterations, initial_state.size)
    solution, gmres_info = scipy.sparse.linalg.gmres(  # type: ignore[unknown]
        A=operator_a,
        b=initial_state,
        rtol=config.precision,
        restart=restart,
        maxiter=config.max_iterations,
        M=preconditioner,
    )

    if gmres_info != 0:
        msg = (
            "SciPy GMRES (von Neumann preconditioner) did not converge "
            f"(info={gmres_info}, max_iterations={config.max_iterations}, "
            f"restart={restart}, order={order})."
        )
        raise RuntimeError(msg)

    return np.asarray(solution, dtype=np.complex128).reshape(condition.metadata.shape)  # type: ignore[cant infer shape]
