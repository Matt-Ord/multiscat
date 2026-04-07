from typing import TYPE_CHECKING, Any, cast

import numpy as np
import scipy.sparse  # type: ignore[import-untyped]
import scipy.sparse.linalg  # type: ignore[import-untyped]
from slate_core.util import timed
from tqdm import tqdm

if TYPE_CHECKING:
    from multiscat.config import OptimizationConfig


@timed
def run_gauss_seidel_gradient_decent(  # cspell: disable-line  # noqa: PLR0913
    target_state: np.ndarray[tuple[int], np.dtype[np.complexfloating]],
    inverse_lower: scipy.sparse.linalg.LinearOperator,
    upper: scipy.sparse.linalg.LinearOperator,
    lower: scipy.sparse.linalg.LinearOperator | None = None,
    initial_state: np.ndarray[tuple[int], np.dtype[np.complexfloating]] | None = None,
    *,
    config: OptimizationConfig,
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    """
    Use gradient decent to solve the linear system (I + L^{-1} U) psi = L^{-1} psi.

    This is equivalent to solving the original linear problem
    (L + U) psi = b, but is more efficient to solve since the operator (I + L^{-1} U)
    is closer to the identity.

    The input to this function is the target state vector b.
    The output of this function is L^{-1} psi, or simply psi if the
    lower operator is provided.

    Optionally, this probelm can be "double preconditioned"
    by applying a first-order Neumann series approximation to the operator
    (I - L^{-1} U) as a preconditioner to the GMRES solver. This should improve
    convergence if L^{-1} U is small.
    """
    n_states = target_state.size

    def _apply_linear_operator(
        state: np.ndarray[tuple[int], np.dtype[np.complex128]],
    ) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
        """
        Apply the preconditioned operator (I + L^{-1} * U) to the state vector.

        In the "simple" problem, we would solve (H_0 + V + C_i u_iv_i^T) psi = b.
        We split this operator into (L + U) psi = b
        where L is the lower operator and U is the upper operator.

        L contains the lower diagonal terms in the scattering
        potential, and the diagonal terms
        (H_0 and the boundary correction C_i u_i v_i^T).

        We then apply the preconditioner L^{-1} to both sides, giving
        (I + L^{-1} * U) psi = L^{-1} b

        We always use the specular preconditioner L^{-1},
        which is required for convergence.
        """
        return state + inverse_lower.matvec(upper.matvec(state))  # type: ignore[unknown]

    def _apply_neumann_preconditioner(
        state: np.ndarray[tuple[int], np.dtype[np.complex128]],
    ) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
        """
        Valuates a first-order Neumann series approximation: (I - L^{-1}U).

        If L^{-1}U is small, this is an approximate solution to the scattering problem.
        """
        return state - inverse_lower.matvec(upper.matvec(state))  # type: ignore[unknown]

    linear_operator = scipy.sparse.linalg.LinearOperator(  # type: ignore[call-arg,unknown]
        (n_states, n_states),
        _apply_linear_operator,
        dtype=np.complex128,  # type: ignore[type-arg],
    )
    preconditioner = (
        scipy.sparse.linalg.LinearOperator(  # type: ignore[call-arg,unknown]
            (n_states, n_states),
            _apply_neumann_preconditioner,
            dtype=np.complex128,  # type: ignore[type-arg],
        )
        if config.use_neumann_preconditioner
        else None
    )

    resid_bar = tqdm(total=1.0, desc="Total Convergence", position=1, leave=False)

    def _callback(pr_norm: float) -> None:
        error = max(0, round(np.log10(pr_norm / config.precision), 3))
        next_progress = round(resid_bar.total - error, 3)
        if next_progress < 0:
            resid_bar.reset(total=error)
            next_progress = 0

        resid_bar.n = next_progress
        resid_bar.refresh()

    restart = min(config.max_iterations, target_state.size)
    solution, gmres_info = cast(
        "tuple[np.ndarray[Any, np.dtype[np.complex128]], int]",
        scipy.sparse.linalg.gmres(  # type: ignore[unknown]
            A=linear_operator,
            b=inverse_lower.matvec(target_state),  # type: ignore[unknown]
            x0=initial_state
            if initial_state is None
            else inverse_lower.matvec(initial_state),  # type: ignore[unknown]
            rtol=config.precision,
            restart=restart,
            maxiter=config.max_iterations,
            M=preconditioner,
            callback=_callback,
            callback_type="pr_norm",
        ),
    )
    resid_bar.close()

    if gmres_info != 0:
        msg = (
            "SciPy GMRES did not converge "
            f"(info={gmres_info}, max_iterations={config.max_iterations}, "
            f"restart={restart})."
        )
        raise RuntimeError(msg)

    return solution if lower is None else cast("Any", lower.matvec(solution))  # type: ignore[unknown]
