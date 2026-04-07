from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
    from collections.abc import Callable


def _check_nan(vec: torch.Tensor, msg: str) -> None:
    if torch.isnan(vec).any():
        raise ValueError(msg)


def _safe_normalize(
    x: torch.Tensor,
    threshold: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    norm = cast("torch.Tensor", torch.norm(x))  # type: ignore[assignment]
    if threshold is None:
        threshold = torch.finfo(norm.dtype).eps
    normalized_x = x / norm if norm > threshold else torch.zeros_like(x)
    return normalized_x, norm


def arnoldi(
    vec: torch.Tensor,  # Matrix vector product
    v: list[torch.Tensor],  # List of existing basis
    hessian: torch.Tensor,  # H matrix
    j: int,
) -> torch.Tensor:
    """
    Arnoldi iteration to find the j th l2-orthonormal vector.

    compute the j-1 th column of Hessenberg matrix.
    """
    _check_nan(vec, "Matrix vector product is Nan")

    for i in range(j):
        hessian[i, j - 1] = torch.dot(vec, v[i])
        vec = vec - hessian[i, j - 1] * v[i]
    new_v, vnorm = _safe_normalize(vec)
    hessian[j, j - 1] = vnorm
    return new_v


def apply_cal_rotation(
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Apply the cal rotation.

    Args:
        a: element h in position j
        b: element h in position j+1
    Returns:
        cosine = a / \sqrt{a^2 + b^2}
        sine = - b / \sqrt{a^2 + b^2}.

    """
    c = torch.sqrt(a * a + b * b)
    return a / c, -b / c


def apply_given_rotation(
    hessian: torch.Tensor,
    cs: torch.Tensor,
    ss: torch.Tensor,
    j: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply givens rotation to H columns."""
    # apply previous rotation to the 0->j-1 columns
    for i in range(j):
        tmp = cs[i] * hessian[i, j] - ss[i] * hessian[i + 1, j]
        hessian[i + 1, j] = cs[i] * hessian[i + 1, j] + ss[i] * hessian[i, j]
        hessian[i, j] = tmp
    cs[j], ss[j] = apply_cal_rotation(hessian[j, j], hessian[j + 1, j])
    hessian[j, j] = cs[j] * hessian[j, j] - ss[j] * hessian[j + 1, j]
    hessian[j + 1, j] = 0
    return hessian, cs, ss


def _a_matrix_as_function(
    a_matrix: torch.Tensor
    | Callable[
        [torch.Tensor],
        torch.Tensor,
    ],
) -> Callable[[torch.Tensor], torch.Tensor]:
    if isinstance(a_matrix, torch.Tensor):

        def out(vec: torch.Tensor) -> torch.Tensor:
            return a_matrix @ vec

        return out
    if callable(a_matrix):
        return a_matrix
    msg = "A must be a function or matrix"
    raise ValueError(msg)


def _identity_preconditioner(vec: torch.Tensor) -> torch.Tensor:
    return vec


def GMRES(  # noqa: N802, PLR0913
    a_matrix: torch.Tensor
    | Callable[
        [torch.Tensor],
        torch.Tensor,
    ],  # Linear operator, matrix or function
    b: torch.Tensor,  # RHS of the linear system in which the first half has the same shape as grad_gx, the second half has the same shape as grad_fy
    initial_guess: torch.Tensor
    | None = None,  # initial guess, tuple has the same shape as b
    preconditioner: torch.Tensor
    | Callable[
        [torch.Tensor],
        torch.Tensor,
    ]
    | None = None,
    max_iter: int | None = None,  # maximum number of GMRES iterations
    tol: float = 1e-6,  # relative tolerance
    atol: float = 1e-6,  # absolute tolerance
    *,
    error_callback: Callable[[int], None] | None = None,
) -> tuple[torch.Tensor, int]:
    """
    Perform GMRES to solve Ax=b.

    Reference: https://web.stanford.edu/class/cme324/saad-schultz.pdf

    Return:
        sol: solution
        j: number of iterations taken.

    """
    apply_a_matrix = _a_matrix_as_function(a_matrix)
    apply_preconditioner = (
        _a_matrix_as_function(preconditioner)
        if preconditioner is not None
        else _identity_preconditioner
    )

    preconditioned_b = apply_preconditioner(b)
    target_state_norm = cast("torch.Tensor", torch.norm(preconditioned_b))  # type: ignore[assignment]

    if max_iter == 0 or target_state_norm < 1e-8:  # noqa: PLR2004
        return b, 0

    if max_iter is None:
        max_iter = b.shape[0]

    if initial_guess is None:
        initial_guess = torch.zeros_like(b)
        r0 = b
    else:
        r0 = b - apply_a_matrix(initial_guess)

    preconditioned_r0 = apply_preconditioner(r0)
    new_v, r_norm = _safe_normalize(preconditioned_r0)
    # initial guess residual
    beta = torch.zeros(max_iter + 1, device=b.device)
    beta[0] = r_norm
    if error_callback is not None:
        error_callback((r_norm / target_state_norm).item())  # type: ignore infer

    v = list[torch.Tensor]()
    v.append(new_v)
    hessian = torch.zeros((max_iter + 1, max_iter + 1), device=b.device)
    cs = torch.zeros(max_iter, device=b.device)  # cosine values at each step
    ss = torch.zeros(max_iter, device=b.device)  # sine values at each step

    j = 0
    for j in range(max_iter):
        p = apply_preconditioner(apply_a_matrix(v[j]))
        # Arnoldi iteration to get the j+1 th basis
        new_v = arnoldi(p, v, hessian, j + 1)
        v.append(new_v)

        hessian, cs, ss = apply_given_rotation(hessian, cs, ss, j)
        _check_nan(cs, f"{j}-th cosine contains NaN")
        _check_nan(ss, f"{j}-th sine contains NaN")
        beta[j + 1] = ss[j] * beta[j]
        beta[j] = cs[j] * beta[j]
        residual = torch.abs(beta[j + 1])
        if error_callback is not None:
            error_callback((residual / target_state_norm).item())  # type: ignore infer

        if residual < tol * target_state_norm or residual < atol:
            break
    y, _ = torch.triangular_solve(
        beta[0 : j + 1].unsqueeze(-1),
        hessian[0 : j + 1, 0 : j + 1],
    )
    v = torch.stack(v[:-1], dim=0)
    sol = initial_guess + v.T @ y.squeeze(-1)
    return sol, j
