from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from array_api_compat.common import array_namespace as _namespace

if TYPE_CHECKING:
    from array_api._2025_12 import Array, ArrayNamespaceFull


type AnyArray = Array[Any, Any]
type Namespace[TArray: AnyArray] = ArrayNamespaceFull[TArray, Any, Any]
type LinearOperator[TArray: AnyArray] = Callable[[TArray], TArray]


def array_namespace[TArray: AnyArray](x: TArray) -> Namespace[TArray]:
    return _namespace(x)  # type: ignore[return-value]


def _scalar_float(x: AnyArray) -> float:
    try:
        return float(x)
    except TypeError:
        return float(x.item())  # type: ignore[union-attr]


def _zeros[TArray: AnyArray](
    shape: tuple[int, ...],
    ref: TArray,
) -> TArray:
    xp = array_namespace(ref)
    kwargs: dict[str, object] = {"dtype": ref.dtype}
    device = getattr(ref, "device", None)
    if device is not None:
        kwargs["device"] = device
    try:
        return xp.zeros(shape, **kwargs)
    except TypeError:
        return xp.zeros(shape, dtype=ref.dtype)


def _check_nan(vec: AnyArray, msg: str) -> None:
    xp = array_namespace(vec)
    if bool(_scalar_float(xp.any(xp.isnan(vec)))):
        raise ValueError(msg)


def _safe_normalize[TArray: AnyArray](
    x: TArray,
    threshold: float | None = None,
) -> tuple[TArray, TArray]:
    xp = array_namespace(x)
    norm = xp.linalg.vector_norm(x)
    limit = xp.finfo(x.dtype).eps if threshold is None else threshold
    normalized_x = x / norm if _scalar_float(norm) > limit else xp.zeros_like(x)
    return normalized_x, norm


def arnoldi[TArray: AnyArray](
    vec: TArray,  # Matrix vector product
    v: list[TArray],  # List of existing basis
    hessian: TArray,  # H matrix
    j: int,
) -> TArray:
    """
    Arnoldi iteration to find the j th l2-orthonormal vector.

    compute the j-1 th column of Hessenberg matrix.
    """
    _check_nan(vec, "Matrix vector product is Nan")
    xp = array_namespace(vec)

    for i in range(j):
        hessian[i, j - 1] = xp.sum(vec * v[i])
        vec = vec - hessian[i, j - 1] * v[i]
    new_v, v_norm = _safe_normalize(vec)
    hessian[j, j - 1] = v_norm
    return new_v


def apply_cal_rotation[TArray: AnyArray](
    a: TArray,
    b: TArray,
) -> tuple[TArray, TArray]:
    r"""
    Apply the cal rotation.

    Args:
        a: element h in position j
        b: element h in position j+1
    Returns:
        cosine = a / \sqrt{a^2 + b^2}
        sine = - b / \sqrt{a^2 + b^2}.

    """
    xp = array_namespace(a)
    c = xp.sqrt(a * a + b * b)
    return a / c, -b / c


def apply_given_rotation[TArray: AnyArray](
    hessian: TArray,
    cs: TArray,
    ss: TArray,
    j: int,
) -> tuple[TArray, TArray, TArray]:
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


def _a_matrix_as_function[TArray: AnyArray](
    a_matrix: TArray | LinearOperator[TArray],
) -> LinearOperator[TArray]:
    if callable(a_matrix):
        return a_matrix

    def out(vec: TArray) -> TArray:
        return a_matrix @ vec

    return out


def _identity_preconditioner[TArray: AnyArray](vec: TArray) -> TArray:
    return vec


def GMRES[TArray: AnyArray](  # noqa: N802, PLR0913
    a_matrix: TArray | LinearOperator[TArray],  # Linear operator, matrix or function
    b: TArray,
    initial_guess: TArray | None = None,
    preconditioner: TArray | LinearOperator[TArray] | None = None,
    max_iter: int | None = None,  # maximum number of GMRES iterations
    tol: float = 1e-6,  # relative tolerance
    atol: float = 1e-6,  # absolute tolerance
    *,
    error_callback: Callable[[float], None] | None = None,
) -> tuple[TArray, int]:
    """
    Perform GMRES to solve Ax=b.

    Reference: https://web.stanford.edu/class/cme324/saad-schultz.pdf

    Return:
        sol: solution
        j: number of iterations taken.

    """
    xp = array_namespace(b)

    apply_a_matrix = _a_matrix_as_function(a_matrix)
    apply_preconditioner = (
        _a_matrix_as_function(preconditioner)
        if preconditioner is not None
        else _identity_preconditioner
    )

    preconditioned_b = apply_preconditioner(b)
    target_state_norm = xp.linalg.vector_norm(preconditioned_b)

    if max_iter == 0 or _scalar_float(target_state_norm) < 1e-8:  # noqa: PLR2004
        return b, 0

    if max_iter is None:
        inferred_iter = int(b.shape[0])  # type: ignore[union-attr]
        max_iter = inferred_iter

    if initial_guess is None:
        initial_guess = xp.zeros_like(b)
        r0 = b
    else:
        r0 = b - apply_a_matrix(initial_guess)

    preconditioned_r0 = apply_preconditioner(r0)
    new_v, r_norm = _safe_normalize(preconditioned_r0)
    # initial guess residual
    beta = _zeros((max_iter + 1,), b)
    beta[0] = r_norm
    if error_callback is not None:
        error_callback(_scalar_float(r_norm / target_state_norm))

    v = list[TArray]()
    v.append(new_v)
    hessian = _zeros((max_iter + 1, max_iter + 1), b)
    cs = _zeros((max_iter,), b)
    ss = _zeros((max_iter,), b)

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
        residual = xp.abs(beta[j + 1])
        if error_callback is not None:
            error_callback(_scalar_float(residual / target_state_norm))

        if (
            _scalar_float(residual) < tol * _scalar_float(target_state_norm)
            or _scalar_float(residual) < atol
        ):
            break
    y = xp.linalg.solve(
        hessian[0 : j + 1, 0 : j + 1],
        xp.expand_dims(beta[0 : j + 1], axis=-1),
    )
    v_matrix = xp.stack(v[:-1], axis=0)
    sol = initial_guess + xp.permute_dims(v_matrix, (1, 0)) @ xp.squeeze(y, axis=-1)
    return sol, j
