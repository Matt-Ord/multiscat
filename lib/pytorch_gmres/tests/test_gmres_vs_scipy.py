from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.sparse.linalg import gmres as scipy_gmres  # type: ignore[scipy]

from pytorch_gmres import GMRES

if TYPE_CHECKING:
    from array_api._2025_12 import Array


type AnyArray = Array[Any, Any]


def test_gmres_matches_scipy_solution() -> None:
    rng = np.random.default_rng(0)
    n = 10

    a = rng.standard_normal((n, n))
    a += n * np.eye(n)
    b = rng.standard_normal(n)

    ours, _ = GMRES(
        cast("AnyArray", a),
        cast("AnyArray", b),
        max_iter=50,
        tol=1e-10,
        atol=1e-10,
    )
    expected, _info = scipy_gmres(a, b, atol=1e-10, rtol=1e-10)  # type: ignore[scipy]

    np.testing.assert_allclose(np.asarray(ours), expected, rtol=1e-6, atol=1e-8)  # type: ignore[scipy]


def test_gmres_matches_scipy_with_preconditioner() -> None:
    rng = np.random.default_rng(1)
    n = 10

    a = rng.standard_normal((n, n))
    a += n * np.eye(n)
    b = rng.standard_normal(n)

    # Jacobi inverse preconditioner M ~= A^{-1} from diagonal entries.
    m_inv = np.diag(1.0 / np.diag(a))

    ours, _ = GMRES(
        cast("AnyArray", a),
        cast("AnyArray", b),
        preconditioner=cast("AnyArray", m_inv),
        max_iter=50,
        tol=1e-10,
        atol=1e-10,
    )
    expected, _info = scipy_gmres(  # type: ignore[scipy]
        a,
        b,
        M=m_inv,
        atol=1e-10,
        rtol=1e-10,
    )

    np.testing.assert_allclose(np.asarray(ours), expected, rtol=1e-6, atol=1e-8)  # type: ignore[scipy]
