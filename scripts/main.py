from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from multiscat.config import (
    NMAX,
    NVFCFIXED_MAX,
    NZFIXED_MAX,
    Config,
    GMRESConfig,
    print_config,
    read_config,
)
from multiscat.fixed_potential import load_fixed_potential
from multiscat.lobatto import LobattoPoints, get_lobatto_derivatives

if TYPE_CHECKING:
    from multiscat.basis import XYBasis
    from multiscat.fixed_potential import FixedPotential
    from multiscat.scattering_condition import ScatteringCondition


def get_lobatto_t_matrix(
    points: LobattoPoints,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Calculate the kinetic energy matrix, T, in a normalized Lobatto basis.

    Formula for this are taken from:
    "QUANTUM SCATTERING VIA THE LOG DERIVATIVE OF THE KOHN VARIATIONAL PRINCIPLE"
    D. E. Manolopoulos and R. E. Wyatt, Chem. Phys. Lett., 1988, 152,23
    """
    derivatives = get_lobatto_derivatives(points)

    # We make use of the formula
    # T_ij = \sum_k=0 M+1 \omega_k u_i'(R_k) u'_j(R_k)
    # to calculate the kinetic matrix T_ij
    return np.einsum("k,ik,jk->ij", points.weights, derivatives, derivatives)  # type: ignore einsum


def get_scattering_energy(
    basis: XYBasis,
    condition: ScatteringCondition,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Get the matrix of scattered energies d.

    Uses the formula
    d = |k in + k scatter|**2 - |k in|**2
    """
    pkx, pky, _pkz = condition.momentum
    abs_squared_k = np.linalg.norm(condition.momentum) ** 2
    # TODO: previously we worked in a sparse basis in d
    # such that di < config.dmax
    # These channels will only have a small scattering contribution
    # It might be good to do this too!

    (kx, ky) = basis.k_points_stacked

    e_int = (pkx + kx) ** 2 + (pky + ky) ** 2
    # d is the difference between the initial energy
    # and the final energy
    return e_int - abs_squared_k


def waves(
    zmax: float,
    w0: float,
) -> tuple[complex, complex, complex]:
    """Construct and store the diagonal matrices a, b, and c.

    As defined in the the log derivative Kohn expression for the S-matrix.

    Parameters
    ----------
    w0 (float): Input value w0.
    zmax (float): Maximum value of z.

    Returns
    -------
    Tuple[np.complex128, np.complex128, np.complex128]:
        - a (np.complex128): Diagonal matrix a.
        - b (np.complex128): Diagonal matrix b.
        - c (np.complex128): Diagonal matrix c.

    """
    dk = np.sqrt(abs(w0))

    if w0 < 0.0:
        theta = dk * zmax
        bcc = np.cos(2.0 * theta)
        bcs = np.sin(2.0 * theta)
        cc = np.cos(theta)
        cs = np.sin(theta)

        a = complex(bcc, -bcs)
        b = (dk**0.5) * complex(cc, -cs)
        c = complex(0.0, dk)
        return a, b, c

    a = complex(0.0, 0.0)
    b = complex(0.0, 0.0)
    c = complex(-dk, 0.0)

    return a, b, c


def precon(
    m: int,
    n: int,
    vfc: np.ndarray[Any, Any],
    d: np.ndarray[Any, Any],
    t: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Construct the matrix factors required for the block lower triangular preconditioner.

    This is required for use in GMRES.

    Parameters
    ----------
    m (int): Size parameter.
    n (int): Size parameter.
    vfc (np.ndarray): Fourier component matrix (complex128).
    nfc (int): Number of Fourier components.
    nfc00 (int): Zero Fourier component.
    d (np.ndarray): Array for d values.
    e (np.ndarray): Array for e values (output).
    f (np.ndarray): Array for f values (output).
    t (np.ndarray): Matrix from tshapes (modified in-place).

    Returns
    -------
    None

    """
    nfc00 = 0
    if m > t.shape[0]:
        msg = "precon 1"
        raise ValueError(msg)

    # Modify t to be H0 by adding the real part of vfc(k, nfc00) to its diagonal elements
    h0 = t.copy()
    for k in range(m):
        h0[k, k] += np.real(vfc[k, nfc00])

    # Get eigenvalues and eigenvectors of h0 (symmetric real matrix)
    e, v = np.linalg.eigh(h0)

    f = np.zeros_like(v)

    for j in range(n):
        g = np.zeros(m, dtype=np.complex128)

        for k in range(m):
            g[k] = v[m - 1, k] / (d[j] + e[k])
            f[k, j] = 0.0

        for i in range(m):
            for k in range(m):
                f[k, j] += v[k, i] * g[i]
    return e, v, f


ComplexArray = np.ndarray[Any, np.dtype[np.complex128]]
DoubleArray = np.ndarray[Any, np.dtype[np.float64]]
IntArray = np.ndarray[Any, np.dtype[np.int64]]


def zrotg(a: complex, b: complex) -> tuple[complex, complex, complex, complex]:
    """Construct and perform a complex Givens rotation.

    Parameters
    ----------
    a : complex number
        First input value.
    b : complex number
        Second input value.

    Returns
    -------
    r : complex number
        Value of r after rotation.
    z : complex number
        Value of z after rotation.
    c : complex number
        c after rotation.
    s : complex number
        s after rotation.

    """
    # Initialize variables
    scale = abs(a) + abs(b)
    if scale == 0.0:
        c = 1.0
        s = 0.0
        r = 0.0
        z = 0.0
    else:
        r = np.sqrt((abs(a) / scale) ** 2 + (abs(b) / scale) ** 2) * scale
        r = -r if a.real * b.real < 0 else r
        c = a / r
        s = b / r
        z = 1.0 if abs(a) >= abs(b) and c != 0.0 else (1.0 / c)

    return r, z, c, s


def gmres(
    m: int,
    ix: IntArray,
    iy: IntArray,
    n: int,
    n00: int,
    vfc: ComplexArray,
    nfc: int,
    _a: ComplexArray,
    _b: ComplexArray,
    _c: ComplexArray,
    d: DoubleArray,
    e: DoubleArray,
    f: ComplexArray,
    t: ComplexArray,
    eps: float,
    ipc: int,
) -> ComplexArray:
    """Complex Generalised Minimal Residual Algorithm (GMRES) subroutine."""
    # Setup constants
    l = 2000  # parameter (l = 2000)
    h = np.zeros((l + 1, l + 1), dtype=np.complex128)
    g = np.zeros(l + 1, dtype=np.complex128)
    z = np.zeros(l + 1, dtype=np.complex128)
    co = np.zeros(l + 1, dtype=np.complex128)
    si = np.zeros(l + 1, dtype=np.complex128)
    temp = np.zeros(1, dtype=np.complex128)

    # Store x matrices in xx rather than write to disk
    xx = np.zeros((m * n, l + 1), dtype=np.complex128)

    # Setup for GMRES(l)
    mn = m * n
    x = np.zeros(mn, dtype=np.complex128)
    s = np.zeros(NMAX, dtype=np.complex128)

    # Initial step
    kount = 0
    xx[:, 0] = x[:]
    y = x.copy()
    ivx, ivy = ix, iy
    upper(x, m, ix, iy, n, vfc, ivx, ivy, nfc)
    x[:] = -x[:]
    x[m * n00] = b[n00] + x[m * n00]
    lower(x, m, ix, iy, n, vfc, ivx, ivy, nfc, c, d, e, f, t)
    x[:] = x[:] - y[:]
    if ipc == 1:
        y[:] = x[:]
        upper(x, m, ix, iy, n, vfc, ivx, ivy, nfc)
        lower(x, m, ix, iy, n, vfc, ivx, ivy, nfc, c, d, e, f, t)
        x[:] = y[:] - x[:]

    xnorm = np.linalg.norm(x)
    g[0] = xnorm

    # Generic recursion (iteration)
    kconv = 0
    p = np.zeros(n, dtype=np.complex128)
    kk = 1
    for k in range(1, l + 1):
        kount += 1
        x /= xnorm
        xx[:, k] = x[:]
        y[:] = x[:]
        upper(x, m, ix, iy, n, vfc, ivx, ivy, nfc)
        lower(x, m, ix, iy, n, vfc, ivx, ivy, nfc, c, d, e, f, t)
        x[:] = y[:] + x[:]

        if ipc == 1:
            y[:] = x[:]
            upper(x, m, ix, iy, n, vfc, ivx, ivy, nfc)
            lower(x, m, ix, iy, n, vfc, ivx, ivy, nfc, c, d, e, f, t)
            x[:] = y[:] - x[:]

        y[:] = xx[:, 0]
        s[:] = y[m * np.arange(1, n + 1)]

        for j in range(1, k):
            y[:] = xx[:, j]
            h[j, k - 1] = np.sum(np.conj(y[:]) * x[:])
            x[:] -= y[:] * h[j, k - 1]

            if j < k:
                for i in range(1, n + 1):
                    s[i - 1] += y[m * i - 1] * z[j - 1]

        for i in range(1, n + 1):
            s[i - 1] = (0.0 + 2.0j) * b[i - 1] * s[i - 1]

        s[n00] = a[n00] + s[n00]
        xnorm = np.linalg.norm(x)
        h[k, k - 1] = xnorm

        for j in range(1, k):
            temp[:] = co[j - 1] * h[j - 1, k - 1] + np.conj(si[j - 1]) * h[j, k - 1]
            h[j, k - 1] = np.conj(co[j - 1]) * h[j, k - 1] - si[j - 1] * h[j - 1, k - 1]
            h[j - 1, k - 1] = temp[:]

        _, _, co[k - 1], si[k - 1] = zrotg(h[k - 1, k - 1], h[k, k - 1])
        g[k] = -si[k - 1] * g[k - 1]
        g[k - 1] = co[k - 1] * g[k - 1]

        z[:] = g[:]

        for j in range(k, 0, -1):
            z[j - 1] /= h[j - 1, j - 1]

            for i in range(1, j):
                z[i - 1] -= h[i - 1, j - 1] * z[j - 1]

        # Convergence test
        unit = 0.0
        diff = 0.0
        for j in range(1, n + 1):
            pj = np.conj(s[j - 1]) * s[j - 1]
            unit += pj
            diff = max(diff, abs(pj - p[j - 1]))
            p[j - 1] = pj

        diff = max(diff, abs(unit - 1.0))

        if diff < eps:
            kconv += 1
        else:
            kconv = 0

        kk = k
        if kconv == 3 or xnorm == 0.0:
            break

    # Back substitution for x
    x[:] = xx[:, 0]
    for j in range(1, kk + 1):
        y[:] = xx[:, j]
        x += y * z[j - 1]

    # All done?
    if kconv < 3 and xnorm > 0.0:
        pass

    # Yes!
    return p


def upper(
    x: ComplexArray,
    m: int,
    ix: IntArray,
    iy: IntArray,
    n: int,
    vfc: ComplexArray,
    ivx: IntArray,
    ivy: IntArray,
    nfc: int,
) -> None:
    """Performs the block upper triangular matrix multiplication y = U*x, where A = L+U.
    The result y is overwritten on x on return.
    """
    for j in range(1, n + 1):
        x[:, j - 1] = 0.0j
        for i in range(j + 1, n + 1):
            for l in range(1, nfc + 1):
                if ix[i - 1] + ivx[l - 1] != ix[j - 1]:
                    continue
                if iy[i - 1] + ivy[l - 1] != iy[j - 1]:
                    continue
                x[:, j - 1] += vfc[:, l - 1] * x[:, i - 1]


def lower(
    x: ComplexArray,
    m: int,
    ix: IntArray,
    iy: IntArray,
    n: int,
    vfc: ComplexArray,
    ivx: IntArray,
    ivy: IntArray,
    nfc: int,
    c: ComplexArray,
    d: DoubleArray,
    e: DoubleArray,
    f: ComplexArray,
    t: ComplexArray,
) -> None:
    """Solves the block lower triangular linear equation L*y = x, where A = L+U.
    The result y is overwritten on x on return.
    """
    mmax = 200
    y = np.zeros(mmax, dtype=np.complex128)
    if m > mmax:
        msg = "m exceeds mmax in lower subroutine"
        raise ValueError(msg)

    for j in range(1, n + 1):
        for i in range(1, j):
            for l in range(1, nfc + 1):
                if ix[i - 1] + ivx[l - 1] != ix[j - 1]:
                    continue
                if iy[i - 1] + ivy[l - 1] != iy[j - 1]:
                    continue
                x[:, j - 1] -= vfc[:, l - 1] * x[:, i - 1]

        for k in range(1, m + 1):
            y[k - 1] = 0.0j
            for l in range(1, m + 1):
                y[k - 1] += x[l - 1, j - 1] * t[l - 1, k - 1]
            y[k - 1] /= d[j - 1] + e[k - 1]

        for k in range(1, m + 1):
            x[k - 1, j - 1]


def process_scattering_condition(
    potential: FixedPotential,
    condition: ScatteringCondition,
    config: GMRESConfig,
) -> None:
    lobatto_points = potential.lobatto_basis
    mz = lobatto_points.points.size - 1
    t_matrix = get_lobatto_t_matrix(lobatto_points)

    # get reciprocal lattice points
    # (also calculate how many channels are required for the calculation)
    d = get_scattering_energy(potential.xy_basis, condition)
    # TODO: write to output
    # if out_file is not None:
    #     with out_file.open("a") as f:
    #         f.write(
    #             f"Number of diffraction channels, n ={d.size}\n",
    #         )

    a = np.zeros(d.size, dtype=np.complex128)
    b = np.zeros(d.size, dtype=np.complex128)
    c = np.zeros(d.size, dtype=np.complex128)

    # TODO: is is sqrt omega here, and is it n-1 elements...
    w = lobatto_points.weights
    z_max = lobatto_points.points[-1]
    for i in range(d.size):
        a[i], b[i], c[i] = waves(z_max, d[i])
        b[i] = b[i] / w[mz]
        c[i] = c[i] / (w[mz] ** 2)

    e, v, f = precon(
        mz,
        d.size,
        potential.data,
        d,
        t_matrix,
    )

    gmres(
        mz,
        potential.xy_basis.nk_points_stacked[0],
        potential.xy_basis.nk_points_stacked[1],
        d.size,
        0,
        potential.data,
        np.prod(potential.xy_basis.shape).item(),
        a,
        b,
        c,
        d,
        e,
        f,
        v,
        config.precision,
        config.preconditioner,
    )

    # TODO: write outputs
    # output(ei, theta, phi, ix, iy, d.size, n00, d, p, config.itest)


def process_potentials(
    config: Config,
) -> None:
    np.zeros((NZFIXED_MAX, NVFCFIXED_MAX), dtype=np.complex128)

    for index in range(config.startindex, config.endindex + 1):
        fourier_file = Path(f"pot{index:05d}.in")
        out_file = Path(f"diffrac{index:05d}.out") if config.itest == 1 else None

        if out_file is not None:
            with out_file.open("a") as f:
                f.write(f"Diffraction intensities for potential: {fourier_file}\n")

        # TODO: Calculate scattering over the incident conditions required
        print("")
        print(f"Calculating scattering for potential: {fourier_file}")
        print("Energy / meV    Theta / deg    Phi / deg        I00         Sum")

        # Initialize the potential
        fixed_potential = load_fixed_potential(config, fourier_file)
        for condition in config.scattering_conditions:
            process_scattering_condition(
                fixed_potential,
                condition,
                config.gmres_config,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multiscat: Close Coupled Scattering Program",
    )
    parser.add_argument("inputfile", type=str, help="Path to the configuration file")

    args = parser.parse_args()

    inputfile = args.inputfile
    if not inputfile:
        print("Error: you must supply a configuration file to run Multiscat.")
        return

    print("\nMultiscat: Close Coupled Scattering Program")
    print("=============================================\n")
    print(f"Reading parameters from input file: {inputfile}\n")

    # Read parameters from config file
    config = read_config(Path(inputfile))

    # Print parameters
    print_config(config)

    process_potentials(config)


if __name__ == "__main__":
    main()
