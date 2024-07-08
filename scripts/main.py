from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from multiscat.config import NMAX, NVFCFIXED_MAX, NZFIXED_MAX, Config
from multiscat.fixed_potential import interpolate_potential_z, load_fixed_potential

if TYPE_CHECKING:
    from multiscat.scattering_condition import ScatteringCondition


def _parse_value(line: str) -> str:
    return line.split("!")[0].strip()


def read_config(file_path: Path) -> Config:
    with file_path.open("r") as file:
        lines = file.readlines()

    fourier_labels_file = _parse_value(lines[0])
    scatt_cond_file = _parse_value(lines[1])
    itest = int(_parse_value(lines[2]))
    ipc = int(_parse_value(lines[3]))
    ipc = min(max(ipc, 0), 1)
    nsf = int(_parse_value(lines[4]))
    nsf = min(max(nsf, 2), 5)
    eps = 0.5 * (10 ** (-nsf))
    nfc = int(_parse_value(lines[5]))
    zmin, zmax = map(float, _parse_value(lines[6]).split(","))
    vmin = float(_parse_value(lines[7]))
    dmax = float(_parse_value(lines[8]))
    imax = int(_parse_value(lines[9]))
    a1 = float(_parse_value(lines[10]))
    a2 = float(_parse_value(lines[11]))
    b2 = float(_parse_value(lines[12]))
    nzfixed = int(_parse_value(lines[13]))
    stepzmin = float(_parse_value(lines[14]))
    stepzmax = float(_parse_value(lines[15]))
    startindex = int(_parse_value(lines[16]))
    endindex = int(_parse_value(lines[17]))
    he_mass = float(_parse_value(lines[18]))

    return Config(
        fourier_labels_file=fourier_labels_file,
        scatt_cond_file=scatt_cond_file,
        itest=itest,
        ipc=ipc,
        eps=eps,
        nsf=nsf,
        nfc=nfc,
        zmin=zmin,
        zmax=zmax,
        vmin=vmin,
        dmax=dmax,
        imax=imax,
        a1=a1,
        a2=a2,
        b2=b2,
        nzfixed=nzfixed,
        stepzmin=stepzmin,
        stepzmax=stepzmax,
        startindex=startindex,
        endindex=endindex,
        he_mass=he_mass,
    )


def print_config(config: Config) -> None:
    print(f"Fourier labels file = {config.fourier_labels_file}")
    print(f"Loading scattering conditions from {config.scatt_cond_file}")
    print(f"Output mode = {config.itest}")
    print(f"GMRES preconditioner flag = {config.ipc}")
    print(f"Convergence sig. figures = {config.nsf}")
    print(f"Total number of Fourier components to use = {config.nfc}")
    print(f"z integration range = ({config.zmin}, {config.zmax})")
    print(f"Max energy of closed channels = {config.dmax}")
    print(f"Max index of channels = {config.imax}")
    print(f"Unit cell (A) = {config.a1} x {config.b2}")
    print(f"Number of z points in Fourier components (nzfixed) = {config.nzfixed}")
    print(
        f"Calculating for potential input files between {config.startindex}.in"
        f"and {config.endindex}.in",
    )
    print(f"Atom Mass = {config.he_mass}")


@dataclass
class LobattoPoints:
    z_points: np.ndarray[Any, np.dtype[np.float64]]
    weights: np.ndarray[Any, np.dtype[np.float64]]


def get_lobatto_points(
    z_start: float,
    z_end: float,
    n_z: int,
) -> LobattoPoints:
    """Calculate an n-point Gauss-Lobatto quadrature rule in the interval a < x < b.

    Function localizes zeros of the derivative of the (n-1)th Legendre polynomial
    and calculates associated weights.
    """
    w = np.zeros(n_z)
    x = np.zeros(n_z)

    l = (n_z + 1) // 2
    shift = 0.5 * (z_end + z_start)
    scale = 0.5 * (z_end - z_start)
    weight = (z_end - z_start) / (n_z * (n_z - 1))

    # Specific to Lobatto quadrature, first point is a
    x[0] = z_start
    w[0] = weight

    for k in range(1, l + 1):
        # As zeros are symmetric, there is only need to find positive ones
        # z is approximated zero of P[n-1] using Francesco Tricomi approximation
        # then accuracy of the zero is improved using Newton-Raphson
        z = np.cos(np.pi * (4 * k - 3) / (4 * n_z - 2))
        p1 = 1.0
        for _i in range(7):
            p2 = 0.0
            p1 = 1.0
            for j in range(1, n_z):
                p3 = p2
                p2 = p1
                p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j

            # p2 gets overwritten to be P[n-1]'
            p2 = (n_z - 1) * (p2 - z * p1) / (1.0 - z * z)
            # p3 gets overwritten to be P[n-1]''
            p3 = (2.0 * z * p2 - n_z * (n_z - 1) * p1) / (1.0 - z * z)
            # Actual Newton-Raphson step
            z = z - p2 / p3

        # Write in shifted and scaled zeros and weights
        x[k - 1] = shift - scale * z
        x[n_z - k] = shift + scale * z

        # Write in weights (they are always positive)
        w[k - 1] = weight / (p1 * p1)
        w[n_z - k] = w[k - 1]

    # Specific to Lobatto quadrature, last point is b
    x[n_z - 1] = z_end
    w[n_z - 1] = weight

    return LobattoPoints(z_points=x, weights=w)


def get_lobatto_points_for_config(
    config: Config,
) -> LobattoPoints:
    n = config.mz + 1
    return get_lobatto_points(config.zmin, config.zmax, n)


def get_lobatto_derivatives(
    points: LobattoPoints,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Calculate the derivative matrix u_i'(R_j) for the lobatto basis.

    Parameters
    ----------
    points : LobattoPoints

    Returns
    -------
    np.ndarray[Any, np.dtype[np.float64]]

    """
    # The derivatives can be evaluated analytically
    # u_i'(R_i) = \sum_j=0 M+1 (R_i - R_j)^-1
    # or for j=\=i
    # u_i'(R_j) = (R_i - R_j)^-1 product_0^M+1 (R_j-R_k) / (R_i - R_k)
    # Where the product excludes k=j and k=i
    n_points = points.z_points.size

    # Calculate the reciprocal of differences (R_i - R_j)^-1, ignoring the diagonal
    diff = points.z_points[:, np.newaxis] - points.z_points[np.newaxis, :]
    reciprocal_diff = np.where(diff != 0, 1.0 / diff, 0)

    # Calculate product_k=0^M+1 (R_j-R_k) / (R_i - R_k)
    mask = np.eye(n_points, dtype=bool)
    products = np.prod(
        np.where(
            # Ignoring the zero elements from the product
            mask[np.newaxis, :, :] | mask[:, np.newaxis, :],
            1,
            (diff[np.newaxis, :, :] * reciprocal_diff[:, np.newaxis, :]),
        ),
        axis=2,
    )

    u_derivative = reciprocal_diff * products
    # Calculate diagonal elements seperately
    # u_i'(R_i) = \sum_j=0 M+1 (R_i - R_j)^-1
    u_derivative[np.arange(n_points), np.arange(n_points)] = np.sum(
        reciprocal_diff,
        axis=1,
    )
    return u_derivative


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
    config: Config,
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

    (kx, ky) = config.xy_basis.k_points_stacked

    e_int = (pkx + kx) ** 2 + (pky + ky) ** 2
    # d is the difference between the initial energy
    # and the final energy
    return e_int - abs_squared_k


def waves(
    config: Config,
    w0: float,
) -> tuple[complex, complex, complex]:
    """Construct and store the diagonal matrices a, b, and c that enter the log derivative Kohn expression for the S-matrix.

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
        theta = dk * config.zmax
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
    nfc00: int,
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
    ivx: DoubleArray,
    ivy: DoubleArray,
    nfc: int,
    a: ComplexArray,
    b: ComplexArray,
    c: ComplexArray,
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
    ivx: DoubleArray,
    ivy: DoubleArray,
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
    ivx: DoubleArray,
    ivy: DoubleArray,
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
    config: Config,
    fourier_file: Path,
    condition: ScatteringCondition,
    *,
    out_file: Path | None = None,
) -> None:
    lobatto_points = get_lobatto_points_for_config(config)
    t_matrix = get_lobatto_t_matrix(lobatto_points)

    # Initialize the potential
    fixed_potential = load_fixed_potential(config, fourier_file)
    interpolated_fixed_potential = interpolate_potential_z(
        fixed_potential,
        config.nfc,
        lobatto_points.z_points,
    )

    # get reciprocal lattice points
    # (also calculate how many channels are required for the calculation)
    d = get_scattering_energy(config, condition)
    if out_file is not None:
        with out_file.open("a") as f:
            f.write(
                f"Number of diffraction channels, n ={d.size}\n",
            )

    a = np.zeros(d.size, dtype=np.complex128)
    b = np.zeros(d.size, dtype=np.complex128)
    c = np.zeros(d.size, dtype=np.complex128)

    # TODO: is is sqrt omega here, and is it n-1 elements...
    w = lobatto_points.weights
    for i in range(d.size):
        a[i], b[i], c[i] = waves(config, d[i])
        b[i] = b[i] / w[config.mz]
        c[i] = c[i] / (w[config.mz] ** 2)

    ivx, ivy, nfc00 = config.label_fourier_components()

    e, v, f = precon(
        config.mz,
        d.size,
        interpolated_fixed_potential.data,
        nfc00,
        d,
        t_matrix,
    )

    gmres(
        config.mz,
        config.xy_basis.nk_points_stacked[0],
        config.xy_basis.nk_points_stacked[1],
        d.size,
        0,
        interpolated_fixed_potential.data,
        ivx,
        ivy,
        config.nfc,
        a,
        b,
        c,
        d,
        e,
        f,
        v,
        config.eps,
        config.ipc,
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

        for condition in config.scattering_conditions:
            process_scattering_condition(
                config,
                fourier_file,
                condition,
                out_file=out_file,
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
