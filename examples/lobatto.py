import numpy as np
from slate_core import plot
from slate_core.metadata import Domain, LobattoSpacedMetadata

from multiscat.polynomial import get_polynomials

if __name__ == "__main__":
    # An example visualizing the Lobatto polynomials, used to represent
    # the scattering potential in a Lobatto basis.
    # The lobatto polynomials are particularly well suited for
    # representing the scattering potential.

    # The lobatto weights are the weights such that
    # the integral of a polynomial is approximated by the sum
    # of the polynomial evaluated at the lobatto points, multiplied
    # by the weights.
    # int_s f(R) dR = sum_k f(R_k) w_k
    # where R_k are the lobatto points and w_k are the weights.
    # In an evenly spaced grid of points, these weights would simply be
    # the average step size between the points.
    #
    # For the lobatto basis to be normalized, the basis functions
    # are defined such that U(R_i)_j = delta_{i,j} / sqrt(w_i). This
    # means that
    # int_s U(R)_i U(R)_j dR = sum_k U_i(R_k)U_j(R_k) w_k = delta_{i,j}
    fig, ax = plot.get_figure()
    for meta in [
        # N=2 corresponds to the trapezium rule
        LobattoSpacedMetadata(2, domain=Domain(delta=7)),
        # N=3 corresponds to Simpson's rule
        LobattoSpacedMetadata(3, domain=Domain(delta=10)),
        LobattoSpacedMetadata(5, domain=Domain(delta=49)),
        LobattoSpacedMetadata(11, domain=Domain(delta=700)),
        LobattoSpacedMetadata(25, domain=Domain(delta=3)),
        LobattoSpacedMetadata(31, domain=Domain(delta=5)),
        LobattoSpacedMetadata(300, domain=Domain(delta=5)),
    ]:
        average_step = meta.delta / meta.fundamental_size
        (line,) = ax.plot(
            meta.values / meta.delta,
            meta.basis_weights / average_step,
        )
    ax.set_xlabel("Lobatto Point (normalized)")
    ax.set_ylabel("Lobatto Weights/ Average Step Size")
    ax.set_title("Lobatto Weights Normalized by Average Step Size")
    fig.show()

    # TODO: this breaks around 35 points  # noqa: FIX002
    # This is due to an issue with finding the Legendre derivative roots
    # we need a better aproach!
    lobatto_metadata = LobattoSpacedMetadata(
        35,
        domain=Domain(delta=2.0),
    )

    points = np.linspace(
        lobatto_metadata.values[0],
        lobatto_metadata.values[-1],
        10000,
        dtype=np.float64,
    )
    fig, ax = plot.get_figure()
    for polynomial in get_polynomials(lobatto_metadata):
        # These are the normalized lobatto polynomials,
        # note at the edge they are larger!
        ax.plot(points, polynomial(points))
    ax.set_title("Lobatto Polynomials")
    fig.show()

    plot.wait_for_close()
