import matplotlib.pyplot as plt
import numpy as np

from multiscat.lobatto import LobattoMetadata, get_polynomials

if __name__ == "__main__":
    # An example visualizing the Lobatto polynomials, used to represent
    # the scattering potential in a Lobatto basis.
    # The lobatto polynomials are particularly well suited for
    # representing the scattering potential.

    # TODO: this breaks around 35 points  # noqa: FIX002
    # This is due to an issue with finding the Legendre derivative roots
    # we need a better aproach!
    lobatto_metadata = LobattoMetadata(35, 2.0)

    points = np.linspace(
        lobatto_metadata.values[0],
        lobatto_metadata.values[-1],
        10000,
        dtype=np.float64,
    )
    fig, ax = plt.subplots()  # type: ignore unknown
    for polynomial in get_polynomials(lobatto_metadata):
        ax.plot(points, polynomial(points))  # type: ignore unknown
    fig.show()
    input()
