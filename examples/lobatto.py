import matplotlib.pyplot as plt
import numpy as np

from multiscat.lobatto import LobattoMetadata, get_polynomials

if __name__ == "__main__":
    # Ok up until around 35 points
    # This is due to an issue with finding the Legendre derivative roots
    lobatto_metadata = LobattoMetadata(35, 2.0)

    fig, ax = plt.subplots()  # type: ignore unknown
    points = np.linspace(
        lobatto_metadata.values[0],
        lobatto_metadata.values[-1],
        10000,
        dtype=np.float64,
    )

    for polynomial in get_polynomials(lobatto_metadata):
        ax.plot(points, polynomial(points))  # type: ignore unknown
    fig.show()
    input()
