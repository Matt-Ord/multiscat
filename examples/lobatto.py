import matplotlib.pyplot as plt
import numpy as np

from multiscat.lobatto import get_lobatto_points

if __name__ == "__main__":
    # Ok up until around 35 points
    # This is due to an issue with finding the Legendre derivative roots
    lobatto_points = get_lobatto_points(35, (-1, 1))

    fig, ax = plt.subplots()  # type: ignore unknown

    fig1, ax1 = plt.subplots()  # type: ignore unknown
    points = np.linspace(
        lobatto_points.points[0],
        lobatto_points.points[-1],
        10000,
        dtype=np.float64,
    )

    for polynomial in lobatto_points.polynomials:
        ax.plot(points, polynomial(points))  # type: ignore unknown
    fig.show()
    fig1.show()
    input()
