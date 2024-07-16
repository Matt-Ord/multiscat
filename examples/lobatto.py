import matplotlib.pyplot as plt
import numpy as np

from multiscat.lobatto import get_lobatto_points

if __name__ == "__main__":
    # Ok up until around 35 points...
    # Need to consider if this is the best implimentation...
    lobatto_points = get_lobatto_points(40, (0, 2))

    fig, ax = plt.subplots()
    points = np.linspace(lobatto_points.points[0], lobatto_points.points[-1], 10000)
    points = lobatto_points.points
    for polynomial in lobatto_points.polynomials:
        ax.plot(points, polynomial(points))
    fig.show()
    input()
