import numpy as np
from matplotlib import pyplot as plt


def plot(x, v, u, c, labels=None):

    ax = plt.subplots()[1]

    # Plot assigned clusters, for each data point in training set
    cluster_membership = np.argmax(u, axis=0)

    for j in range(c):
        ax.scatter(
            x[0][cluster_membership == j],
            x[1][cluster_membership == j],
            alpha=0.5,
            edgecolors="none")

    # Mark the center of each fuzzy cluster
    for pt in v:
        ax.plot(pt[0], pt[1], 'rs')

    ax.legend()
    ax.grid(True)
    plt.show()
