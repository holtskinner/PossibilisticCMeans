import numpy as np
from matplotlib import pyplot as plt


def plot(x, v, u, c, labels=None):

    ax = plt.subplots()[1]

    colors = ["#000000", "#0072b2", "#cc79a7", "#009e73"]

    # Plot assigned clusters, for each data point in training set
    cluster_membership = np.argmax(u, axis=0)

    for j in range(c):
        ax.scatter(
            x[0][cluster_membership == j],
            x[1][cluster_membership == j],
            # label=labels[j],
            alpha=0.5,
            edgecolors="none",
            c=colors[j])

    # Mark the center of each fuzzy cluster
    for pt in v:
        ax.plot(pt[0], pt[1], 'rs')

    ax.legend()
    ax.grid(True)
    plt.show()
