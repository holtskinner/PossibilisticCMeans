import numpy as np
from matplotlib import pyplot as plt


def plot(xy, v, u, c):

    fig, ax = plt.subplots()

    # Plot assigned clusters, for each data point in training set
    cluster_membership = np.argmax(u, axis=0)

    for j in range(c):
        ax.scatter(
            xy[0][cluster_membership == j],
            xy[1][cluster_membership == j],
            label=f"Class {j}",
            alpha=0.5,
            edgecolors="none",
            c=np.random.random(3))

    # Mark the center of each fuzzy cluster
    for pt in v:
        ax.plot(pt[0], pt[1], 'rs')

    ax.grid(True)
    plt.show()
