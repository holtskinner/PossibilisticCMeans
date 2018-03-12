import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def plot(x, v, u, c, labels):

    ax = plt.subplots()[1]

    colors = ["#cc79a7", "#0072b2", "#d55e00", "#009e73"]

    # Plot assigned clusters, for each data point in training set
    cluster_membership = np.argmax(u, axis=0)

    for j in range(c):
        ax.scatter(
            x[2][cluster_membership == j],
            x[3][cluster_membership == j],
            label=labels[j],
            alpha=0.5,
            edgecolors="none",
            c=colors[j])

    # Mark the center of each fuzzy cluster
    for pt in v:
        ax.plot(pt[2], pt[3], 'rs')

    ax.legend()
    ax.grid(True)
    plt.show()
