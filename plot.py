import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def plot(x, v, u, c, labels=None):

    ax = plt.subplots()[1]

    # Plot assigned clusters, for each data point in training set
    cluster_membership = np.argmax(u, axis=0)

    x = PCA(n_components=2).fit_transform(x).T

    for j in range(c):
        ax.scatter(
            x[0][cluster_membership == j],
            x[1][cluster_membership == j],
            alpha=0.5,
            edgecolors="none")

    ax.legend()
    ax.grid(True)
    plt.show()
