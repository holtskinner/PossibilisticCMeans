import numpy as np
from matplotlib import pyplot as plt


def plot(xy):

    t = np.ndarray.transpose(xy)
    fig, ax = plt.subplots()
    colors = ["#cc79a7", "#0072b2", "#d55e00", "#009e73"]

    ax.scatter(
        t[0],
        t[1],
        c=colors[0],
        s=50,
        label=f"Class {0}",
        alpha=0.5,
        edgecolors="none")

    ax.legend()
    ax.grid(True)
    plt.show()
