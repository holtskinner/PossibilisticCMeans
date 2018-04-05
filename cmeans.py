import numpy as np
from scipy.spatial.distance import cdist


def _eta(u, d, m):

    u = u ** m
    n = np.sum(u * d, axis=1) / np.sum(u, axis=1)

    return n


def _update_clusters(x, u, m):
    um = u ** m
    v = um.dot(x.T) / np.atleast_2d(um.sum(axis=1)).T
    return v


def _hcm_criterion(x, v, n, m, metric):

    d = cdist(x.T, v, metric=metric)

    y = np.argmin(d, axis=1)

    u = np.zeros((v.shape[0], x.shape[1]))

    for i in range(x.shape[1]):
        u[y[i]][i] = 1

    return u, d


def _fcm_criterion(x, v, n, m, metric):

    d = cdist(x.T, v, metric=metric).T

    # Sanitize Distances (Avoid Zeroes)
    d = np.fmax(d, np.finfo(x.dtype).eps)

    exp = -2. / (m - 1)
    d2 = d ** exp

    u = d2 / np.sum(d2, axis=0, keepdims=1)

    return u, d


def _pcm_criterion(x, v, n, m, metric):

    d = cdist(x.T, v, metric=metric)
    d = np.fmax(d, np.finfo(x.dtype).eps)

    d2 = (d ** 2) / n
    exp = 1. / (m - 1)
    d2 = d2.T ** exp
    u = 1. / (1. + d2)

    return u, d


def _cmeans(x, c, m, e, max_iterations, criterion_function, metric="euclidean", v0=None, n=None):

    if not x.any() or len(x) < 1 or len(x[0]) < 1:
        print("Error: Data is in incorrect format")
        return

    # Num Features, Datapoints
    S, N = x.shape

    if not c or c <= 0:
        print("Error: Number of clusters must be at least 1")

    if not m:
        print("Error: Fuzzifier must be greater than 1")
        return

    # Initialize the cluster centers
    # If the user doesn't provide their own starting points,
    if v0 is None:
        # Pick random values from dataset
        xt = x.T
        v0 = xt[np.random.choice(xt.shape[0], c, replace=False), :]

    # List of all cluster centers (Bookkeeping)
    v = np.empty((max_iterations, c, S))
    v[0] = np.array(v0)

    # Membership Matrix Each Data Point in eah cluster
    u = np.zeros((max_iterations, c, N))

    # Number of Iterations
    t = 0

    while t < max_iterations - 1:

        u[t], d = criterion_function(x, v[t], n, m, metric)
        v[t + 1] = _update_clusters(x, u[t], m)

        # Stopping Criteria
        if np.linalg.norm(v[t + 1] - v[t]) < e:
            break

        t += 1

    return v[t], v[0], u[t - 1], u[0], d, t


# Public Facing Functions
def hcm(x, c, e, max_iterations, metric="euclidean", v0=None):
    return _cmeans(x, c, 1, e, max_iterations, _hcm_criterion, metric, v0=v0)


def fcm(x, c, m, e, max_iterations, metric="euclidean", v0=None):

    return _cmeans(x, c, m, e, max_iterations, _fcm_criterion, metric, v0=v0)


def pcm(x, c, m, e, max_iterations, metric="euclidean", v0=None):
    """

    Parameters
    ---

    `x` 2D array, size (S, N)
        Data to be clustered. N is the number of data sets;
        S is the number of features within each sample vector.

    `c` int
        Number of clusters

    `m` float, optional
        Fuzzifier

    `e` float, optional
        Convergence threshold

    `max_iterations` int, optional
        Maximum number of iterations

    `v0` array-like, optional
        Initial cluster centers

    Returns
    ---

    `v` 2D Array, size (S, c)
        Cluster centers

    `v0` 2D Array (S, c)
        Inital Cluster Centers

    `u` 2D Array (S, N)
        Final partitioned matrix

    `u0` 2D Array (S, N)
        Initial partition matrix

    `d` 2D Array (S, N)
        Distance Matrix

    `t` int
        Number of iterations run

    """

    v, v0, u, u0, d, t = fcm(x, c, m, e, max_iterations, metric=metric, v0=v0)
    n = _eta(u, d, m)
    return _cmeans(x, c, m, e, t, _pcm_criterion, metric, v0=v, n=n)
