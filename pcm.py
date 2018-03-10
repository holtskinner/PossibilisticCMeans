import numpy as np
import skfuzzy as fuzz
from scipy.spatial.distance import cdist


def eta():
    return 1


def fcm_criterion_function():
    return


def pcm_criterion_function(x, v, n, m, metric="euclidean"):

    # Criterion Function
    d = cdist(x.T, v, metric=metric).T / n
    u = 1 / (1 + d**(1 / (m - 1)))

    # Update Clusters
    um = u**m
    v = (um.dot(x.T).T / um.sum(axis=1)).T

    print(v)

    return u, v


def fcm(x, c, m=2, e=0.00001, max_iterations=1000, v0=None):
    return


def pcm(x, c, m=2, e=0.0001, max_iterations=1000, v0=None):
    """
    Possibilistic C-Means Algorithm

    # Parameters

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

    # Returns

    `v` 2D Array, size (S, c)  
        Cluster centers

    `u` 2D Array (S, N)  
        Final partitioned matrix
    
    `u0` 2D Array (S, N)  
        Initial partition matrix
    
    `d` 2D Array (S, N)  
        Distance Matrix
    
    `t` int  
        Number of iterations run

    `f` float  
        Final fuzzy partition coeffiient

    """

    if not x.any() or len(x) < 1 or len(x[0]) < 1:
        print("Error: Data is in incorrect format")
        return

    # Num Features, Datapoints
    S, N = x.shape

    if not c or c <= 0:
        print("Error: Number of clusters must be at least 1")

    if not m or m <= 1:
        print("Error: Fuzzifier must be greater than 1")
        return

    # Initialize the cluster centers
    # If the user doesn't provide their own starting points,
    if v0 is None or len(v0) != c:
        # Pick random values from dataset
        v0 = x.T[np.random.choice(N, c, replace=True), :]

    # List of all cluster centers (Bookkeeping)
    v = np.zeros((max_iterations, c, S))
    v[0] = np.array(v0)

    # Membership Matrix Each Data Point in eah cluster
    u = np.zeros((max_iterations, c, N))

    n = eta()

    # Number of Iterations
    t = 0

    while t < max_iterations - 1:

        u[t], v[t] = pcm_criterion_function(x, v[t - 1], n, m)

        # Stopping Criteria
        if np.linalg.norm(v[t] - v[t - 1]) < e:
            break

        t += 1

    return v[t], u[t - 1], u[0], None, t, None
