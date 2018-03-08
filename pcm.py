import numpy as np
import skfuzzy as fuzz


def eta():
    return 1


def criterion_function(x, v, eta, m):
    d = np.square(np.linalg.norm(x - v))

    ex = 2 / (m - 1)
    fz = np.power(d / eta, ex)
    u = 1 / (1 + fz)
    return u


def update_cluster_centers(x, c, m, u):

    # v = np.zeros((c, ))
    # TODO Fix Dimensionality
    for i in range(c):
        numerator = 0
        denominator = 0

        for k in range(n):
            fuzzified = np.power(u[i, k], m)
            numerator += fuzzified * x[k]
            denominator += fuzzified

        print(numerator)

        print(denominator)
        print("\n")
        v[i] = numerator / denominator

    return v


def pcm(x, c, m=2, e=0.01, max_iterations=100, v0=None):
    """
    Possibilistic C-Means Algorithm

    ### Parameters

    `x` 2D array, size (N, S)  
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

    ### Returns

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

    N = len(x)  # Number of datapoints
    S = len(x[0])  # Number of features

    if not c or c <= 0:
        print("Error: Number of clusters must be at least 1")

    if not m or m <= 1:
        print("Error: Fuzzifier must be greater than 1")
        return

    # Initialize the cluster centers
    # If the user doesn't provide their own starting points,
    if not v0 or not v0.any() or len(v0) != c:
        # Pick random values from dataset
        v0 = x[np.random.choice(x.shape[0], c, replace=True), :]

    # List of all cluster centers (Bookkeeping)
    v = np.zeros((max_iterations, c, S))
    v[0] = np.array(v0)

    # Membership Matrix Each Data Point in eah cluster
    u = np.zeros((max_iterations, N, c))

    n = eta()

    # Number of Iterations
    t = 0

    while t < max_iterations - 1:

        # Cycle through datapoints's
        for k in range(N):
            # cycle through clusters
            for i in range(c):
                u[t, k, i] = criterion_function(x[k], v[t, i], n, m)

        t += 1

        v[t] = update_cluster_centers(x, c, m, u[t])

        # Stopping Criteria
        if np.linalg.norm(v[t] - v[t - 1]) < e:
            break

    return v[t], u[t], u[0], None, t, None
