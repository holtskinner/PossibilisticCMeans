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


def update_cluster_centers():
    return


def pcm(x, c, m=2, e=0.01, max_iterations=100, v0=None):
    """
    Possibilistic C-Means Algorithm

    ### Parameters

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

    if not x.any():
        print("Error: Data cannot be None")
        return

    if not c or c <= 0:
        print("Error: Number of clusters must be at least 1")

    if not m or m <= 1:
        print("Error: Fuzzifier must be greater than 1")
        return

    # Initialize the cluster centers
    # If the user doesn't provide their own starting points,
    if not v0 or not v0.any():
        # Pick random values from dataset
        v0 = np.random.choice(x, c, True)
    else:
        # Turn the input array into a numpy array
        v0 = np.array(v0)

    # List of all cluster centers (Bookkeeping)
    v = np.array([v0])

    # Partition Matrix
    u = np.array([[[]]])

    n = eta()

    # Number of Iterations
    t = 0

    while t <= max_iterations:

        # Cycle through x's
        for k in range(len(x)):
            for i in range(len(v[t])):
                u[t, i, k] = criterion_function(x[k], v[i], n, m)

        t += 1
        # Stopping Criteria
        if np.linalg.norm(v[t] - v[t - 1]) < e:
            break

    return v


data = np.random.randint(100, size=5)
print(data)
pcm(x=data, c=2, m=2)
