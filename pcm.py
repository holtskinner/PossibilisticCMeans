import numpy as np
import skfuzzy as fuzz


def pcm(x, c, m=2, e=0.01, max_iterations=100, v=None):
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

    `v` array-like, optional  
        Initial cluster centers

    ### Returns

    `cluster_centers` 2D Array, size (S, c)

    `u` 2D Array (S, N)  
        Final partitioned matrix
    
    `u0` 2D Array (S, N)  
        Initial partition matrix
    
    `d` 2D Array (S, N)  
        Distance Matrix
    
    `p` int  
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
    if not v or not v.any():
        # Pick random values from dataset
        v = np.random.choice(x, c, True)
    else:
        # Turn the input array into a numpy array
        v = np.array(v)

    return


data = np.random.randint(100, size=5)
print(data)
pcm(x=data, c=2, m=2)
