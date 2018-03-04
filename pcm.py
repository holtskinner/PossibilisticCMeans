import numpy as np


def pcm(x, c, m=2, e=0.01, max_iterations=100, v=None):

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

    print(v)

    return 0


data = np.random.randint(100, size=5)
print(data)
pcm(data, 2, 2)
