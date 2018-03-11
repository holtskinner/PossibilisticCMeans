import numpy as np
import sklearn as sk
import skfuzzy as fuzz
from plot import plot
from pcm import pcm

num_datapoints = 1000
num_features = 2

# Define three cluster centers
centers = np.array([[4, 2], [1, 7], [5, 6]])

# Define three cluster sigmas in x and y, respectively
sigmas = np.array([[0.8, 0.3], [0.3, 0.5], [1.1, 0.7]])

# Generate test data
np.random.seed(42)  # Set seed for reproducibility
x = np.zeros(1)
y = np.zeros(1)

for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
    x = np.hstack((x,
                   np.random.standard_normal(num_datapoints) * xsigma + xmu))
    y = np.hstack((y,
                   np.random.standard_normal(num_datapoints) * ysigma + ymu))

xy = np.vstack((x, y))

# v, u, u0, d, _, t, f = fuzz.cmeans(
    # data=xy, c=3, error=0.001, m=2, maxiter=1000)

# print(v)

v, u, u0, d, t, f = pcm(x=xy, c=3, m=2, e=0.001, max_iterations=1000, v0=centers)

plot(xy, v, u, 3)
# print(v)