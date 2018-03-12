import numpy as np
import sklearn as sk
import skfuzzy as fuzz
from plot import plot
import cmeans

num_datapoints = 1000
num_features = 2

# Define three cluster centers
centers = np.array([[10, 10], [0, 0], [5, 5]])

# Define three cluster sigmas in x and y, respectively
sigmas = np.array([[0.8, 0.3], [0.3, 0.5], [1.1, 0.7]])

# Generate test data
# np.random.seed(42)  # Set seed for reproducibility
x = np.zeros(1)
y = np.zeros(1)

for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
    x = np.hstack((x,
                   np.random.standard_normal(num_datapoints) * xsigma + xmu))
    y = np.hstack((y,
                   np.random.standard_normal(num_datapoints) * ysigma + ymu))

xy = np.vstack((x, y))

v, u, u0, d, _, t, f = fuzz.cmeans(xy, 3, 5, 0.0001, 100)

v, u, u0, d, t, f = cmeans.pcm(
    x=xy, c=3, m=2, e=0.001, max_iterations=100, metric="euclidean", v0=v)

plot(xy, v, u, 3)
