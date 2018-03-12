import numpy as np
import sklearn as sk
import sklearn.datasets as ds
import skfuzzy as fuzz
from plot import plot
import cmeans

num_samples = 100000
num_features = 2
c = 3
fuzzifier = 2
error = 0.001
maxiter = 100

# x = ds.make_blobs(num_samples, num_features, c)[0].T

iris = ds.load_iris()

x = np.array(iris.data)
np.random.shuffle(x)

x = x.T

labels = iris.target_names

v, u, u0, d, t, f = cmeans.fcm(x, c, fuzzifier, error, maxiter)

# v, u, u0, d, _, t, f = fuzz.cmeans(x, c, fuzzifier, error, maxiter)

v, u, u0, d, t, f = cmeans.pcm(
    x, c, fuzzifier, error, maxiter, v0=v, u0=u, d=d)

plot(x, v, u, c, labels)
