import numpy as np
import sklearn as sk
import sklearn.datasets as ds
import skfuzzy as fuzz
from plot import plot
import cmeans

num_samples = 100000
num_features = 3
c = 3
fuzzifier = 2
error = 0.001
maxiter = 1000

x = ds.make_blobs(num_samples, num_features, c)[0].T

np.random.shuffle(x)

v, u, u0, d, t = cmeans.fcm(x, c, fuzzifier, error,
                            maxiter, metric="Euclidean")

v, u, u0, d, t = cmeans.pcm(
    x, c, fuzzifier, error, maxiter)

plot(x, v, u, c)


iris = ds.load_iris()

labels = iris.target_names
target = iris.target
iris = np.array(iris.data)

iris = iris.T

v, u, u0, d, t = cmeans.fcm(
    iris, c, fuzzifier, error, maxiter, metric="Euclidean")

v, u, u0, d, t = cmeans.pcm(
    iris, c, fuzzifier, error, maxiter, v0=v, u0=u, d=d)

plot(iris, v, u, c, labels)
