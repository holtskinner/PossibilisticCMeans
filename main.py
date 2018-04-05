import numpy as np
import sklearn as sk
import sklearn.datasets as ds
import skfuzzy as fuzz
from plot import plot
import cmeans

num_samples = 500
num_features = 2
c = 4
fuzzifier = 1.2
error = 0.0001
maxiter = 100

# np.random.seed(100)

x = ds.make_blobs(num_samples, num_features, c)[0]

# np.random.shuffle(x)

x = x.T

# v, v0, u, u0, d, t = cmeans.fcm(x, c, fuzzifier, error, maxiter)

# plot(x, v, u, c)

v, v0, u, u0, d, t = cmeans.hcm(x, c, error, maxiter)

print(t)

plot(x, v, u, c)

'''
iris = ds.load_iris()

labels = iris.target_names
target = iris.target
iris = np.array(iris.data).T

c = 3
v, v0, u, u0, d, t = cmeans.fcm(iris, c, fuzzifier, error, maxiter)

print(v0)
plot(iris, v, u, c, labels)
'''
