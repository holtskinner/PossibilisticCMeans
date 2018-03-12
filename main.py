import numpy as np
import sklearn as sk
from sklearn.datasets import make_blobs
import skfuzzy as fuzz
from plot import plot
import cmeans

num_samples = 100000
num_features = 2
c = 6
fuzzifier = 2
error = 0.001
maxiter = 100

x = make_blobs(num_samples, num_features, c)[0].T

v, u, u0, d, t, f = cmeans.fcm(x, c, fuzzifier, error, maxiter)

plot(x, v, u, c)

v, u, u0, d, t, f = cmeans.pcm(x, c, fuzzifier, error, maxiter, v0=v)

plot(x, v, u, c)
