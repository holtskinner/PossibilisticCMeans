import numpy as np
import sklearn as sk
import skfuzzy as fuzzy
from plot import plot
from pcm import pcm

num_datapoints = 1000
num_features = 2

x = np.random.uniform(low=0, high=1000, size=(num_datapoints, num_features))
y = np.random.uniform(low=2000, high=3000, size=(num_datapoints, num_features))

xy = np.concatenate((x, y), axis=0)
np.random.shuffle(xy)

plot(xy)

y = pcm(x=xy, c=2, m=2)
