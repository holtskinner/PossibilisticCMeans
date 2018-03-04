import numpy as np
import sklearn as sk
import skfuzzy as fuzzy
import pcm

num_datapoints = 1000
num_features = 2

data = np.random.rand(num_datapoints, num_features)

print(data)
pcm(x=data, c=2, m=2)
