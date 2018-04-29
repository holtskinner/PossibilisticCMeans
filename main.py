import numpy as np
import sklearn as sk
import sklearn.datasets as ds
import skfuzzy as fuzz
from plot import plot
import cmeans


def generate_data(num_samples, num_features, c, shuffle=True):

    x = ds.make_blobs(num_samples, num_features, c, shuffle=False)[0]

    x = x.T

    labels = np.zeros(num_samples)
    labels[0:100] = 0
    labels[100:200]

    j = num_samples / c

    for i in range(c):
        labels[(i * j):((i + 1) * j)] = i

    return x, labels


def verify_clusters(x, c, v, u, labels):

    ssd_actual = 0

    for i in range(c):
        # All points in class
        x1 = x[labels == i]
        # Mean of class
        m = np.mean(x1, axis=0)

        for pt in x1:
            ssd_actual += np.linalg.norm(pt - m)

    clm = np.argmax(u, axis=0)
    ssd_clusters = 0

    for i in range(c):
        # Points clustered in a class
        x2 = x[clm == i]

        for pt in x2:
            ssd_clusters += np.linalg.norm(pt - v[i])

    print(ssd_clusters / ssd_actual)


num_samples = 300000
num_features = 2
c = 3
fuzzifier = 1.2
error = 0.001
maxiter = 100

# np.random.seed(100)

x, labels = generate_data(num_samples, num_features, c, shuffle=False)

v, v0, u, u0, d, t = cmeans.fcm(x, c, fuzzifier, error, maxiter)

plot(x.T, v, u, c)

print("Blobs")
verify_clusters(x.T, c, v, u, labels)

iris = ds.load_iris()

labels = iris.target_names
target = iris.target
iris = np.array(iris.data).T

c = 3

v, v0, u, u0, d, t = cmeans.fcm(iris, c, fuzzifier, error, maxiter)
iris = iris.T

print("Iris")

verify_clusters(iris, c, v, u, target)


digits = ds.load_digits()

labels = digits.target

digits = np.array(digits.data).T

c = 10

v, v0, u, u0, d, t = cmeans.pcm(digits, c, fuzzifier, error, maxiter)

print("Digits")
verify_clusters(digits.T, c, v, u, labels)
# plot(digits.T, v, u, c)

c = 2
cancer = ds.load_breast_cancer()

labels = cancer.target

cancer = np.array(cancer.data).T
v, v0, u, u0, d, t = cmeans.fcm(cancer, c, fuzzifier, error, maxiter)

plot(cancer.T, v, u, c)

print("Cancer")
verify_clusters(cancer.T, c, v, u, labels)
