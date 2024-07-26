#!/usr/bin/env python3

import numpy as np
pca = __import__('1-pca').pca

import_data = np.load('data.npz')

np.random.seed(1)
a = import_data['a']
b = import_data['b']
c = import_data['c']
d = 2 * a
e = -5 * b
f = 10 * c

X = np.array([a, b, c, d, e, f]).T
m = X.shape[0]

T = pca(X, 100)
print(T)
print(T.shape)