#!/usr/bin/env python3

import numpy as np
pca = __import__('1-pca').pca

# X = np.loadtxt("mnist2500_X.txt")
X = np.genfromtxt("mnist2500_X.txt", dtype=float, delimiter=',', encoding="latin-1")
print('X:', X.shape)
print(X)
T = pca(X, 50)
print('T:', T.shape)
print(T)