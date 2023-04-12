#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
lib_train = np.load('C:/Users/CAMPUSNA/Downloads/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']

print(X_3D)
print(Y)
print(np.shape(X_3D))
X = X_3D.reshape((X_3D.shape[0], -1)).T
print(np.shape(X))

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_3D[i])
    plt.title(Y[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()