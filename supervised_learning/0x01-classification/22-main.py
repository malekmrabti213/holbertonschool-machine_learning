
#!/usr/bin/env python3

import numpy as np
Deep = __import__('22-deep_neural_network').DeepNeuralNetwork

np.random.seed(22)
nx, m = np.random.randint(100, 1000, 2).tolist()
l = np.random.randint(3, 10)
sizes = np.random.randint(5, 20, l - 1).tolist()
sizes.append(1)
d = Deep(nx, sizes)

X = np.random.randn(nx, m)
Y = np.random.randint(0, 2, (1, m))
A, cost = d.train(X, Y)
print(A)
print(np.round(cost, 10))
for k, v in sorted(d.weights.items()):
    print(k, np.round(v, 10))
for k, v in sorted(d.cache.items()):
    print(k, np.round(v, 10))
try:
    d.weights = 10
    print('Fail: private attribute weights is overwritten as a public attribute')
except:
    pass
try:
    d.cache = 10
    print('Fail: private attribute cache is overwritten as a public attribute')
except:
    pass