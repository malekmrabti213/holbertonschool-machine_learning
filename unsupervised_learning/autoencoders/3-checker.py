#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

autoencoder = __import__('3-variational').autoencoder
# # MAIN IN TASK
# (x_train, _), (x_test, _) = mnist.load_data()
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((-1, 784))
# x_test = x_test.reshape((-1, 784))
# np.random.seed(0)
# tf.set_random_seed(0)
# encoder, decoder, auto = autoencoder(784, [512], 2)
# encoder.summary()
# decoder.summary()
# auto.summary()
# auto.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True,
#                 validation_data=(x_test, x_test))
# encoded, mu, log_sig = encoder.predict(x_test[:10])
# print(mu)
# print(np.exp(log_sig / 2))
# reconstructed = decoder.predict(encoded).reshape((-1, 28, 28))
# x_test = x_test.reshape((-1, 28, 28))

# for i in range(10):
#     ax = plt.subplot(2, 10, i + 1)
#     ax.axis('off')
#     plt.imshow(x_test[i])
#     ax = plt.subplot(2, 10, i + 11)
#     ax.axis('off')
#     plt.imshow(reconstructed[i])
# plt.show()


# l1 = np.linspace(-3, 3, 25)
# l2 = np.linspace(-3, 3, 25)
# L = np.stack(np.meshgrid(l1, l2, sparse=False, indexing='ij'), axis=2)
# G = decoder.predict(L.reshape((-1, 2)), batch_size=125)

# for i in range(25*25):
#     ax = plt.subplot(25, 25, i + 1)
#     ax.axis('off')
#     plt.imshow(G[i].reshape((28, 28)))
# plt.show()


#MAIN 1 CHECKER
np.random.seed(0)
tf.set_random_seed(0)
encoder, decoder, auto = autoencoder(784, [512, 256], 2)
if len(auto.layers) == 3:
    print(auto.layers[0].input_shape == (None, 784))
    print(auto.layers[1] is encoder)
    print(auto.layers[2] is decoder)

# #MAIN 2 CHECKER
# np.random.seed(0)
# tf.set_random_seed(0)
# encoder, decoder, auto = autoencoder(784, [512, 256], 2)
# if len(auto.layers) == 3:
#     print(auto.layers[0].input_shape == (None, 784))
#     print(auto.layers[1] is encoder)
#     print(auto.layers[2] is decoder)
def compare(val1, val2, threshold):

    result = True

    if val1==0:
        diff = abs(val1 - val2)
    else:
        diff = abs((val1 - val2)/val1)

        if diff > threshold:
            result = False

    return result

with open('1-test', 'w+') as f:   
    x_test = np.load("MNIST.npz")["X_test"]
    x_test = x_test[:256].reshape((-1, 784))
    reference = 5.437645e+02
    eval=auto.evaluate(x_test, x_test, verbose=False)
    threshold=0.001
    result=compare(eval,reference,threshold)
    f.write(str(result)+ '\n')
    f.write(auto.optimizer.__class__.__name__ + '\n')
