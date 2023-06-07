#!/usr/bin/env python3

# import tensorflow.keras as K

import tensorflow as tf
from tensorflow import keras as K

identity_block = __import__('2-identity_block').identity_block
X1 = K.Input(shape=(56, 56, 256))
Y1 = identity_block(X1, [64, 64, 256])
model1 = K.models.Model(inputs=X1, outputs=Y1)
model1.summary()
X2 = K.Input(shape=(28, 28, 512))
Y2 = identity_block(X2, [128, 128, 512])
model2 = K.models.Model(inputs=X2, outputs=Y2)
model2.summary()