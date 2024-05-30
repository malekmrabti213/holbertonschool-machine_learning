#!/usr/bin/env python3
"""
Task 5
"""
import tensorflow as tf
from tensorflow import keras as K

def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using Keras.

    :param X: K.Input, shape (m, 28, 28, 1) containing the input images for the network
    m is the number of images

    The model should consist of the following layers in order:
    - Convolutional layer with 6 kernels of shape 5x5 with same padding
    - Max pooling layer with kernels of shape 2x2 with 2x2 strides
    - Convolutional layer with 16 kernels of shape 5x5 with valid padding
    - Max pooling layer with kernels of shape 2x2 with 2x2 strides
    - Fully connected layer with 120 nodes
    - Fully connected layer with 84 nodes
    - Fully connected softmax output layer with 10 nodes

    All layers requiring initialization should initialize their kernels with the he_normal initialization method,
    with the seed set to zero for each initializer to ensure reproducibility.
    All hidden layers requiring activation should use the relu activation function.

    :return: A K.Model compiled to use Adam optimization (with default hyperparameters) and accuracy metrics
    """
    
    he_normal = K.initializers.HeNormal(seed=0)
    
    A1 = K.layers.Conv2D(6, 5, activation='relu', kernel_initializer=he_normal, padding='same')(X)
    A2 = K.layers.MaxPooling2D()(A1)
    A3 = K.layers.Conv2D(16, 5, activation='relu', kernel_initializer=he_normal)(A2)
    A4 = K.layers.MaxPooling2D()(A3)
    A5 = K.layers.Flatten()(A4)
    A6 = K.layers.Dense(120, activation='relu', kernel_initializer=he_normal)(A5)
    A7 = K.layers.Dense(84, activation='relu', kernel_initializer=he_normal)(A6)
    Y = K.layers.Dense(10, activation='softmax', kernel_initializer=he_normal)(A7)
    
    model = K.Model(inputs=X, outputs=Y)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model
