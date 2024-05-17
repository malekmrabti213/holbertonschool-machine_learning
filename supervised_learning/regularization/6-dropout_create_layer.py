#!/usr/bin/env python3

import tensorflow as tf

def dropout_create_layer(prev, n, activation, keep_prob,training=True):
    """
    Function that creates a layer of a neural network using dropout regularization

    :param prev: tensor, output of the previous layer
    :param n: number of nodes in the new layer
    :param activation: activation function for the new layer
    :param keep_prob: probability that a node will be kept
    :param training: boolean, whether the model is in training mode

    :return: output tensor of the new layer
    """

    # Initialize layer weights using He initialization
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')

    # Create dense layer
    dense_layer = tf.keras.layers.Dense(units=n, activation=activation, kernel_initializer=init)

    # Apply dense layer to previous layer
    output = dense_layer(prev)

    # Apply dropout regularization
    dropout_layer = tf.keras.layers.Dropout(rate=1-keep_prob)
    output = dropout_layer(output, training=training)

    return output