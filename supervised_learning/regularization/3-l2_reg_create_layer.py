#!/usr/bin/env python3
"""
Create layer with L2 regularization
"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Function that creates a TensorFlow layer including L2 regularization

    :param prev: tensor, output of previous layer
    :param n: number of nodes in the new layer
    :param activation: activation function used on the layer
    :param lambtha: L2 regularization parameter

    :return: output of the new layer
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                        mode='fan_avg')

    # Create Dense layer with parameters
    new_layer = tf.keras.layers.Dense(n,
                                       activation=activation,
                                       kernel_initializer=initializer,
                                       kernel_regularizer=tf.keras.regularizers.L2(lambtha))

    # Apply layer to input
    output = new_layer(prev)

    return output
