#!/usr/bin/env python3

import tensorflow as tf

def create_layer(prev, n, activation):
    """create a of a neural network layer and calculate its output """
    layer = tf.layers.Dense(n, activation=activation, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"), name='layer')
    return layer(prev)