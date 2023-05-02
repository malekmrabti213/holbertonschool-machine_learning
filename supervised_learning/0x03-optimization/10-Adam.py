#!/usr/bin/env python3

import tensorflow as tf

def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    optimizer = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return optimizer.minimize(loss)