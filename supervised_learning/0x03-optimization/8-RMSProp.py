#!/usr/bin/env python3


import tensorflow as tf

def create_RMSProp_op(loss, alpha, beta2, epsilon):
    optimizer = tf.train.RMSPropOptimizer(alpha, beta2, epsilon=epsilon)
    return optimizer.minimize(loss)