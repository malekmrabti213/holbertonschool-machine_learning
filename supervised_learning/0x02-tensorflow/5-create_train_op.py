#!/usr/bin/env python3

import tensorflow as tf

def create_train_op(loss, alpha):
    """create the training operation for a neural network"""
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train_op = optimizer.minimize(loss)
    return train_op