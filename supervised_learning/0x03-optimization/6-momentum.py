#!/usr/bin/env python3

import tensorflow as tf

def create_momentum_op(loss, alpha, beta1):
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    return optimizer.minimize(loss)