#!/usr/bin/env python3

import tensorflow as tf

def calculate_loss(y, y_pred):
    """calculate the loss of a prediction"""
    return tf.losses.softmax_cross_entropy(y, y_pred)