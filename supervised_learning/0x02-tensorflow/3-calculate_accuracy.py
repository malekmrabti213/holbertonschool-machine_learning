#!/usr/bin/env python3

import tensorflow as tf

def calculate_accuracy(y, y_pred):
    """calculate the accuracy of a prediction"""
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy