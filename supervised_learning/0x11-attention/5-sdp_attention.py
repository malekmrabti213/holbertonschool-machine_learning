#!/usr/bin/env python3

import tensorflow as tf

def sdp_attention(Q, K, V, mask=None):
    QK = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    attention = QK / tf.sqrt(dk)
    if mask is not None:
        attention += mask * -1e9
    weights = tf.nn.softmax(attention, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights