#!/usr/bin/env python3

import tensorflow as tf

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        s_prev_exp = tf.expand_dims(s_prev, 1)
        score = self.V(tf.nn.tanh(self.W(s_prev_exp) + self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = weights * hidden_states
        context = tf.reduce_sum(context, axis=1)
        return context, weights