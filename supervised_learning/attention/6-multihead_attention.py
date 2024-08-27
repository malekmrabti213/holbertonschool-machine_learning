#!/usr/bin/env python3
"""
module containing class MultiHeadAttention:
Class constructor: def __init__(self, dm, h)
Public instance method:
    def call(self, Q, K, V, mask)

"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Class MultiHeadAttention : inherits from tensorflow.keras.layers.Layer
         to perform multi head attention
    """

    def __init__(self, dm, h):
        """
        Class contructor
        Args:
            dm: integer representing the dimensionality of the model
            h: integer representing the number of heads
        Public instance attributes:
            h: number of heads
            dm: dimensionality of the model
            depth: depth of each attention head
            Wq: Dense layer with dm units, used to generate the query matrix
            Wk: Dense layer with dm units, used to generate the key matrix
            Wv: Dense layer with dm units, used to generate the value matrix
            linear: Dense layer with dm units, used to generate the attention
                output
        """
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = int(dm / h)
        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)
        self.linear = tf.keras.layers.Dense(units=dm)

    def reshape_tensor(self, x, batch_size):
        """Rearrange shape of tensor to be
        (batch_size, heads, seq_lenght, -1)"""
        x = tf.reshape(x, shape=(batch_size, -1, self.h, self.depth))
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return x

    def call(self, Q, K, V, mask):
        """
        Public instance method that perform multi head attention
        Args:
            Q: tensor of shape (batch, seq_len_q, dk) containing the input to
                generate the query matrix
            K: tensor of shape (batch, seq_len_v, dk) containing the input to
                generate the key matrix
            V: tensor of shape (batch, seq_len_v, dv) containing the input to
                generate the value matrix
            mask: always None
        Returns: output, weights
            output: tensor with its last two dimensions as (..., seq_len_q, dm)
                containing the scaled dot product attention
            weights: tensor with its last three dimensions as
                (..., h, seq_len_q, seq_len_v) containing the attention weights
        """
        batch = tf.shape(Q)[0]

        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = self.reshape_tensor(Q, batch)
        K = self.reshape_tensor(K, batch)
        V = self.reshape_tensor(V, batch)

        output, weights = sdp_attention(Q, K, V, mask)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(output, (batch, -1, self.dm))

        output = self.linear(attention_output)

        return output, weights
