#!/usr/bin/env python3
"""
module containing class EncoderBlock:
Class constructor: def __init__(self, dm, h, hidden, drop_rate=0.1)
Public instance method:
    def call(self, x, training, mask=None)
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    Class EncoderBlock : inherits from tensorflow.keras.layers.Layer
         to create an encoder block for a transformer
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class contructor
        Args:
            dm: dimensionality of the model
            h: number of heads
            hidden: number of hidden units in the fully connected layer
            drop_rate: dropout rate
        Public instance attributes:
            mha: MultiHeadAttention layer
            dense_hidden: hidden dense layer with hidden units
                and relu activation
            dense_output: output dense layer with dm units
            layernorm1: first layer norm layer, with epsilon=1e-6
            layernorm2: second layer norm layer, with epsilon=1e-6
            dropout1: first dropout layer
            dropout2: second dropout layer
        """
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training, mask=None):
        """
        Public instance method that create an encoder block for a transformer
        Args:
            x: tensor of shape (batch, input_seq_len, dm)
                containing the input to the encoder block
            training: boolean to determine if the model is training
            mask: mask to be applied for multi head attention
        Returns: tensor of shape (batch, input_seq_len, dm)
            containing the block's output
        """
        multihead_output, _ = self.mha(x, x, x, mask)
        multihead_output = self.dropout1(multihead_output,
                                         training=training)

        addnorm_output = self.layernorm1(x + multihead_output)

        feedforward_output = self.dense_hidden(addnorm_output)
        feedforward_output = self.dense_output(feedforward_output)
        feedforward_output = self.dropout2(feedforward_output,
                                           training=training)

        addnorm_output = self.layernorm2(addnorm_output +
                                         feedforward_output)

        return addnorm_output
