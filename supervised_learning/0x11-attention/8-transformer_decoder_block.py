#!/usr/bin/env python3
"""
module containing class DecoderBlock:
Class constructor: def __init__(self, dm, h, hidden, drop_rate=0.1)
Public instance method:
    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask)
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    Class DecoderBlock : inherits from tensorflow.keras.layers.Layer
         to create a decoder block for a transformer
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
            mha1: first MultiHeadAttention layer
            mha2: second MultiHeadAttention layer
            dense_hidden: hidden dense layer with hidden units
                and relu activation
            dense_output: output dense layer with dm units
            layernorm1: first layer norm layer, with epsilon=1e-6
            layernorm2: second layer norm layer, with epsilon=1e-6
            layernorm3: third layer norm layer, with epsilon=1e-6
            dropout1: first dropout layer
            dropout2: second dropout layer
            dropout3: third dropout layer
        """
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Public instance method that create a decoder block for a transformer
        Args:
            x: tensor of shape (batch, target_seq_len, dm)
                containing the input to the decoder block
            encoder_output: tensor of shape (batch, input_seq_len, dm)
                containing the output of the encoder
            training: boolean to determine if the model is training
            look_ahead_mask: mask to be applied to the first multi head
                attention layer
            padding_mask: mask to be applied to the second multi head
                attention layer
        Returns: tensor of shape (batch, target_seq_len, dm)
            containing the block's output
        """
        multihead_output1, _ = self.mha1(x, x, x, look_ahead_mask)
        multihead_output1 = self.dropout1(multihead_output1,
                                          training=training)

        addnorm_output1 = self.layernorm1(x + multihead_output1)

        multihead_output2, _ = self.mha2(addnorm_output1,
                                         encoder_output,
                                         encoder_output,
                                         padding_mask)
        multihead_output2 = self.dropout2(multihead_output2,
                                          training=training)

        addnorm_output2 = self.layernorm2(addnorm_output1 + multihead_output2)

        feedforward_output = self.dense_hidden(addnorm_output2)
        feedforward_output = self.dense_output(feedforward_output)
        feedforward_output = self.dropout3(feedforward_output,
                                           training=training)

        addnorm_output3 = self.layernorm3(addnorm_output2 +
                                          feedforward_output)

        return addnorm_output3