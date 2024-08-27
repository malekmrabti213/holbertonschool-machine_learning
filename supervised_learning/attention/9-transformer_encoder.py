#!/usr/bin/env python3
"""
module containing class Encoder:
Class constructor:
    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1)
Public instance method:
    def call(self, x, training, mask)
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    Class Encoder : inherits from tensorflow.keras.layers.Layer
         to create the encoder for a transformer
    """

    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """
        Class contructor
        Args:
            N: number of blocks in the encoder
            dm: dimensionality of the model
            h: number of heads
            hidden: number of hidden units in the fully connected layer
            input_vocab: size of the input vocabulary
            max_seq_len: maximum sequence length possible
            drop_rate: dropout rate
        Public instance attributes:
            N: number of blocks in the encoder
            dm: dimensionality of the model
            embedding: embedding layer for the inputs
            positional_encoding: numpy.ndarray of shape (max_seq_len, dm)
                containing the positional encodings
            blocks: list of length N containing all of the EncoderBlock's
            dropout: dropout layer, to be applied to the positional encodings
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(
            input_dim=input_vocab, output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = []
        for _ in range(N):
            self.blocks.append(EncoderBlock(dm, h, hidden, drop_rate))
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training, mask):
        """
        Public instance method that create encoder for a transformer
        Args:
            x: tensor of shape (batch, input_seq_len, dm)
                containing the input to the encoder
            training: boolean to determine if the model is training
            mask: mask to be applied for multi head attention
        Returns: tensor of shape (batch, input_seq_len, dm)
            containing the encoder output
        """
        input_seq_len = x.shape[1]

        embedding_vector = self.embedding(x)
        embedding_vector *= tf.math.sqrt(tf.cast(self.dm, dtype=tf.float32))
        embedding_vector += self.positional_encoding[:input_seq_len, :]

        encoder_output = self.dropout(embedding_vector,
                                      training=training)

        for block in range(self.N):
            encoder_output = self.blocks[block](encoder_output, training, mask)

        return encoder_output
