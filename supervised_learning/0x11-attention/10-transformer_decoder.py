#!/usr/bin/env python3
"""
module containing class Decoder:
Class constructor:
    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1)
Public instance method:
    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask)
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    Class Decoder : inherits from tensorflow.keras.layers.Layer
         to create the decoder for a transformer
    """

    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        """
        Class contructor
        Args:
            N: number of blocks in the encoder
            dm: dimensionality of the model
            h: number of heads
            hidden: number of hidden units in the fully connected layer
            target_vocab: size of the target vocabulary
            max_seq_len: maximum sequence length possible
            drop_rate: dropout rate
        Public instance attributes:
            N: number of blocks in the encoder
            dm: dimensionality of the model
            embedding: embedding layer for the targets
            positional_encoding: numpy.ndarray of shape (max_seq_len, dm)
                containing the positional encodings
            blocks: list of length N containing all of the DecoderBlock's
            dropout: dropout layer, to be applied to the positional encodings
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(
            input_dim=target_vocab, output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = []
        for _ in range(N):
            self.blocks.append(DecoderBlock(dm, h, hidden, drop_rate))
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Public instance method that create decoder for a transformer
        Args:
            x: tensor of shape (batch, target_seq_len, dm)
                containing the input to the decoder
            encoder_output: tensor of shape (batch, input_seq_len, dm)
                containing the output of the encoder
            training: boolean to determine if the model is training
            look_ahead_mask: mask to be applied to the first multi head
                attention layer
            padding_mask: mask to be applied to the second multi head
                attention layer
        Returns: tensor of shape (batch, target_seq_len, dm)
            containing the decoder output
        """
        target_seq_len = x.shape[1]

        embedding_vector = self.embedding(x)
        embedding_vector *= tf.math.sqrt(tf.cast(self.dm, dtype=tf.float32))
        embedding_vector += self.positional_encoding[:target_seq_len, :]

        decoder_output = self.dropout(embedding_vector,
                                      training=training)

        for block in range(self.N):
            decoder_output = self.blocks[block](decoder_output,
                                                encoder_output,
                                                training,
                                                look_ahead_mask,
                                                padding_mask)

        return decoder_output