#!/usr/bin/env python3
""" task3 """

import tensorflow as tf
import transformers
import tensorflow_datasets as tfds


class Dataset:
    """
    """
    def __init__(self):
        """
        """
        # Load the dataset
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True
        )

        # Initialize tokenizers
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

        # Update data_train and data_valid by tokenizing the examples
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        """
        # Create lists to hold sentences for training tokenizers
        pt_sentences = []
        en_sentences = []

        # Collect sentences from the dataset
        for pt, en in data:
            pt_sentences.append(pt.numpy().decode('utf-8'))
            en_sentences.append(en.numpy().decode('utf-8'))

        # Load pre-trained tokenizers and train them
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased', use_fast=True,
            clean_up_tokenization_spaces=True)
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased', use_fast=True,
            clean_up_tokenization_spaces=True)

        # Train the tokenizer on your dataset
        tp = tokenizer_pt.train_new_from_iterator(pt_sentences,
                                                  vocab_size=2**13)
        te = tokenizer_en.train_new_from_iterator(en_sentences,
                                                  vocab_size=2**13)

        self.tokenizer_pt = tp
        self.tokenizer_en = te

        return self.tokenizer_pt, self.tokenizer_en

    def encode(self, pt, en):
        """
        """
        # Define special tokens for Portuguese and English
        pt_start_token_id = len(self.tokenizer_pt)
        pt_end_token_id = len(self.tokenizer_pt) + 1

        en_start_token_id = len(self.tokenizer_en)
        en_end_token_id = len(self.tokenizer_en) + 1

        # Convert tensors to strings
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        # Tokenize sentences (without adding special tokens)
        pt_tokens = self.tokenizer_pt.encode(pt_text,
                                             add_special_tokens=False)
        en_tokens = self.tokenizer_en.encode(en_text,
                                             add_special_tokens=False)

        # Add start and end tokens for each sentence
        pt_tokens = [pt_start_token_id] + pt_tokens + [pt_end_token_id]
        en_tokens = [en_start_token_id] + en_tokens + [en_end_token_id]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        """
        # Use tf.py_function to wrap the encode method
        pt_tokens, en_tokens = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        # Set the shape of the tensors
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
