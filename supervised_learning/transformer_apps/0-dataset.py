#!/usr/bin/env python3
"""
"""
import tensorflow_datasets as tfds
import transformers


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
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

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

        # Load a pre-trained tokenizer and adapt it
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', use_fast=True, clean_up_tokenization_spaces=True)
        tokenizer_en = transformers.AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True, clean_up_tokenization_spaces=True )

        # Train the tokenizer on your dataset
        tp=tokenizer_pt.train_new_from_iterator(pt_sentences, vocab_size=2**13)
        te=tokenizer_en.train_new_from_iterator(en_sentences, vocab_size=2**13)

        self.tokenizer_pt = tp
        self.tokenizer_en = te

        return self.tokenizer_pt, self.tokenizer_en
