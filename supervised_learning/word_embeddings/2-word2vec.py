#!/usr/bin/env python3
"""
Train a Word2Vec model using gensim.
"""
from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates and trains a Word2Vec model using the gensim library.

    Args:
        sentences (list of list of str): List of tokenized sentences to be trained on.
        vector_size: Dimensionality of the word vectors. Defaults to 100.
        min_count: Ignores all words with total frequency lower than this. Defaults to 5.
        window: Maximum distance between the current and predicted word
        negative: If > 0, negative sampling will be used
        epochs: Number of iterations over the corpus. Defaults to 5.
        seed: Seed for the random number generator. Defaults to 0.
        workers: Use these many worker threads to train the model

    Returns:
        gensim.models.Word2Vec: The trained Word2Vec model.
    """
    sg = 0 if cbow else 1

    model = Word2Vec(sentences=sentences,
                     vector_size=vector_size,
                     window=window,
                     min_count=min_count,
                     negative=negative,
                     seed=seed,
                     workers=workers,
                     epochs=epochs,
                     sg=sg)

    return model
