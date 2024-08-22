#!/usr/bin/env python3

import gensim

def fasttext_model(sentences, size=100, min_count=5, window=5, negative=5, cbow=True,
             iterations=5, seed=0, workers=1):
    """
    Create and train a FastText model using gensim.

    Args:
    sentences (list of list of str): List of sentences to be trained on.
    size (int): Dimensionality of the embedding layer.
    min_count (int): Minimum number of occurrences of a word for use in training.
    window (int): Maximum distance between the current and predicted word within a sentence.
    negative (int): Size of negative sampling.
    cbow (bool): True for CBOW, False for Skip-gram.
    iterations (int): Number of iterations to train over.
    seed (int): Seed for the random number generator.
    workers (int): Number of worker threads to train the model.

    Returns:
    gensim.models.FastText: Trained FastText model.
    """
        
    if not cbow:
        sg = 1
    else:
        sg = 0
    model = gensim.models.FastText(sentences=sentences, size=size, min_count=min_count, window=window, negative=negative, sg=sg, seed=seed, iter=iterations, workers=workers)
    return model