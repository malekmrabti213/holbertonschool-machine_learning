#!/usr/bin/env python3
"""
    NLP --WE --Task2 --Train Word2Vec
"""

from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates and trains a gensim word2vec model.

    :param sentences: list of sentences to be trained on
    :type sentences: list of list of str
    :param vector_size: dimensionality of the embedding layer
    :type vector_size: int
    :param min_count: minimum number of occurrences of a word for use in training
    :type min_count: int
    :param window: maximum distance between the current and predicted word within a sentence
    :type window: int
    :param negative: size of negative sampling
    :type negative: int
    :param cbow: boolean to determine training type; True for CBOW, False for Skip-gram
    :type cbow: bool
    :param epochs: number of iterations to train over
    :type epochs: int
    :param seed: seed for the random number generator
    :type seed: int
    :param workers: number of worker threads to train the model
    :type workers: int

    :return: Trained Word2Vec model
    :rtype: gensim.models.Word2Vec
    """
    if cbow is True:
        sg = 0
    else:
        sg = 1
    model = Word2Vec(sentences=sentences,
                     vector_size=vector_size,
                     window=window,
                     min_count=min_count,
                     negative=negative,
                     seed=seed,
                     workers=workers,
                     epochs=epochs,
                     sg=cbow)

    return model
