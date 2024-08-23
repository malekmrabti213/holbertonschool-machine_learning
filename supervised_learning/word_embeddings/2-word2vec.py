#!/usr/bin/env python3

"""
NLP --WE --Task2 --Train Word2Vec
"""

from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """

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
