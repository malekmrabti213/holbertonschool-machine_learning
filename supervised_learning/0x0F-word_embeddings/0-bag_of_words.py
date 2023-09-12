#!/usr/bin/env python3

from sklearn.feature_extraction.text import CountVectorizer

def bag_of_words(sentences, vocab=None):
    vectorizer = CountVectorizer(vocabulary=vocab)
    E = vectorizer.fit_transform(sentences)
    F = vectorizer.get_feature_names()
    return E.toarray(), F