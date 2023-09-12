#!/usr/bin/env python3

def gensim_to_keras(model):
    return model.wv.get_keras_embedding(True)