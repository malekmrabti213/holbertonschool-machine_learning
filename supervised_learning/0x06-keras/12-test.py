#!/usr/bin/env python3

import tensorflow.keras as K

def test_model(network, data, labels, verbose=True):
    return network.evaluate(data, labels, verbose=verbose)