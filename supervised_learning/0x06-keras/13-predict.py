#!/usr/bin/env python3

import tensorflow.keras as K

def predict(network, data, verbose=False):
    return network.predict(data, verbose=verbose)