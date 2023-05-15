#!/usr/bin/env python3

import tensorflow.keras as K

def train_model(network, data, labels, batch_size, epochs, validation_data=None, verbose=True, shuffle=False):
    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       validation_data=validation_data, verbose=verbose, shuffle=shuffle)