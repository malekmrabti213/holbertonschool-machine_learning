#!/usr/bin/env python3

import tensorflow.keras as K

def save_weights(network, filename, save_format='h5'):
    network.save_weights(filename, save_format=save_format)

def load_weights(network, filename):
    network.load_weights(filename)