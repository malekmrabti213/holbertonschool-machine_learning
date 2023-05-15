#!/usr/bin/env python3

import tensorflow.keras as K

def save_config(network, filename):
    with open(filename, 'w') as f:
        f.write(network.to_json())

def load_config(filename):
    with open(filename, 'r') as f:
        config_json = f.read()
        return K.models.model_from_json(config_json)