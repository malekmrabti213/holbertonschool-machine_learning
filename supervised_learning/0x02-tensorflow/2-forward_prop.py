#!/usr/bin/env python3
"""
contains the function forward_prop
"""

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes, activations):
    """create the graph for the forward propagation of a neural network"""
    prev = x
    for i, n in enumerate(layer_sizes):
        layer_output = create_layer(prev, n, activations[i])
        prev = layer_output
    
    return prev