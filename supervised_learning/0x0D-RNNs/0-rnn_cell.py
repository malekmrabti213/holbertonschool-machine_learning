#!/usr/bin/env python3

import numpy as np

class RNNCell:
    def __init__(self, i, h, o):
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))
    
    def forward(self, h_prev, x_t):
        cat = np.concatenate([h_prev, x_t], axis=1)
        h = np.matmul(cat, self.Wh)+ self.bh
        h = np.tanh(h)
        z = np.matmul(h, self.Wy) + self.by
        exp_z = np.exp(z)
        y = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return h, y