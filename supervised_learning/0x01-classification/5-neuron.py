#!/usr/bin/env python3
"""Neuron Module 5"""
import numpy as np


class Neuron():
    """Neuron Class"""
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """W getter method"""
        return self.__W

    @property
    def b(self):
        """b getter method"""
        return self.__b

    @property
    def A(self):
        """A getter method"""
        return self.__A

    def forward_prop(self, X):
        """Forward Propagation"""
        Z = np.dot(self.W, X) + self.b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Cost Function"""
        m = Y.shape[1]
        loss = - (Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = np.sum(loss) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluate"""
        A = self.forward_prop(X)
        P = np.where(A > 0.5, 1, 0)
        C = self.cost(Y, A)
        return P, C

    def gradient_descent(self, X, Y, A, alpha=0.5):
        """Gradient Descent"""
        m = Y.shape[1]
        dZ = A - Y
        db = 1 / m * np.sum(dZ)
        dW = 1 / m * np.dot(dZ, X.T)
        self.__W -= alpha * dW
        self.__b -= alpha * db