#!/usr/bin/env python3
"""
    Policy Gradient
"""

import numpy as np

def policy(state, weight):
    z = state.dot(weight)
    
    # Apply numerical stability fix by subtracting max of z
    z -= np.max(z)
    
    exp = np.exp(z)
    return exp / np.sum(exp)

# Vectorized softmax Jacobian
def softmax_grad(softmax):
    # print("this is before")
    # print(softmax)
    s = softmax.reshape(-1,1)
    # print("this is after")
    # print(s)
    return np.diagflat(s) - np.dot(s, s.T)

def policy_gradient(state, weight):
    # Calculate action probabilities using the policy function
    probs = policy(state, weight)
    # print(probs)

    # Sample an action based on the calculated probabilities
    action = np.random.choice(len(probs), p=probs)

    # Compute the gradient of the chosen action's log probability
    dsoftmax = softmax_grad(probs)[action, :]
    dlog = dsoftmax / probs[action]
    # print("fix")
    # print(dlog)
    # Compute the gradient of the policy and save it with the chosen action
    # print(state)
    # print(state.T)
    state = state.reshape(-1, 1)
    # print(state)
    grad = state.dot(dlog[None, :])

    return action, grad
