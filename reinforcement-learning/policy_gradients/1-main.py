#!/usr/bin/env python3
"""
Main file
"""
import gymnasium as gym
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient

env = gym.make('CartPole-v1')
np.random.seed(0)

weight = np.random.rand(4, 2)
state , _ = env.reset(seed=0)
print(weight)
print(state)
# print(type(state))


action, grad = policy_gradient(state, weight)
print(action)
print(grad)

env.close()
