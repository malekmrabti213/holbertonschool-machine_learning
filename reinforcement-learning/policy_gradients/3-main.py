#!/usr/bin/env python3

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
train = __import__('train').train

np.random.seed(0)

env = gym.make('CartPole-v1', render_mode="human")

scores = train(env, 10, 0.000045, 0.98, True)

env.close()