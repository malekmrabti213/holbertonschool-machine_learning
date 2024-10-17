#!/usr/bin/env python3
"""
play
"""
import numpy as np

def play(env, Q, max_steps=100):
    """
    Function that plays frozen lake game
    """
    state = env.reset()[0]
    env.render()
    for step in range(max_steps):
        action = np.argmax(Q[state])
        new_state, reward, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            return reward
        state = new_state

    env.close()
