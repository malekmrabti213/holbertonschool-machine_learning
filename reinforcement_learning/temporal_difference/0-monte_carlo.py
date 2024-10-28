#!/usr/bin/env python3
"""
Task 0
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """monte carlo"""
    # evaluate episodes
    for i in range(episodes):
        s, _ = env.reset()
        episode = []
        for j in range(max_steps):
            action = policy(s)
            s_new, reward, terminated, truncated, _ = env.step(action)
            episode.append([s, action, reward, s_new])
            if terminated or truncated:
                break
            s = s_new
        episode = np.array(episode, dtype=int)
        G = 0
        for j, step in enumerate(episode[::-1]):
            s, action, reward, _ = step
            G = gamma * G + reward
            if s not in episode[:i, 0]:
                V[s] = V[s] + alpha * (G - V[s])
    return V
