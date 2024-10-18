#!/usr/bin/env python3
"""
3-q_learning
"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Function that performs Q-learning
    """
    rewards = []
    initial_epsilon = epsilon

    for episode in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        done = terminated or truncated  
        total_rewards = 0

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, terminated, truncated, _ = env.step(action)

            if done and reward == 0:
                reward = -1

            Q[state, action] = Q[state, action] + alpha * (reward + gamma *\
                np.max(Q[new_state, :]) - Q[state, action])

            total_rewards += reward
            state = new_state

            if done == True:
                break

        epsilon = (min_epsilon + (initial_epsilon - min_epsilon) *
                    np.exp(-epsilon_decay * episode))

        # epsilon = max(min_epsilon, initial_epsilon - epsilon_decay * (episode + 1))
        rewards.append(total_rewards)

    return Q, rewards
