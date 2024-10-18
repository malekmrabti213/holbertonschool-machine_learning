#!/usr/bin/env python3
"""
module containing function train
"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Function that performs Q-learning
    
    Args:
    env: The environment object.
    Q: The Q-table to update.
    episodes: The total number of episodes to train.
    max_steps: Max steps per episode.
    alpha: Learning rate.
    gamma: Discount factor for future rewards.
    epsilon: Initial epsilon for the epsilon-greedy policy.
    min_epsilon: Minimum value of epsilon after decay.
    epsilon_decay: The rate at which epsilon decays.

    Returns:
    Q: Updated Q-table.
    rewards: List of rewards per episode.
    """
    rewards = []
    initial_epsilon = epsilon

    for episode in range(episodes):
        # Reset the environment to get the initial state
        state = env.reset()[0]
        terminated = False
        truncated = False
        done = terminated or truncated
        total_rewards = 0

        # Iterate through the steps within an episode
        for _ in range(max_steps):
            # Select action using epsilon-greedy policy
            action = epsilon_greedy(Q, state, epsilon)

            # Execute action in the environment
            new_state, reward, terminated, truncated, _ = env.step(action)
            # Penalize for terminating the episode with zero reward
            if done and reward == 0:
                reward = -1

            # Update Q-value using the Q-learning update rule
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state, :]) - Q[state, action])

            total_rewards += reward
            state = new_state

            # Break if the episode has terminated or truncated
            if done :
                break

        # Decay epsilon to balance exploration and exploitation
        epsilon = max(min_epsilon, initial_epsilon - epsilon_decay * (episode + 1))

        # Store total reward for this episode
        rewards.append(total_rewards)

    return Q, rewards
