#!/usr/bin/env python3
"""
2-sarsa_lambtha
"""

import numpy as np

def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """SARSA(λ) with eligibility trace"""

    # Get the number of states and actions from the Q matrix
    states, actions = Q.shape

    # Store the initial maximum epsilon value
    max_epsilon = epsilon

    def epsilon_greedy(epsilon, Qs):
        # Epsilon-greedy policy to choose actions
        p = np.random.uniform()
        if p > epsilon:
            return np.argmax(Qs)  # Exploit (choose the action with the highest Q-value)
        else:
            return np.random.randint(0, actions)  # Explore (choose a random action)

    # Iterate over episodes
    for i in range(episodes):
        E = np.zeros((states, actions))  # Initialize the eligibility trace
        s_prev = env.reset()  # Reset the environment and get the initial state
        action_prev = epsilon_greedy(epsilon, Q[s_prev])  # Choose the initial action using epsilon-greedy

        # Iterate within each episode
        for j in range(max_steps):
            s, reward, done, info = env.step(action_prev)  # Take an action and observe the next state and reward
            action = epsilon_greedy(epsilon, Q[s])  # Choose the next action using epsilon-greedy
            delta = reward + (gamma * Q[s, action]) - Q[s_prev, action_prev]  # Calculate the temporal difference error
            E[s_prev, action_prev] += 1  # Update the eligibility trace
            E = E * gamma * lambtha  # Decay the eligibility trace
            Q = Q + (alpha * delta * E)  # Update the Q-values using the SARSA(λ) update rule

            if done:
                break

            s_prev = s  # Update the previous state
            action_prev = action  # Update the previous action

        # Decay epsilon for exploration-exploitation trade-off
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay * i)

    return Q  # Return the updated Q-values
