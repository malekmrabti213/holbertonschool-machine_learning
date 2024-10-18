#!/usr/bin/env python3
"""
play
"""
import numpy as np

def play(env, Q, max_steps=100):
    """
    Function that has the trained agent play an episode

    Args:
        env: FrozenLakeEnv instance
        Q: numpy.ndarray containing Q-table
        max_steps: maximum number of steps in the episode

    Returns:
        Total rewards for the episode and rendered outputs
    """
    state = env.reset()[0]  # Adjust for Gymnasium's new API
    total_rewards = 0
    rendered_outputs = []  # List to track rendered outputs

    for step in range(max_steps):
        # Render the current state and capture the output
        rendered_output = env.render()  
        rendered_outputs.append(rendered_output)  # Store the rendered output
        
        # Select the best action based on the Q-table
        action = np.argmax(Q[state])
        
        # Execute the action and observe the next state and reward
        new_state, reward, terminated, truncated, _ = env.step(action)
        
        # Accumulate the rewards
        total_rewards += reward
        
        # Check if the episode has ended
        if terminated or truncated:
            # Render the final state before breaking
            rendered_output = env.render()
            rendered_outputs.append(rendered_output)  # Store the last rendered output
            break
        
        # Move to the next state
        state = new_state

    env.close()  # Properly close the environment
    return total_rewards, rendered_outputs  # Return rewards and rendered outputs
