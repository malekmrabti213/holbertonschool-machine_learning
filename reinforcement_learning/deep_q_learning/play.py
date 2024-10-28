#!/usr/bin/env python3

from keras import layers
from keras import __version__
import tensorflow as tf
tf.keras.__version__ = __version__
try:
    from rl.processors import Processor
except:
    tf.keras.__version__ = __version__
    from rl.processors import Processor

import cv2
import numpy as np
import keras

from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory

import gymnasium as gym


class AtariProcessor(Processor):
    """Preprocessing Images"""
    def process_observation(self, observation):
        if isinstance(observation, tuple):
            observation = observation[0]
        # Ensure it's a NumPy array
        observation = np.array(observation)
        img = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (84, 84))
        return img

    def process_state_batch(self, batch):
        """Rescale the images"""
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        """Clip the rewards between -1 and 1"""
        return np.clip(reward, -1., 1.)
    
def create_q_model(actions=4, window=4):
    """Create the model for the agent"""

    inputs = layers.Input(shape=(window, 84, 84))
    x = layers.Permute((2, 3, 1))(inputs)

    x = layers.Conv2D(32, 8, strides=4, activation="relu",data_format="channels_last")(x)
    x = layers.Conv2D(64, 4, strides=2, activation="relu",data_format="channels_last")(x)
    x = layers.Conv2D(64, 3, strides=1, activation="relu",data_format="channels_last")(x)

    x = layers.Flatten()(x)

    x = layers.Dense(512, activation="relu")(x)
    action = layers.Dense(actions, activation="linear")(x)

    return keras.Model(inputs=inputs, outputs=action)

class CompatibilityWrapper(gym.Wrapper):
    def step(self, action):
        # Call the step method of the underlying environment
        observation, reward, terminated, truncated, info = self.env.step(action)
        # Combine terminated and truncated into a single boolean 'done'
        done = terminated or truncated
        # Return the format expected by the old API
        return observation, reward, done, info

    def reset(self, **kwargs):
        # Call the reset method of the underlying environment
        observation, info = self.env.reset(**kwargs)
        # Return only the observation to match the old format
        return observation

    def render(self, *args, **kwargs):
        # Call the environment's render method without any arguments
        return self.env.render()  # Removes 'mode' argument
    
# Create and wrap the Breakout environment
env = gym.make("Breakout-v4", render_mode="human")
env = CompatibilityWrapper(env)
env.reset()

# Create the model and load the trained weights
model = create_q_model(actions=4)
model.load_weights('policy.h5')

# Create the agent with GreedyQPolicy
memory = SequentialMemory(limit=100000, window_length=4)

policy = GreedyQPolicy()

agent = DQNAgent(
    model=model,
    nb_actions=4,
    policy=policy,
    memory=memory,
    processor=AtariProcessor(),
    gamma=.99,
    train_interval=4,
    delta_clip=1.
)

agent.compile(keras.optimizers.legacy.Adam(lr=0.0001), metrics=['mae'])

# Play the game using the trained agent
agent.test(env, nb_episodes=2, visualize=True)
